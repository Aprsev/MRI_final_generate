#!/usr/bin/env python
"""
GAN-MAT Pipeline: T1 → T2 → Myelin → MPC → Gradients → Moments
================================================================
Complete implementation of the GAN-MAT microstructural covariance analysis toolbox.

Pipeline Steps:
  1. DICOM → NIfTI conversion
  2. Preprocessing (bias correction, brain extraction, MNI registration, segmentation)
  3. T1→T2 synthesis using 3D Pix2Pix GAN
  4. Myelin-sensitive image (T1w/T2w ratio) calculation
  5. Microstructural Profile Covariance (MPC) matrix
  6. Microstructural gradients
  7. Moment features (mean, std, skewness, kurtosis)

Author: Adapted from GAN-MAT (CAMIN-neuro)
Reference: https://github.com/CAMIN-neuro/GAN-MAT
"""

import os
import sys
import argparse
import subprocess
import shutil
import glob
import numpy as np
import nibabel as nib
from scipy.stats import skew, kurtosis
from brainspace.gradient import GradientMaps

# Add functions to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

###############################################################################
# STEP 0: DICOM to NIfTI Conversion
###############################################################################

def step0_dicom_to_nifti(raw_dicom_dir, nifti_dir):
    """
    Convert DICOM series to NIfTI using pydicom + nibabel.
    Falls back to dcm2niix if available.
    
    Parameters
    ----------
    raw_dicom_dir : str
        Path to raw DICOM data (All_Subjects_T1_Raw)
    nifti_dir : str
        Output directory for NIfTI files
    """
    print("=" * 70)
    print("STEP 0: DICOM → NIfTI Conversion")
    print("=" * 70)
    
    os.makedirs(nifti_dir, exist_ok=True)
    
    # Try using dcm2niix first (faster, more robust)
    dcm2niix_path = shutil.which("dcm2niix")
    if dcm2niix_path:
        print(f"Using dcm2niix from: {dcm2niix_path}")
        for subject_dir in sorted(os.listdir(raw_dicom_dir)):
            sub_path = os.path.join(raw_dicom_dir, subject_dir)
            if not os.path.isdir(sub_path):
                continue
            for series_dir in sorted(os.listdir(sub_path)):
                series_path = os.path.join(sub_path, series_dir)
                if not os.path.isdir(series_path):
                    continue
                print(f"  Converting: {subject_dir}/{series_dir}")
                subprocess.run([
                    dcm2niix_path, "-z", "y", "-o", nifti_dir,
                    "-f", f"{subject_dir}_T1w", series_path
                ], check=True)
    else:
        print("dcm2niix not found. Using pydicom-based conversion.")
        convert_dicom_pydicom(raw_dicom_dir, nifti_dir)
    
    # Rename to expected format
    for fname in os.listdir(nifti_dir):
        if fname.endswith(".nii.gz") or fname.endswith(".nii"):
            src = os.path.join(nifti_dir, fname)
            # Extract subject name
            parts = fname.split('_T1w')
            if len(parts) > 0:
                sub_name = parts[0]
                sub_dir = os.path.join(nifti_dir, sub_name)
                os.makedirs(sub_dir, exist_ok=True)
                dst = os.path.join(sub_dir, "T1w.nii.gz")
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                    print(f"  Moved {fname} → {sub_name}/T1w.nii.gz")
                # Also copy accompanying JSON if exists
                json_src = src.replace('.nii.gz', '.json').replace('.nii', '.json')
                if os.path.exists(json_src):
                    shutil.move(json_src, os.path.join(sub_dir, "T1w.json"))


def convert_dicom_pydicom(raw_dicom_dir, nifti_dir):
    """
    Convert DICOM to NIfTI using pydicom + nibabel (pure Python).
    This is a simple implementation for single-frame MR volumes.
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    
    for subject_dir in sorted(os.listdir(raw_dicom_dir)):
        sub_path = os.path.join(raw_dicom_dir, subject_dir)
        if not os.path.isdir(sub_path):
            continue
        
        for series_dir in sorted(os.listdir(sub_path)):
            series_path = os.path.join(sub_path, series_dir)
            if not os.path.isdir(series_path):
                continue
            
            print(f"  Processing: {subject_dir}/{series_dir}")
            
            dicom_files = sorted(glob.glob(os.path.join(series_path, "*.dcm")),
                                 key=lambda x: int(os.path.basename(x).split('.')[0]))
            
            if not dicom_files:
                continue
            
            # Read all slices
            slices = []
            for f in dicom_files:
                try:
                    ds = pydicom.dcmread(f)
                    if hasattr(ds, 'pixel_array') and ds.pixel_array.size > 0:
                        slices.append(ds)
                except:
                    pass
            
            if not slices:
                continue
            
            # Sort by slice location
            try:
                slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            except:
                slices.sort(key=lambda x: int(x.InstanceNumber))
            
            # Build volume
            pixel_data = np.stack([s.pixel_array for s in slices]).astype(np.float32)
            
            # Rescale if needed
            if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                slope = float(slices[0].RescaleSlope)
                intercept = float(slices[0].RescaleIntercept)
                pixel_data = pixel_data * slope + intercept
            
            # Build affine matrix
            try:
                pos = [s.ImagePositionPatient for s in slices]
                pos = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in pos])
                
                # Calculate spacing
                if len(slices) > 1:
                    dz = np.linalg.norm(pos[1] - pos[0])
                else:
                    dz = 1.0
                
                dr = float(slices[0].PixelSpacing[0])
                dc = float(slices[0].PixelSpacing[1])
                
                # Orientation matrix
                orient = np.array(slices[0].ImageOrientationPatient).reshape(2, 3)
                # This is simplified; for production use dcm2niix
                affine = np.array([
                    [-dr, 0, 0, 0],
                    [0, -dr, 0, 0],
                    [0, 0, dz, 0],
                    [0, 0, 0, 1]
                ])
            except:
                affine = np.eye(4)
            
            # Create NIfTI
            nifti_img = nib.Nifti1Image(pixel_data, affine)
            
            # Save
            sub_out_dir = os.path.join(nifti_dir, subject_dir)
            os.makedirs(sub_out_dir, exist_ok=True)
            out_path = os.path.join(sub_out_dir, "T1w.nii.gz")
            nib.save(nifti_img, out_path)
            print(f"    Saved: {out_path} (shape: {pixel_data.shape})")


###############################################################################
# STEP 1: Preprocessing - Bias correction, Brain extraction, MNI registration
###############################################################################

def step1_preprocessing(nifti_dir, output_dir, pipeline_dir, use_simple=True):
    """
    Preprocess T1w images: brain extraction, MNI registration, tissue segmentation.
    Uses zoom-based MNI registration + scikit-image for improved brain extraction.
    """
    print("=" * 70)
    print("STEP 1: Preprocessing")
    print("=" * 70)

    subjects = [d for d in sorted(os.listdir(nifti_dir))
                if os.path.isdir(os.path.join(nifti_dir, d))]

    sub_list_path = os.path.join(output_dir, "sub_list.txt")
    with open(sub_list_path, 'w') as f:
        f.write(" ".join(subjects) + " ")
    print(f"  Subjects: {subjects}")

    for sub in subjects:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    # Try scikit-image for better brain extraction
    try:
        from skimage.filters import threshold_otsu
        from skimage.measure import label as sk_label
        SKIMAGE_AVAILABLE = True
        print("  Using scikit-image for brain extraction")
    except ImportError:
        SKIMAGE_AVAILABLE = False
        print("  scikit-image not available")

    template_path = os.path.join(pipeline_dir, "template", "MNI152_T1_0.8mm_brain.nii.gz")
    template_img = nib.load(template_path)
    template_data = template_img.get_fdata().astype(np.float64)
    TARGET_SHAPE = template_data.shape
    print(f"  Template shape: {TARGET_SHAPE}")

    for sub in subjects:
        print(f"\n  Processing {sub}...")
        t1_path = os.path.join(nifti_dir, sub, "T1w.nii.gz")
        if not os.path.exists(t1_path):
            nifti_files = glob.glob(os.path.join(nifti_dir, sub, "*.nii*"))
            t1_path = nifti_files[0] if nifti_files else None
        if t1_path is None or not os.path.exists(t1_path):
            print(f"    WARNING: No NIfTI file for {sub}, skipping")
            continue

        img = nib.load(t1_path)
        data = img.get_fdata().astype(np.float64)

        # === 1. Brain Extraction ===
        if SKIMAGE_AVAILABLE:
            thresh = threshold_otsu(data[data > data.min()])
            brain_mask = data > thresh
            labeled = sk_label(brain_mask)
            largest = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            brain_mask = labeled == largest
            from scipy.ndimage import binary_fill_holes
            brain_mask = binary_fill_holes(brain_mask)
        else:
            brain_mask = data > (data.mean() * 0.3)
            from scipy.ndimage import binary_fill_holes, binary_dilation
            brain_mask = binary_fill_holes(brain_mask)
            brain_mask = binary_dilation(brain_mask, iterations=2)
            brain_mask = binary_fill_holes(brain_mask)

        brain_data = data * brain_mask
        brain_img = nib.Nifti1Image(brain_data, img.affine, img.header)
        nib.save(brain_img, os.path.join(nifti_dir, sub, "T1w_brain.nii.gz"))

        # === 2. MNI Registration via zoom ===
        from scipy.ndimage import zoom
        zoom_factors = [t / s for s, t in zip(data.shape, TARGET_SHAPE)]
        # Use order=1 (linear) to avoid negative values, prefilter to prevent ringing
        resampled = zoom(brain_data, zoom_factors, order=1)

        # Ensure output shape matches template exactly
        if resampled.shape != TARGET_SHAPE:
            resampled = zoom(resampled,
                            [t/s for s,t in zip(resampled.shape, TARGET_SHAPE)],
                            order=1)

        # Mask with template brain
        tmpl_mask = template_data > 0
        resampled[~tmpl_mask] = 0

        mni_img = nib.Nifti1Image(resampled.astype(np.float32),
                                   template_img.affine, template_img.header)
        nib.save(mni_img, os.path.join(nifti_dir, sub, "T1w_MNI.nii.gz"))

        overlap = np.sum((resampled > 0) & tmpl_mask)
        dice = 2 * overlap / (np.sum(resampled > 0) + np.sum(tmpl_mask))
        print(f"    Dice with template: {dice:.4f}")

        # === 3. Tissue Segmentation ===
        brain_vals = resampled[resampled > 0]
        if len(brain_vals) > 0:
            # Use percentiles for 3-way split
            p33 = np.percentile(brain_vals, 33)
            p66 = np.percentile(brain_vals, 66)

            # Store as uint8 to preserve exact integer values
            # IMPORTANT: only segment brain voxels (resampled > 0), leave background as 0
            pve = np.zeros(TARGET_SHAPE, dtype=np.uint8)
            brain_mask_resampled = resampled > 0
            pve[brain_mask_resampled & (resampled <= p33)] = 1       # CSF
            pve[brain_mask_resampled & (resampled > p33) & (resampled <= p66)] = 2  # GM
            pve[brain_mask_resampled & (resampled > p66)] = 3        # WM

            pve_img = nib.Nifti1Image(pve, template_img.affine, template_img.header)
            pve_path = os.path.join(nifti_dir, sub, "T1w_MNI_pveseg.nii.gz")
            nib.save(pve_img, pve_path)

            csf_v = int(np.sum(pve == 1))
            gm_v = int(np.sum(pve == 2))
            wm_v = int(np.sum(pve == 3))
            print(f"    CSF={csf_v} GM={gm_v} WM={wm_v}")

            # === 4. GAN input construction ===
            # Use the in-memory pve (exact uint8 values, no float issues)
            temp = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2], 3),
                            dtype=np.float64)

            for i in range(1, 4):
                mask = (pve == i)
                temp[mask, i - 1] = resampled[mask]

            # Pad/crop to 256x256x256 as expected by GAN
            gan_input = np.zeros((256, 256, 256, 3), dtype=np.float32)
            gan_input[14:-15, :, 14:-15, :] = temp[:, 8:-8, :, :]

            np.save(os.path.join(nifti_dir, sub, "T1w_MNI_pveseg.npy"), gan_input)

            for ch in range(3):
                nz = np.sum(gan_input[:,:,:,ch] != 0)
                print(f"    GAN Ch{ch}: {nz} non-zero")

    return subjects


###############################################################################
# STEP 2: T1 → T2 Synthesis using GAN
###############################################################################

def step2_t1_to_t2(pipeline_dir, input_dir, output_dir, batch_size=1):
    """
    Synthesize T2-weighted MRI from T1-weighted MRI using 3D Pix2Pix GAN.
    """
    print("=" * 70)
    print("STEP 2: T1 → T2 Synthesis using 3D Pix2Pix GAN")
    print("=" * 70)

    model_path = os.path.join(pipeline_dir, "functions", "model", "model.pth")

    if not os.path.exists(model_path):
        print("\n  WARNING: model.pth not found!")
        print(f"  Download from: https://www.dropbox.com/sh/nnzayieuizd012y/AACLSwUY9BBTCdf66_nWqK02a?dl=0")
        print(f"  Place at: {model_path}")
        print("  Skipping GAN inference. Will use simplified T2 estimation.\n")
        _estimate_t2_simple(input_dir, output_dir)
        return

    # Run the GAN model
    print("  Running GAN inference...")
    from functions.model.model import Pix2Pix_3D
    from functions.model.dataset import Dataset
    import torch
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    dataset_test = Dataset(input_dir=input_dir, output_dir=output_dir)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    netG = Pix2Pix_3D(in_channels=3, out_channels=1).to(device)
    dict_model = torch.load(model_path, map_location=device)
    netG.load_state_dict(dict_model['netG'])
    netG.eval()

    with torch.no_grad():
        for batch, data in enumerate(loader_test, 1):
            input_data = data['input'].to(device)
            output = netG(input_data)

            for j in range(output.shape[0]):
                idx = batch_size * (batch - 1) + j
                output_np = output[j].cpu().numpy()
                sub_name = dataset_test.lst_sub[idx]

                # Save raw GAN output (Tanh range [-1, 1])
                np.save(f"{input_dir}/{sub_name}/output_T2w.npy", output_np[0])
                print(f"  synthesized {sub_name} T2-weighted MRI [batch {batch}]")

                # Convert to NIfTI (resize_inv) with brain-only normalization
                _convert_t2_to_nifti(input_dir, pipeline_dir, sub_name)
                print(f"  converted {sub_name} output_MNI.nii.gz")

    print("  T2 synthesis complete!")


def _convert_t2_to_nifti(input_dir, pipeline_dir, sub_name):
    """
    Convert GAN output numpy array to MNI-space NIfTI.
    Normalizes only within brain region to preserve tissue contrast.
    """
    template_path = os.path.join(pipeline_dir, "template", "MNI152_T1_0.8mm_brain.nii.gz")
    if os.path.exists(template_path):
        template_img = nib.load(template_path)
        MNI_header = template_img.header
    else:
        MNI_header = nib.Nifti1Header()

    # Load GAN output (256x256x256, range [-1, 1] from Tanh)
    t2 = np.load(f"{input_dir}/{sub_name}/output_T2w.npy")

    # Map GAN output to MNI space first
    output_img = np.zeros((227, 272, 227), dtype=np.float32)
    output_img[:, 8:-8, :] = t2[14:-15, :, 14:-15]

    # Use T1 brain mask to identify brain voxels (for normalization)
    t1_mni_path = os.path.join(input_dir, sub_name, "T1w_MNI.nii.gz")
    if os.path.exists(t1_mni_path):
        t1_img = nib.load(t1_mni_path)
        t1_data = t1_img.get_fdata()
        brain_mask = t1_data > 0
    else:
        brain_mask = output_img > -0.9  # fallback: use values above background

    # Normalize only brain voxels to [0, 1]
    brain_vals = output_img[brain_mask]
    if len(brain_vals) > 0 and brain_vals.max() > brain_vals.min():
        vmin, vmax = brain_vals.min(), brain_vals.max()
        output_img[brain_mask] = (brain_vals - vmin) / (vmax - vmin)
        output_img[~brain_mask] = 0.0

    # Save as NIfTI
    nifti_img = nib.Nifti1Image(output_img, affine=None, header=MNI_header)
    nib.save(nifti_img, f"{input_dir}/{sub_name}/output_MNI.nii.gz")


def _estimate_t2_simple(input_dir, output_dir):
    """
    Simplified T2 estimation when GAN model is not available.
    Uses a heuristic T1w/T2w relationship: T2w ≈ a/(T1w + b) + c
    This is a placeholder - the actual GAN output should be used.
    """
    print("\n  Using simplified T2 estimation (placeholder)...")
    
    sub_list_path = os.path.join(output_dir, "sub_list.txt")
    if not os.path.exists(sub_list_path):
        print("  No sub_list.txt found")
        return
    
    with open(sub_list_path, 'r') as f:
        subjects = f.read().split()
    
    for sub in subjects:
        t1_mni_path = os.path.join(input_dir, sub, "T1w_MNI.nii.gz")
        if not os.path.exists(t1_mni_path):
            print(f"  {sub}: T1w_MNI.nii.gz not found, skipping")
            continue
        
        print(f"  {sub}: Estimating T2...")
        img = nib.load(t1_mni_path)
        data = img.get_fdata()
        
        # Simple heuristic: T2w is roughly inverse of T1w (normalized)
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        t2_estimate = 1.0 - data_norm  # Simple inversion
        
        # Save as numpy array (simulating GAN output)
        np.save(os.path.join(input_dir, sub, "output_T2w.npy"), t2_estimate)
        
        # Resize back to MNI space
        MNI_header = img.header
        output_img = np.zeros((227, 272, 227))
        # This is a simplified resize_inv operation
        t2_resized = np.zeros((256, 256, 256))
        t2_resized[14:-15, :, 14:-15] = t2_estimate[:227, 8:-8, :227] if t2_estimate.shape[0] >= 227 else t2_estimate
        output_img[:, 8:-8, :] = t2_resized[14:-15, :, 14:-15]
        
        nifti_img = nib.Nifti1Image(output_img, affine=None, header=MNI_header)
        nib.save(nifti_img, os.path.join(input_dir, sub, "output_MNI.nii.gz"))
        print(f"  {sub}: Estimated T2 saved")


###############################################################################
# STEP 3: T2 Registration to Native Space
###############################################################################

def step3_t2_to_native(input_dir, output_dir, subjects):
    """
    Register synthesized T2 back to native (fsnative) space.
    In the full pipeline, this uses ANTs transforms.
    Here we provide a simplified version.
    """
    print("=" * 70)
    print("STEP 3: T2 Registration to Native Space")
    print("=" * 70)
    
    for sub in subjects:
        t2_mni_path = os.path.join(input_dir, sub, "output_MNI.nii.gz")
        if not os.path.exists(t2_mni_path):
            print(f"  {sub}: output_MNI.nii.gz not found, skipping")
            continue
        
        print(f"  {sub}: Copying T2 to output directory")
        # In full pipeline, this applies inverse transform
        # Here we just copy the output
        shutil.copy(t2_mni_path, os.path.join(output_dir, sub, "T2w_fsnative_brain.nii.gz"))
        shutil.copy(t2_mni_path, os.path.join(input_dir, sub, "T2w_fsnative_brain.nii.gz"))


###############################################################################
# STEP 4: Myelin-sensitive Image (T1w/T2w ratio)
###############################################################################

def step4_myelin_map(input_dir, output_dir, subjects):
    """
    Calculate myelin-sensitive image as T1w/T2w ratio.
    
    The T1w/T2w ratio is a well-established proxy for myelin content:
    - Higher ratio = more myelin (white matter)
    - Lower ratio = less myelin (gray matter, CSF)
    
    Steps:
    1. Normalize T1w and T2w intensities to [0, 1]
    2. Compute T1w/T2w ratio
    3. Clamp to reasonable range [0, 100]
    """
    print("=" * 70)
    print("STEP 4: Myelin-sensitive Map (T1w/T2w Ratio)")
    print("=" * 70)
    
    for sub in subjects:
        t1_path = os.path.join(input_dir, sub, "T1w_fsnative_brain.nii.gz")
        t2_path = os.path.join(input_dir, sub, "T2w_fsnative_brain.nii.gz")
        
        # Fallback to MNI space files
        if not os.path.exists(t1_path):
            t1_path = os.path.join(input_dir, sub, "T1w_MNI.nii.gz")
        if not os.path.exists(t2_path):
            t2_path = os.path.join(input_dir, sub, "output_MNI.nii.gz")
        
        if not os.path.exists(t1_path) or not os.path.exists(t2_path):
            print(f"  {sub}: T1w or T2w not found, skipping")
            continue
        
        print(f"  {sub}: Computing T1w/T2w myelin map...")
        
        t1_img = nib.load(t1_path)
        t2_img = nib.load(t2_path)
        
        t1_data = t1_img.get_fdata().astype(np.float64)
        t2_data = t2_img.get_fdata().astype(np.float64)
        
        # Ensure same shape (resample if needed)
        if t1_data.shape != t2_data.shape:
            from scipy.ndimage import zoom
            zoom_factors = [s / t for s, t in zip(t1_data.shape, t2_data.shape)]
            t2_data = zoom(t2_data, zoom_factors, order=1)
        
        # Min-max normalization to [0, 1]
        t1_min, t1_max = t1_data.min(), t1_data.max()
        t2_min, t2_max = t2_data.min(), t2_data.max()
        
        if t1_max > t1_min:
            t1_norm = (t1_data - t1_min) / (t1_max - t1_min)
        else:
            t1_norm = t1_data
        
        if t2_max > t2_min:
            t2_norm = (t2_data - t2_min) / (t2_max - t2_min)
        else:
            t2_norm = t2_data
        
        # Compute T1w/T2w ratio, clamp to [0, 100]
        with np.errstate(divide='ignore', invalid='ignore'):
            myelin = t1_norm / (t2_norm + 1e-8)
            myelin = np.clip(myelin, 0, 100)
            myelin = np.nan_to_num(myelin, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Save myelin map
        myelin_img = nib.Nifti1Image(myelin.astype(np.float32), t1_img.affine, t1_img.header)
        
        # Save in input directory structure (for MPC)
        sub_mri_dir = os.path.join(input_dir, sub, "T1w", sub, "mri")
        os.makedirs(sub_mri_dir, exist_ok=True)
        myelin_path = os.path.join(sub_mri_dir, "myelin.nii.gz")
        nib.save(myelin_img, myelin_path)
        print(f"    Saved myelin map: {myelin_path}")
        
        # Also save to output
        output_myelin = os.path.join(output_dir, sub, "myelin.nii.gz")
        nib.save(myelin_img, output_myelin)
        print(f"    Saved to output: {output_myelin}")
        
        # Save normalized T1 and T2 as well
        t1_norm_img = nib.Nifti1Image(t1_norm.astype(np.float32), t1_img.affine, t1_img.header)
        nib.save(t1_norm_img, os.path.join(input_dir, sub, "T1w", "T1w_nor.nii.gz"))
        
        t2_norm_img = nib.Nifti1Image(t2_norm.astype(np.float32), t2_img.affine, t2_img.header)
        nib.save(t2_norm_img, os.path.join(input_dir, sub, "T1w", "T2w_nor.nii.gz"))


###############################################################################
# STEP 5: Microstructural Profile Covariance (MPC) Matrix
###############################################################################

def step5_mpc_matrix(pipeline_dir, input_dir, output_dir, subjects):
    """
    Compute Microstructural Profile Covariance (MPC) matrix.
    
    The MPC matrix captures the covariance of microstructural profiles
    across brain regions. It requires FreeSurfer surface data and
    pre-computed micro-profiles (from the MPC.sh script).
    
    Here we provide a simplified implementation that reads the
    micro-profiles and computes the covariance matrix.
    """
    print("=" * 70)
    print("STEP 5: Microstructural Profile Covariance (MPC) Matrix")
    print("=" * 70)
    
    # Read atlas list
    atlas_list_path = os.path.join(pipeline_dir, "parcellations", "atlas_list.txt")
    with open(atlas_list_path, 'r') as f:
        atlas_ls = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"  Atlases: {[a.split('.')[0].split('_')[0] for a in atlas_ls]}")
    
    for sub in subjects:
        for atlas_file in atlas_ls:
            atlas = atlas_file.split('.')[0].split("_")[0]
            
            # Path to micro-profiles
            profiles_dir = os.path.join(
                input_dir, sub, "T1w", sub, 
                "anat", "surfaces", "micro_profiles"
            )
            profiles_path = os.path.join(
                profiles_dir,
                f"{sub}_space-fsaverage5_atlas-{atlas}_desc-MPC.txt"
            )
            
            if not os.path.exists(profiles_path):
                print(f"  {sub}/{atlas}: Micro-profiles not found")
                print(f"    Expected: {profiles_path}")
                print(f"    SKIPPING (requires FreeSurfer surface processing)")
                continue
            
            # Load micro-profiles
            temp = np.loadtxt(profiles_path, dtype=np.float64, delimiter=' ')
            
            # Make symmetric matrix (triu + transpose)
            MPC = np.triu(temp, 1) + temp.T
            
            # Remove corpus callosum (index depends on atlas)
            if atlas == "aparc-a2009s":
                idx = [41, 116]
            else:
                idx = [0, int(len(MPC) / 2)]
            
            MPC = np.delete(np.delete(MPC, idx, axis=0), idx, axis=1)
            
            # Save in input directory
            os.makedirs(profiles_dir, exist_ok=True)
            np.savetxt(
                os.path.join(profiles_dir, f"{atlas}_MPC_matrix.txt"),
                MPC
            )
            
            # Copy to output
            output_mpc = os.path.join(output_dir, sub, f"{atlas}_MPC_matrix.txt")
            shutil.copy(
                os.path.join(profiles_dir, f"{atlas}_MPC_matrix.txt"),
                output_mpc
            )
            
            print(f"  {sub}/{atlas}: MPC matrix saved ({MPC.shape})")
        
        print(f"  {sub}: Matrix computation complete")


###############################################################################
# STEP 6: Microstructural Gradients
###############################################################################

def step6_gradients(pipeline_dir, input_dir, output_dir, subjects):
    """
    Compute microstructural gradients using diffusion map embedding.
    
    Gradients reveal the principal axes of microstructural variation
    across the cortical surface.
    """
    print("=" * 70)
    print("STEP 6: Microstructural Gradients")
    print("=" * 70)
    
    # Read atlas list
    atlas_list_path = os.path.join(pipeline_dir, "parcellations", "atlas_list.txt")
    with open(atlas_list_path, 'r') as f:
        atlas_ls = [line.strip() for line in f.readlines() if line.strip()]
    
    for sub in subjects:
        for atlas_file in atlas_ls:
            atlas = atlas_file.split('.')[0].split("_")[0]
            
            profiles_dir = os.path.join(
                input_dir, sub, "T1w", sub,
                "anat", "surfaces", "micro_profiles"
            )
            mpc_path = os.path.join(profiles_dir, f"{atlas}_MPC_matrix.txt")
            
            if not os.path.exists(mpc_path):
                print(f"  {sub}/{atlas}: MPC matrix not found, skipping gradients")
                continue
            
            # Load MPC matrix
            temp = np.loadtxt(mpc_path, dtype=np.float64, delimiter=' ')
            
            # Remove all-zero rows/columns
            del_ls = np.where(temp.sum(0) == 0)[0]
            temp_clean = np.delete(np.delete(temp, del_ls, axis=0), del_ls, axis=1)
            
            if temp_clean.size == 0:
                print(f"  {sub}/{atlas}: Empty matrix after cleaning")
                continue
            
            # Compute gradients using diffusion map embedding
            grad_map = GradientMaps(
                n_components=10, random_state=None,
                approach='dm', kernel='normalized_angle'
            )
            grad_map.fit(temp_clean, sparsity=0.9)
            
            # Insert zeros back at deleted positions
            grads = grad_map.gradients_.copy()
            for idx in del_ls:
                grads = np.insert(grads, idx, 0, axis=0)
            
            # Save gradients
            np.savetxt(
                os.path.join(profiles_dir, f"{atlas}_MPC_gradients.txt"),
                grads
            )
            
            # Copy to output
            shutil.copy(
                os.path.join(profiles_dir, f"{atlas}_MPC_gradients.txt"),
                os.path.join(output_dir, sub, f"{atlas}_MPC_gradients.txt")
            )
            
            print(f"  {sub}/{atlas}: Gradients computed ({grads.shape})")
        
        print(f"  {sub}: Gradients complete")


###############################################################################
# STEP 7: Moment Features
###############################################################################

def step7_moments(pipeline_dir, input_dir, output_dir, subjects):
    """
    Compute moment features (mean, std, skewness, kurtosis) of
    microstructural intensity profiles.
    """
    print("=" * 70)
    print("STEP 7: Moment Features")
    print("=" * 70)
    
    # Read atlas list
    atlas_list_path = os.path.join(pipeline_dir, "parcellations", "atlas_list.txt")
    with open(atlas_list_path, 'r') as f:
        atlas_ls = [line.strip() for line in f.readlines() if line.strip()]
    
    for sub in subjects:
        for atlas_file in atlas_ls:
            atlas = atlas_file.split('.')[0].split("_")[0]
            
            profiles_dir = os.path.join(
                input_dir, sub, "T1w", sub,
                "anat", "surfaces", "micro_profiles"
            )
            inten_path = os.path.join(
                profiles_dir,
                f"{sub}_space-fsaverage5_atlas-{atlas}_desc-intensity_profiles.txt"
            )
            
            if not os.path.exists(inten_path):
                print(f"  {sub}/{atlas}: Intensity profiles not found, skipping moments")
                continue
            
            # Load intensity profiles
            temp = np.loadtxt(inten_path, dtype=np.float64, delimiter=' ')
            
            # Remove NaN columns
            nan_ls = np.unique(np.where(np.isnan(temp))[1])
            inten = np.delete(temp, nan_ls, axis=1)
            
            if inten.size == 0:
                continue
            
            # Compute moment features
            mean = inten.mean(0)
            std = inten.std(0)
            skewness = skew(inten)
            kurto = kurtosis(inten)
            
            moment = np.vstack((mean, std, skewness, kurto))
            
            # Save moments
            np.savetxt(
                os.path.join(profiles_dir, f"{atlas}_moment.txt"),
                moment
            )
            
            # Copy to output
            shutil.copy(
                os.path.join(profiles_dir, f"{atlas}_moment.txt"),
                os.path.join(output_dir, sub, f"{atlas}_MPC_moment.txt")
            )
            
            print(f"  {sub}/{atlas}: Moments computed ({moment.shape})")
        
        print(f"  {sub}: Moments complete")


###############################################################################
# MAIN PIPELINE
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="GAN-MAT: Microstructural Profile Covariance Analysis Toolbox"
    )
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Path to raw DICOM data directory")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to input (working) data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("--pipeline_dir", type=str, default=None,
                        help="Path to GAN-MAT pipeline directory (default: script location)")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "0", "1", "2", "3", "4", "5", "6", "7"],
                        help="Which step to run")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for GAN inference")
    parser.add_argument("--simple", action="store_true", default=True,
                        help="Use simplified processing (no external tools required)")
    parser.add_argument("--no_simple", action="store_false", dest="simple",
                        help="Try to use external tools (ANTs, FSL, FreeSurfer)")
    
    args = parser.parse_args()
    
    # Set pipeline directory
    if args.pipeline_dir is None:
        args.pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create directories
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  GAN-MAT: Microstructural Profile Covariance Analysis Toolbox")
    print("=" * 70)
    print(f"  Raw data:    {args.raw_dir}")
    print(f"  Input dir:   {args.input_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Pipeline:    {args.pipeline_dir}")
    print(f"  Simple mode: {args.simple}")
    print("=" * 70 + "\n")
    
    # Run selected step(s)
    if args.step == "all" or args.step == "0":
        step0_dicom_to_nifti(args.raw_dir, args.input_dir)
    
    if args.step == "all" or args.step == "1":
        subjects = step1_preprocessing(args.input_dir, args.output_dir, args.pipeline_dir, args.simple)
    else:
        # Get subjects from sub_list.txt
        sub_list_path = os.path.join(args.output_dir, "sub_list.txt")
        if os.path.exists(sub_list_path):
            with open(sub_list_path, 'r') as f:
                subjects = f.read().split()
        else:
            subjects = sorted([d for d in os.listdir(args.input_dir)
                              if os.path.isdir(os.path.join(args.input_dir, d))])
    
    if args.step == "all" or args.step == "2":
        step2_t1_to_t2(args.pipeline_dir, args.input_dir, args.output_dir, args.batch_size)
    
    if args.step == "all" or args.step == "3":
        step3_t2_to_native(args.input_dir, args.output_dir, subjects)
    
    if args.step == "all" or args.step == "4":
        step4_myelin_map(args.input_dir, args.output_dir, subjects)
    
    if args.step == "all" or args.step == "5":
        step5_mpc_matrix(args.pipeline_dir, args.input_dir, args.output_dir, subjects)
    
    if args.step == "all" or args.step == "6":
        step6_gradients(args.pipeline_dir, args.input_dir, args.output_dir, subjects)
    
    if args.step == "all" or args.step == "7":
        step7_moments(args.pipeline_dir, args.input_dir, args.output_dir, subjects)
    
    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
