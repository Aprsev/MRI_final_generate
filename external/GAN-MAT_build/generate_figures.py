#!/usr/bin/env python
"""
Generate visualization figures for the GAN-MAT pipeline documentation.
Creates images showing the processing results at each stage.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob

# Input directories
BASE_DIR = r"D:\Desktop\ZJU\grade3\25-26spring\磁共振成像原理及应用\Labatory\final\GAN-MAT_build"
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGS_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# Pick a representative subject
SUBJECT = "mricourse_child1_20240803"


def load_nifti(path):
    """Load NIfTI file and return data + img."""
    if not os.path.exists(path):
        return None, None
    img = nib.load(path)
    data = img.get_fdata()
    return data, img


def mid_slice(data, axis=2):
    """Get middle slice along given axis."""
    idx = data.shape[axis] // 2
    if axis == 0:
        return data[idx, :, :]
    elif axis == 1:
        return data[:, idx, :]
    else:
        return data[:, :, idx]


def plot_comparison_slices(title, nifti_paths, labels, cmaps, ncols=4, save_name=None):
    """Create comparison figure with multiple slices."""
    n = len(nifti_paths)
    nrows = (n + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()
    
    for i in range(n):
        ax = axes[i]
        data, _ = load_nifti(nifti_paths[i])
        if data is None:
            ax.text(0.5, 0.5, "Not available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(labels[i] if i < len(labels) else f"Image {i+1}")
            continue
        
        # Get three orthogonal views
        s_axial = mid_slice(data, axis=2)
        s_coronal = mid_slice(data, axis=1)
        s_sagittal = mid_slice(data, axis=0)
        
        # Rotate for display
        s_axial = np.rot90(s_axial)
        s_coronal = np.rot90(s_coronal)
        s_sagittal = np.rot90(s_sagittal)
        
        ax.imshow(s_axial, cmap=cmaps[i] if i < len(cmaps) else 'gray', aspect='auto')
        ax.set_title(f"{labels[i] if i < len(labels) else ''}\n(Axial)", fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_name:
        plt.savefig(os.path.join(FIGS_DIR, save_name), dpi=150, bbox_inches='tight')
        print(f"Saved: {save_name}")
    plt.close()


def create_overview_figure():
    """Create overview figure showing the entire pipeline results for one subject."""
    sub_input = os.path.join(INPUT_DIR, SUBJECT)
    
    # Define files
    files = {
        "Original T1": os.path.join(sub_input, "T1w.nii.gz"),
        "Brain Extracted": os.path.join(sub_input, "T1w_brain.nii.gz"),
        "MNI Registered": os.path.join(sub_input, "T1w_MNI.nii.gz"),
        "Tissue Seg.": os.path.join(sub_input, "T1w_MNI_pveseg.nii.gz"),
        "Synthesized T2": os.path.join(sub_input, "output_MNI.nii.gz"),
        "Myelin Map": os.path.join(sub_input, "T1w", SUBJECT, "mri", "myelin.nii.gz"),
    }
    
    n = len(files)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['gray', 'gray', 'gray', 'viridis', 'gray', 'hot']
    
    for i, (label, path) in enumerate(files.items()):
        ax = axes[i]
        data, _ = load_nifti(path)
        if data is None:
            ax.text(0.5, 0.5, f"Not Found:\n{os.path.basename(path)}", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=9)
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.axis('off')
            continue
        
        # Mid axial slice
        s = mid_slice(data, axis=2)
        s = np.rot90(s)
        
        if label == "Tissue Seg.":
            # Custom colormap for segmentation
            seg_cmap = LinearSegmentedColormap.from_list('seg', 
                ['black', 'blue', 'green', 'red'], N=4)
            ax.imshow(s, cmap=seg_cmap, aspect='auto', vmin=0, vmax=3)
        else:
            ax.imshow(s, cmap=colors[i], aspect='auto')
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(f"GAN-MAT Pipeline Results for {SUBJECT}", 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "pipeline_overview.png"), dpi=150, bbox_inches='tight')
    print("Saved: pipeline_overview.png")
    plt.close()


def create_pipeline_diagram():
    """Create a pipeline flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define pipeline steps
    steps = [
        ("DICOM → NIfTI", 1, 4.5),
        ("Bias Correction\nBrain Extraction", 3, 4.5),
        ("MNI Registration\nTissue Segmentation", 5, 4.5),
        ("GAN T1→T2\nSynthesis", 7, 4.5),
        ("T2 → Native\nSpace", 9, 4.5),
        ("T1w/T2w Ratio\nMyelin Map", 11, 4.5),
        ("MPC Matrix\nGradients\nMoments", 13, 4.5),
    ]
    
    # Draw boxes and arrows
    for i, (text, x, y) in enumerate(steps):
        # Box
        bbox = dict(boxstyle="round,pad=0.3", facecolor="lightblue", 
                   edgecolor="steelblue", linewidth=2)
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
               fontweight='bold', bbox=bbox)
        
        # Arrow to next step
        if i < len(steps) - 1:
            ax.annotate('', xy=(steps[i+1][1] - 0.8, y), 
                       xytext=(x + 0.8, y),
                       arrowprops=dict(arrowstyle='->', color='steelblue', lw=2))
    
    # Input/Output labels
    ax.text(1, 2, "Raw DICOM\nT1-weighted\nMRI Scans", ha='center', va='center',
           fontsize=10, bbox=dict(boxstyle="round", facecolor='lightyellow', edgecolor='orange'))
    
    ax.text(13, 2, "Microstructural\nCovariance\nMatrix", ha='center', va='center',
           fontsize=10, bbox=dict(boxstyle="round", facecolor='lightgreen', edgecolor='green'))
    
    ax.text(15, 2, "Microstructural\nGradients\n& Moments", ha='center', va='center',
           fontsize=10, bbox=dict(boxstyle="round", facecolor='lightcoral', edgecolor='red'))
    
    # Arrows from I/O to pipeline
    ax.annotate('', xy=(steps[0][1] - 0.5, 3.5), xytext=(1, 2.8),
               arrowprops=dict(arrowstyle='->', color='orange', lw=2, linestyle='dashed'))
    ax.annotate('', xy=(13, 2.8), xytext=(steps[-2][1] + 0.5, 3.5),
               arrowprops=dict(arrowstyle='->', color='green', lw=2, linestyle='dashed'))
    ax.annotate('', xy=(14, 2), xytext=(13.5, 2),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.set_title("GAN-MAT Pipeline Flow", fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(FIGS_DIR, "pipeline_diagram.png"), dpi=150, bbox_inches='tight')
    print("Saved: pipeline_diagram.png")
    plt.close()


def create_myelin_histogram():
    """Create histogram of myelin values for one subject."""
    sub_input = os.path.join(INPUT_DIR, SUBJECT)
    myelin_path = os.path.join(sub_input, "T1w", SUBJECT, "mri", "myelin.nii.gz")
    
    data, _ = load_nifti(myelin_path)
    if data is None:
        print(f"Myelin map not found for {SUBJECT}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    valid = data[data > 0].flatten()
    ax.hist(valid, bins=100, color='darkred', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("T1w/T2w Ratio (Myelin Proxy)", fontsize=12)
    ax.set_ylabel("Voxel Count", fontsize=12)
    ax.set_title("Distribution of Myelin-sensitive Signal", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Myelin map mid-slice
    ax = axes[1]
    s = mid_slice(data, axis=2)
    s = np.rot90(s)
    im = ax.imshow(s, cmap='hot', aspect='auto')
    ax.set_title("Myelin Map (Axial View)", fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='T1w/T2w Ratio')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "myelin_analysis.png"), dpi=150, bbox_inches='tight')
    print("Saved: myelin_analysis.png")
    plt.close()


def create_t1_t2_comparison():
    """Create T1 vs T2 comparison figure."""
    sub_input = os.path.join(INPUT_DIR, SUBJECT)
    
    t1_path = os.path.join(sub_input, "T1w_MNI.nii.gz")
    t2_path = os.path.join(sub_input, "output_MNI.nii.gz")
    pve_path = os.path.join(sub_input, "T1w_MNI_pveseg.nii.gz")
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    
    t1_data, _ = load_nifti(t1_path)
    t2_data, _ = load_nifti(t2_path)
    pve_data, _ = load_nifti(pve_path)
    
    if t1_data is not None:
        # Axial, coronal, sagittal views
        views = [
            (t1_data, 2, "Axial"),
            (t1_data, 1, "Coronal"),
            (t1_data, 0, "Sagittal"),
        ]
        
        for col, (data, axis, view_name) in enumerate(views):
            ax = axes[0, col]
            s = mid_slice(data, axis=axis)
            s = np.rot90(s)
            ax.imshow(s, cmap='gray', aspect='auto')
            ax.set_title(f"T1w - {view_name}", fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # Tissue segmentation
        if pve_data is not None:
            ax = axes[0, 3]
            s = mid_slice(pve_data, axis=2)
            s = np.rot90(s)
            seg_cmap = LinearSegmentedColormap.from_list('seg', 
                ['black', 'blue', 'green', 'red'], N=4)
            ax.imshow(s, cmap=seg_cmap, aspect='auto', vmin=0, vmax=3)
            ax.set_title("Tissue Segmentation\nCSF/GM/WM", fontsize=11, fontweight='bold')
            ax.axis('off')
    
    if t2_data is not None:
        views = [
            (t2_data, 2, "Axial"),
            (t2_data, 1, "Coronal"),
            (t2_data, 0, "Sagittal"),
        ]
        
        for col, (data, axis, view_name) in enumerate(views):
            ax = axes[1, col]
            s = mid_slice(data, axis=axis)
            s = np.rot90(s)
            ax.imshow(s, cmap='gray', aspect='auto')
            ax.set_title(f"Synthesized T2w - {view_name}", fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # T1w/T2w ratio overlay
        ax = axes[1, 3]
        t2_s = mid_slice(t2_data, axis=2)
        t1_s = mid_slice(t1_data, axis=2)
        if t1_data is not None:
            ratio = np.rot90(t1_s / (t2_s + 1e-8))
            ratio = np.clip(ratio, 0, 5)
            im = ax.imshow(ratio, cmap='hot', aspect='auto')
            ax.set_title("T1w/T2w Ratio\n(Myelin Proxy)", fontsize=11, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"T1 → T2 Conversion Results: {SUBJECT}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, "t1_t2_comparison.png"), dpi=150, bbox_inches='tight')
    print("Saved: t1_t2_comparison.png")
    plt.close()


def create_subject_summary():
    """Create a summary of all subjects processed."""
    subjects = sorted([d for d in os.listdir(INPUT_DIR) 
                       if os.path.isdir(os.path.join(INPUT_DIR, d))])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Create table-like display
    y_pos = 0.95
    ax.text(0.02, y_pos, f"Total Subjects: {len(subjects)}", fontsize=14, fontweight='bold',
           transform=ax.transAxes)
    y_pos -= 0.04
    
    # Column headers
    headers = ["#", "Subject ID", "T1 (RAW)", "T1w_MNI", "T2 (SYNTH)", "Myelin", "Status"]
    x_positions = [0.02, 0.06, 0.45, 0.55, 0.65, 0.75, 0.88]
    for x, header in zip(x_positions, headers):
        ax.text(x, y_pos, header, fontsize=9, fontweight='bold', transform=ax.transAxes,
               bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.2'))
    
    y_pos -= 0.03
    
    for i, sub in enumerate(subjects[:20]):  # Show first 20
        y_pos -= 0.035
        sub_dir = os.path.join(INPUT_DIR, sub)
        
        t1_raw = os.path.exists(os.path.join(sub_dir, "T1w.nii.gz"))
        t1_mni = os.path.exists(os.path.join(sub_dir, "T1w_MNI.nii.gz"))
        t2_syn = os.path.exists(os.path.join(sub_dir, "output_MNI.nii.gz"))
        myelin = os.path.exists(os.path.join(sub_dir, "T1w", sub, "mri", "myelin.nii.gz"))
        
        status = "✓" if all([t1_raw, t1_mni, t2_syn, myelin]) else "Partial"
        
        ax.text(x_positions[0], y_pos, str(i+1), fontsize=8, transform=ax.transAxes)
        ax.text(x_positions[1], y_pos, sub[:40], fontsize=7, transform=ax.transAxes)
        ax.text(x_positions[2], y_pos, "✓" if t1_raw else "✗", fontsize=10,
               color='green' if t1_raw else 'red', transform=ax.transAxes)
        ax.text(x_positions[3], y_pos, "✓" if t1_mni else "✗", fontsize=10,
               color='green' if t1_mni else 'red', transform=ax.transAxes)
        ax.text(x_positions[4], y_pos, "✓" if t2_syn else "✗", fontsize=10,
               color='green' if t2_syn else 'red', transform=ax.transAxes)
        ax.text(x_positions[5], y_pos, "✓" if myelin else "✗", fontsize=10,
               color='green' if myelin else 'red', transform=ax.transAxes)
        ax.text(x_positions[6], y_pos, status, fontsize=9, 
               color='green' if status == "✓" else 'orange', transform=ax.transAxes,
               fontweight='bold')
    
    if len(subjects) > 20:
        ax.text(0.02, y_pos - 0.04, f"... and {len(subjects) - 20} more subjects", 
               fontsize=10, transform=ax.transAxes, style='italic')
    
    ax.set_title("GAN-MAT Processing Summary", fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(FIGS_DIR, "subject_summary.png"), dpi=150, bbox_inches='tight')
    print("Saved: subject_summary.png")
    plt.close()


if __name__ == "__main__":
    print("Generating figures for GAN-MAT documentation...")
    create_pipeline_diagram()
    create_overview_figure()
    create_t1_t2_comparison()
    create_myelin_histogram()
    create_subject_summary()
    print(f"\nAll figures saved to: {FIGS_DIR}")
