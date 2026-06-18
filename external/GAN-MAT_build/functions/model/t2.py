import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Pix2Pix_3D
from dataset import Dataset


def test(args):
    pipeline_dir = args.pipeline_dir
    input_dir = args.input_dir
    output_dir = args.output_dir
    batch_size = args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_test = Dataset(input_dir=input_dir, output_dir=output_dir)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    netG = Pix2Pix_3D(in_channels=3, out_channels=1).to(device)
    model_path = "{}/functions/model/model.pth".format(pipeline_dir)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Please download model.pth from: "
            "https://www.dropbox.com/sh/nnzayieuizd012y/AACLSwUY9BBTCdf66_nWqK02a?dl=0"
        )
    
    dict_model = torch.load(model_path, map_location=device)
    netG.load_state_dict(dict_model['netG'])

    with torch.no_grad():
        netG.eval()

        for batch, data in enumerate(loader_test, 1):
            input = data['input'].to(device)
            output = netG(input)

            for j in range(output.shape[0]):
                idx = batch_size * (batch - 1) + j
                output_ = output[j]
                output_ = output_.cpu().numpy()

                np.save(input_dir + "/{}/output_T2w.npy".format(dataset_test.lst_sub[idx]), output_[0])

                print("synthesizing {} T2-weighted MRI".format(dataset_test.lst_sub[idx]))
