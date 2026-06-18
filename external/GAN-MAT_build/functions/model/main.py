import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True' 

from t2 import test

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_dir", type=str, dest="pipeline_dir")
parser.add_argument("--input_dir", type=str, dest="input_dir")
parser.add_argument("--output_dir", type=str, dest="output_dir")
parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
args = parser.parse_args()

test(args)
