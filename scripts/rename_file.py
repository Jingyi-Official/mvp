import argparse
import glob
import trimesh
import os
import matplotlib.pyplot as plt
def rename(dir_path):
    image_path = 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pc from mesh")
    parser.add_argument("--input_dir", type=str, required=True, help="input folder")
    args = parser.parse_args()
    rename(args.input_dir)
