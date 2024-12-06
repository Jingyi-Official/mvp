import argparse
import glob
import trimesh
import os
import matplotlib.pyplot as plt

def mesh_to_pc(input_mesh_path, output_uv_path):
    #### Loading mesh ####
    mesh = trimesh.load(input_mesh_path)

    #### Extracting and setting uv ####
    uv = mesh.vertices[:, :2]

    #### Saving figure ####
    fig, ax = plt.subplots()
    ax.scatter(uv[:, 0], uv[:, 1], s=2)
    ax.axis('off')
    fig.set_size_inches(256 / 100, 256 / 100)
    fig.savefig(output_uv_path, format='png', pad_inches=0, dpi=100)


def extract_pc(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for each in glob.glob(os.path.join(input_dir, "**/*.obj"), recursive=True):
        out_path = os.path.join(output_dir, f"{os.path.basename(each).split('_')[0]}.png")
        mesh_to_pc(each, out_path)



# input_mesh_path = "/mnt/ssd/jingyi/Projects/hamer/data/hand1/mesh"  
# output_uv_path = "/mnt/ssd/jingyi/Projects/hamer/data/jingyi/images"  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pc from mesh")
    parser.add_argument("--input_dir", type=str, required=True, help="obj file folder")
    parser.add_argument("--output_dir", type=str, required=True, help="output pc projection folder")
    args = parser.parse_args()
    extract_pc(args.input_dir, args.output_dir)
