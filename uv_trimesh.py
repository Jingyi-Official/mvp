import argparse

import trimesh
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-input_mesh_path', type=str)
parser.add_argument('-output_uv_path', type=str)
args = parser.parse_args()

#### Loading mesh ####
mesh = trimesh.load(args.input_mesh_path)

#### Extracting and setting uv ####
uv = mesh.vertices[:, :2]

#### Saving figure ####
fig, ax = plt.subplots()
ax.scatter(uv[:, 0], uv[:, 1], s=2)
ax.axis('off')
fig.set_size_inches(256 / 100, 256 / 100)
fig.savefig(args.output_uv_path, format='jpg', pad_inches=0, dpi=100)