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
plt.savefig(args.output_uv_path, format='jpg', bbox_inches='tight', pad_inches=0)