import numpy as np
import trimesh
from PIL import Image
import argparse
import glob
import os


def generate_normal_map(obj_file, output_image="normal_map.png", resolution=(1024, 1024)):
    mesh = trimesh.load(obj_file)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Provided OBJ file is not a valid trimesh object.")

    width, height = resolution
    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    vertices = mesh.vertices
    faces = mesh.faces
    vertex_normals = mesh.vertex_normals

    min_bounds = np.min(vertices, axis=0)
    max_bounds = np.max(vertices, axis=0)
    normalized_vertices = (vertices - min_bounds) / (max_bounds - min_bounds)

    # 遍历每个三角形并烘焙法线到 2D 网格
    for face in faces:
        # 获取三角形顶点和对应法线
        v0, v1, v2 = normalized_vertices[face]
        n0, n1, n2 = vertex_normals[face]

        # 绘制三角形并插值法线（略复杂，这里可以使用烘焙工具优化）

    # 转换法线到 [0, 255]
    normal_map = (normal_map + 1) / 2.0  # 归一化到 [0, 1]
    normal_map = (normal_map * 255).astype(np.uint8)

    # 保存为图像
    normal_map_image = Image.fromarray(normal_map, mode="RGB")
    normal_map_image.save(output_image)
    print(f"Normal map saved to {output_image}")

def extract_normal(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for each in glob.glob(os.path.join(input_dir, "**/*.obj"), recursive=True):
        out_path = os.path.join(output_dir, f"normal_{os.path.basename(each).split('_')[1]}.jpg")
        generate_normal_map(each, out_path, resolution=(256, 256))



# input_mesh_path = "/mnt/ssd/jingyi/Projects/hamer/data/hand1/mesh"  
# output_uv_path = "/mnt/ssd/jingyi/Projects/hamer/data/jingyi/images"  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pc from mesh")
    parser.add_argument("--input_dir", type=str, required=True, help="obj file folder")
    parser.add_argument("--output_dir", type=str, required=True, help="output pc projection folder")
    args = parser.parse_args()
    extract_normal(args.input_dir, args.output_dir)
