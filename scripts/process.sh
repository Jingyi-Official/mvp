#!/bin/bash
# bash scripts/process-sequence.sh /mnt/ssd/jingyi/Projects/hamer/data/jingyi/jingyi.mp4
# Check if path is specified as an argument
if [ -z "$1" ]; then
  echo "Please specify the path to the folder containing the data."
  exit 1
fi

# Check if the path exists
if [ ! -d "$1" ]; then
  echo "The specified path does not exist or is not a directory."
  exit 1
fi

path=$(readlink -f "$1")
parent_dir=$(dirname "$path")
project_dir=$(dirname "$parent_dir")
echo $path # /mnt/ssd/jingyi/Projects/hamer/data/jingyi
echo $parent_dir # /mnt/ssd/jingyi/Projects/hamer/data
echo $project_dir # /mnt/ssd/jingyi/Projects/hamer

# Run crop
if [ ! -f "$path" ]; then
  echo "Running crop.py in $path"
  python3 $project_dir/scripts/crop.py --video_path $path/video.mp4 --output_folder $path/video_crop.mp4
fi

# Run image extraction
if [ ! -f "$path" ]; then
  echo "Running extract_img.py in $path"
  python3 $project_dir/scripts/extract_img.py --video_path $path/video_crop.mp4 --output_folder $path/images
fi

# Run mask extraction
if [ ! -f "$path" ]; then
  echo "Running extract_mask.py in $path"
  python3 $project_dir/scripts/extract_img.py --video_path $path/images --output_folder $path/masks
fi

# Run hamer
if [ ! -f "$path" ]; then
  echo "Running hamer in $path"
  python3 $project_dir/demo.py --img_folder $path --out_folder $path --batch_size=48 --side_view --save_mesh --full_frame
fi

# if [ ! -d "$path/masks" ]; then
#   echo "Running mask in $path"
#   python scripts/custom/run-sam.py --data_dir $path
#   # python scripts/custom/run-rvm.py --data_dir $path
#   python scripts/custom/extract-largest-connected-components.py --data_dir $path
# fi

# if [ ! -f "$path/poses.npz" ]; then
#   python scripts/custom/run-romp.py --data_dir $path
# fi

# if [ ! -f "$path/poses_optimized.npz" ]; then
#   echo "Refining SMPL..."
#   python scripts/custom/refine-smpl.py --data_dir $path --gender $2 # --silhouette
# fi

# if [ ! -f "$path/output.mp4" ]; then
#   python scripts/visualize-SMPL.py --path $path --gender $2 --pose $path/poses_optimized.npz --headless --fps 1
# fi