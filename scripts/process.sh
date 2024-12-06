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


# # # # Run crop
# if [ ! -f "$path" ]; then
#   echo "Running crop.py in $path"
#   python3 $project_dir/scripts/crop.py --input_video $path/video.mp4 --output_video $path/video_crop.mp4
# fi

# # # # # Run image extraction
# if [ ! -f "$path" ]; then
#   echo "Running extract_img.py in $path"
#   python3 $project_dir/scripts/extract_img.py --video_path $path/video_crop.mp4 --output_folder $path/image
# fi

# # # # # Run hamer
# if [ ! -f "$path" ]; then
#   echo "Running hamer in $path"
#   python3 $project_dir/demo.py --img_folder $path/image --out_folder $path --batch_size=48 --side_view --save_mesh --full_frame
# fi

# # Run mask extraction
if [ ! -f "$path" ]; then
  echo "Running extract_mask.py in $path"
  python3 $project_dir/scripts/extract_mask.py --input_dir $path/image --output_dir $path/back --n_hands 1
fi

# # # run point cloud
if [ ! -f "$path" ]; then
  echo "Running extract_pc.py in $path"
  python3 $project_dir/scripts/extract_pc.py --input_dir $path/mesh --output_dir $path/uv 
fi

# # run normal
# # if [ ! -f "$path" ]; then
# #   echo "Running extract_normal.py in $path"
# #   python3 $project_dir/scripts/extract_normal.py --input_dir $path/mesh --output_dir $path/normal 
# # fi

# # run overlapping
if [ ! -f "$path" ]; then
  echo "Running hamer in $path"
  python3 $project_dir/demo_blur.py --img_folder $path/image --out_folder $path --batch_size=48 --side_view --save_mesh --full_frame
fi

# run skeloton ===========================
# if [ ! -f "$path" ]; then
#   echo "Running extract_skeloton.py in $path"
#   python3 $project_dir/scripts/extract_skeloton.py --input_dir $path/keypoints --output_dir $path/skeloton --size 1536
# fi
