import os
import glob
import argparse

from PIL import Image
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('-input_videos_directory', type=str)
parser.add_argument('-output_frames_directory', type=str)
args = parser.parse_args()

#### Defining functionality ####
def get_video_name(path):
    return path.split('/')[-1].split('.')[0]

#### Locating videos in the provided directory ####
video_paths = glob.glob(f'{args.input_videos_directory}/video_*')

#### Creating output directory ####
os.makedirs(args.output_frames_directory, exist_ok=True)

#### Transforming and saving videos ####
for pth in video_paths:
    vid_name = get_video_name(pth)
    os.makedirs(f'{args.output_frames_directory}/{vid_name}', exist_ok=True)

    vid = cv2.VideoCapture(pth)

    frame_count = 0

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
    
        # Save each frame as an image
        frame_filename = os.path.join(f'{args.output_frames_directory}/{vid_name}/frame_{frame_count+1}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    vid.release()