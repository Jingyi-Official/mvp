import cv2
import os
import argparse

def video_to_images(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        output_path = os.path.join(output_folder, f"{frame_count+1:06d}.png")
        cv2.imwrite(output_path, frame)
        print(f"Save to: {output_path}")

        frame_count += 1

    cap.release()
    print("Done")


# video_path = "/mnt/ssd/jingyi/Projects/hamer/data/jingyi/jingyi.mp4"  
# output_folder = "/mnt/ssd/jingyi/Projects/hamer/data/jingyi/images"  
# video_to_images(video_path, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from video")
    parser.add_argument("--video_path", type=str, required=True, help="MP4 path")
    parser.add_argument("--output_folder", type=str, required=True, help="output folder")
    args = parser.parse_args()
    video_to_images(args.video_path, args.output_folder)