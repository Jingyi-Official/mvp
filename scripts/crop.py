import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

def create_hand_centered_video(input_video, output_video, output_size=(1024, 1024)):
 
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Cannot open: {input_video}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, output_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x * width for lm in hand_landmarks.landmark]
                y_coords = [lm.y * height for lm in hand_landmarks.landmark]

                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2

                crop_x1 = max(center_x - output_size[0] // 2, 0)
                crop_y1 = max(center_y - output_size[1] // 2, 0)
                crop_x2 = min(crop_x1 + output_size[0], width)
                crop_y2 = min(crop_y1 + output_size[1], height)


                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                padded_frame = cv2.copyMakeBorder(
                    cropped_frame,
                    top=(output_size[1] - cropped_frame.shape[0]) // 2,
                    bottom=(output_size[1] - cropped_frame.shape[0]) // 2,
                    left=(output_size[0] - cropped_frame.shape[1]) // 2,
                    right=(output_size[0] - cropped_frame.shape[1]) // 2,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
                )

                padded_frame = cv2.resize(padded_frame, output_size)
                out.write(padded_frame)
                break 
        else:

            resized_frame = cv2.resize(frame, output_size)
            out.write(resized_frame)


    cap.release()
    out.release()
    hands.close()
    print(f"Done: {output_video}")

# input_video = "/mnt/ssd/jingyi/Projects/hamer/data/test/video.mp4"  
# output_video = "/mnt/ssd/jingyi/Projects/hamer/data/test/video_crop.mp4"  
# create_hand_centered_video(input_video, output_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop video")
    parser.add_argument("--videoinput_video_path", type=str, required=True, help="MP4 path")
    parser.add_argument("--output_video", type=str, required=True, help="output path")
    args = parser.parse_args()
    create_hand_centered_video(args.input_video, args.output_video)