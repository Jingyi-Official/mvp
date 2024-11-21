import cv2
import mediapipe as mp
import numpy as np
import os

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse

def generate_hand_mask(image_path, output_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read: {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    results = hands.process(rgb_image)


    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            height, width, _ = image.shape
            points = np.array([[int(lm.x * width), int(lm.y * height)] for lm in hand_landmarks.landmark], np.int32)


            cv2.fillPoly(mask, [points], 255) 


    cv2.imwrite(output_path, mask)
    print(f"Save to: {output_path}")

    hands.close()


def generate_masks_for_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for image_file in os.listdir(input_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, f"mask_{image_file}")
            generate_hand_mask(input_path, output_path)

# input_directory = "/mnt/ssd/jingyi/Projects/hamer/data/jingyi/images"  
# output_directory = "/mnt/ssd/jingyi/Projects/hamer/data/jingyi/masks"  
# generate_masks_for_directory(input_directory, output_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrack mask")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    generate_masks_for_directory(args.input_dir, args.output_dir)
