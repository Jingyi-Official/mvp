import mediapipe as mp
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import os
import argparse

#SAM model
#you need to download this first!!
#wget -q \
#'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)

#mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def extract_hand_points(image_path):
    """
   hand landmarks from the image using mediapipe

    """
    keypoint_path = f"{os.path.basename(image_path).split('.')[0]}.npy"
    keypoint_path = os.path.dirname(os.path.dirname(image_path)) + "/keypoints/" + keypoint_path

    hand_points = np.load(keypoint_path)[0][:, 0:2]

    return hand_points

def extract_bboex(image_path):
    """
   hand landmarks from the image using mediapipe

    """
    bboxes_path = f"{os.path.basename(image_path).split('.')[0]}.npy"
    bboxes_path = os.path.dirname(os.path.dirname(image_path)) + "/bboxes/" + bboxes_path

    bboxes = np.load(bboxes_path)[0]

    return bboxes


def segment_hands_with_sam(image_path, output_path, n_hands):
    """
    Segment hands in an image using SAM and by hand landmarks from mediapipe.
    """

    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read: {image_path}")
        return

    # hand points using vit
    hand_points = extract_hand_points(image_path)
    input_box = extract_bboex(image_path)

    if not hand_points.all():
        print(f"no hands detected in {image_path}. skipping...")
        return

    #SAM's input image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    #make segmentation masks using SAM
    input_points = np.array(hand_points)
    input_labels = np.ones(len(input_points))
    if n_hands == 1:
        multimask_output = False
    else:
        multimask_output = True
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_box,
        multimask_output=multimask_output
    )
    if n_hands ==1:
        mask = masks[2]
        mask = (mask * 255).astype(np.uint8)

    else:
        mask = np.zeros_like(masks[0], dtype=np.uint8)
        for i, m in enumerate(masks):
            mask[m > 0] = 255
        mask = mask.astype(np.uint8)

    # get largest components
    # Saveing
    cv2.imwrite(output_path, mask)
    print(f"mask saved to: {output_path}")

def process_directory_with_sam(input_dir, output_dir, n_hands):
    """
    Process all images in a directory
    """
    os.makedirs(output_dir, exist_ok=True)

    for image_file in os.listdir(input_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, image_file)
            segment_hands_with_sam(input_path, output_path,n_hands)



# input_images = r"C:\Users\motzk\Documents\Master\VIS\hands\video_crop_folder"
# output_folder = r"C:\Users\motzk\Documents\Master\VIS\hands\sams"
# process_directory_with_sam(input_images, output_folder,2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrack mask")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_hands", type=str, required=True)
    args = parser.parse_args()

    process_directory_with_sam(args.input_dir, args.output_dir ,args.n_hands)
