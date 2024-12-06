import os
import cv2
import numpy as np
import argparse
import glob
import trimesh
import matplotlib.pyplot as plt
import math
import os
import warnings
import mmcv
batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)

def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """
    pose_kpt_color={
        0: dict(name='wrist', id=0, color=[255, 255, 255], type='', swap=''),
        1: dict(name='thumb1', id=1, color=[255, 128, 0], type='', swap=''),
        2: dict(name='thumb2', id=2, color=[255, 128, 0], type='', swap=''),
        3: dict(name='thumb3', id=3, color=[255, 128, 0], type='', swap=''),
        4: dict(name='thumb4', id=4, color=[255, 128, 0], type='', swap=''),
        5: dict(name='forefinger1', id=5, color=[255, 153, 255], type='', swap=''),
        6: dict(name='forefinger2', id=6, color=[255, 153, 255], type='', swap=''),
        7: dict(name='forefinger3', id=7, color=[255, 153, 255], type='', swap=''),
        8: dict(name='forefinger4', id=8, color=[255, 153, 255], type='', swap=''),
        9: dict(name='middle_finger1', id=9, color=[102, 178, 255], type='', swap=''),
        10: dict(name='middle_finger2', id=10, color=[102, 178, 255], type='', swap=''),
        11: dict(name='middle_finger3', id=11, color=[102, 178, 255], type='', swap=''),
        12: dict(name='middle_finger4', id=12, color=[102, 178, 255], type='', swap=''),
        13: dict(name='ring_finger1', id=13, color=[255, 51, 51], type='', swap=''),
        14: dict(name='ring_finger2', id=14, color=[255, 51, 51], type='', swap=''),
        15: dict(name='ring_finger3', id=15, color=[255, 51, 51], type='', swap=''),
        16: dict(name='ring_finger4', id=16, color=[255, 51, 51], type='', swap=''),
        17: dict(name='pinky_finger1', id=17, color=[0, 255, 0], type='', swap=''),
        18: dict(name='pinky_finger2', id=18, color=[0, 255, 0], type='', swap=''),
        19: dict(name='pinky_finger3', id=19, color=[0, 255, 0], type='', swap=''),
        20: dict(name='pinky_finger4', id=20, color=[0, 255, 0], type='', swap='')
    }
    pose_link_color={
        0: dict(link=('wrist', 'thumb1'), id=0, color=[255, 128, 0]),
        1: dict(link=('thumb1', 'thumb2'), id=1, color=[255, 128, 0]),
        2: dict(link=('thumb2', 'thumb3'), id=2, color=[255, 128, 0]),
        3: dict(link=('thumb3', 'thumb4'), id=3, color=[255, 128, 0]),
        4: dict(link=('wrist', 'forefinger1'), id=4, color=[255, 153, 255]),
        5: dict(link=('forefinger1', 'forefinger2'), id=5, color=[255, 153, 255]),
        6: dict(link=('forefinger2', 'forefinger3'), id=6, color=[255, 153, 255]),
        7: dict(link=('forefinger3', 'forefinger4'), id=7, color=[255, 153, 255]),
        8: dict(link=('wrist', 'middle_finger1'), id=8, color=[102, 178, 255]),
        9: dict( link=('middle_finger1', 'middle_finger2'), id=9, color=[102, 178, 255]),
        10: dict( link=('middle_finger2', 'middle_finger3'), id=10, color=[102, 178, 255]),
        11: dict( link=('middle_finger3', 'middle_finger4'), id=11, color=[102, 178, 255]),
        12: dict(link=('wrist', 'ring_finger1'), id=12, color=[255, 51, 51]),
        13: dict( link=('ring_finger1', 'ring_finger2'), id=13, color=[255, 51, 51]),
        14: dict( link=('ring_finger2', 'ring_finger3'), id=14, color=[255, 51, 51]),
        15: dict( link=('ring_finger3', 'ring_finger4'), id=15, color=[255, 51, 51]),
        16: dict(link=('wrist', 'pinky_finger1'), id=16, color=[0, 255, 0]),
        17: dict( link=('pinky_finger1', 'pinky_finger2'), id=17, color=[0, 255, 0]),
        18: dict( link=('pinky_finger2', 'pinky_finger3'), id=18, color=[0, 255, 0]),
        19: dict( link=('pinky_finger3', 'pinky_finger4'), id=19, color=[0, 255, 0])
    }
    joint_weights=[1.] * 21
    sigmas=[
        0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035, 0.018,
        0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019, 0.022,
        0.031
    ]


    img = mmcv.imread(img)
    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)
            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                if kpt_score > kpt_score_thr:
                    color = tuple(pose_kpt_color[kid]["color"])
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                   radius, color, -1)
                        transparency = max(0, min(1, kpt_score))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                                   color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)
            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                        and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                        and pos2[1] > 0 and pos2[1] < img_h
                        and kpts[sk[0], 2] > kpt_score_thr
                        and kpts[sk[1], 2] > kpt_score_thr):
                    color = tuple(int(c) for c in pose_link_color[sk_id])
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        X = (pos1[0], pos2[0])
                        Y = (pos1[1], pos2[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                        angle = math.degrees(
                            math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = 2
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)),
                            (int(length / 2), int(stickwidth)), int(angle), 0,
                            360, 1)
                        cv2.fillConvexPoly(img_copy, polygon, color)
                        transparency = max(
                            0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img


def render_hand(input_path, output_path, size):
    sklt_img = np.zeros((size, size, 3), dtype=np.float64)
    keypoints = np.load(input_path)
    sklt_img = imshow_keypoints(sklt_img, keypoints)

    cv2.imwrite(output_path, sklt_img)
    print(f"Saved: {output_path}")

def render_from_keypoints(input_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for kpt_path in glob.glob(os.path.join(input_dir, "**/*.npy"), recursive=True):
        skl_path = os.path.join(output_dir, f"skeloton_{os.path.basename(kpt_path).split('_')[1].split('.')[0]}.jpg")
        render_hand(kpt_path, skl_path, size)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pc from mesh")
    parser.add_argument("--input_dir", type=str, required=True, help="keypoints file folder")
    parser.add_argument("--output_dir", type=str, required=True, help="output skeloton folder")
    parser.add_argument("--size", type=int, required=True, help="image size")
    args = parser.parse_args()
    render_from_keypoints(args.input_dir, args.output_dir, args.size)
