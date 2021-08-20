from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np

from .erqa import ERQA


def parse_args():
    parser = ArgumentParser(
        description='Edge Restoration Quality Assessment (ERQA)')

    parser.add_argument('image1', type=Path, help='Path to first image')
    parser.add_argument('image2', type=Path, help='Path to second image')
    parser.add_argument('--vis', type=Path, help='Path to metric visualization')

    parser.add_argument(
        '--version', choices=['1.0', '1.1'], default='1.1', help='Which version of the metric to use')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.vis is not None:
        save_vis = True
    else:
        save_vis = False

    metric = ERQA(version=args.version)

    image1 = cv2.imread(str(args.image1))
    image2 = cv2.imread(str(args.image2))

    score = metric(image1, image2, return_vis=save_vis)
    if save_vis:
        score, vis = score
        vis = (255 * vis).astype(np.uint8)
        cv2.imwrite(str(args.vis), vis)

    print(score)


if __name__ == '__main__':
    main()
