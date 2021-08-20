from functools import partial

import cv2
import numpy as np


def make_slice(img, left, right, axis):
    sl = [slice(None)] * img.ndim
    sl[axis] = slice(left, right)

    return img[tuple(sl)]


def shift1d(img, gt, shift=1, axis=0):
    if shift > 0:
        x1, x2 = shift, img.shape[axis]
        x3, x4 = 0, -shift  # gt
    elif shift == 0:
        x1, x2, x3, x4 = 0, img.shape[axis], 0, img.shape[axis]
    else:
        x1, x2 = 0, shift
        x3, x4 = -shift, img.shape[axis]

    img = make_slice(img, x1, x2, axis=axis)
    gt = make_slice(gt, x3, x4, axis=axis)

    return img, gt


def shift2d(img, gt, a=1, b=1):
    img, gt = shift1d(img, gt, a, axis=0)
    img, gt = shift1d(img, gt, b, axis=1)

    return img, gt


class ERQA:
    def __init__(self, shift_compensation=True, penalize_wider_edges=None, global_compensation=True, version='1.0'):
        """
        shift_compensation - if one-pixel shifts of edges are compensated
        """
        # Set global defaults
        self.global_compensation = global_compensation
        self.shift_compensation = shift_compensation

        # Set version defaults
        if version == '1.0':
            self.penalize_wider_edges = False
        elif version == '1.1':
            self.penalize_wider_edges = True
        else:
            raise ValueError('There is no version {} for ERQA'.format(version))

        # Override version defaults
        if penalize_wider_edges is not None:
            self.penalize_wider_edges = penalize_wider_edges

        # Set detector
        self.edge_detector = partial(cv2.Canny, threshold1=100, threshold2=200)

    def __call__(self, img, gt, return_vis=False):
        assert gt.shape == img.shape
        assert gt.shape[2] == 3, 'Compared images should be in BGR format'

        if self.global_compensation:
            img, gt = self._global_compensation(img, gt)

        edge = self.edge_detector(img) // 255
        gt_edge = self.edge_detector(gt) // 255

        true_positive, false_negative = self.match_edges(edge, gt_edge)

        f1 = self.f1_matches(edge, true_positive, false_negative)

        if return_vis:
            vis = self.generate_visualization(edge, true_positive, false_negative)
            return f1, vis
        else:
            return f1

    def _global_compensation(self, img, gt_img, window_range=3, metric='mse'):
        window = range(-window_range, window_range + 1)

        if metric == 'mse':
            def metric(x, y):
                return np.mean((x.astype(float) - y.astype(float)) ** 2)
        else:
            raise ValueError('Unsupported metric "{}" for global compensation'.format(metric))

        shifts = {}
        for i in window:
            for j in window:
                shifted_img, cropped_gt_img = shift2d(img, gt_img, i, j)

                metric_value = metric(shifted_img, cropped_gt_img)
                shifts[(i, j)] = metric_value

        (i, j), _ = min(shifts.items(), key=lambda x: x[1])

        return shift2d(img, gt_img, i, j)

    def match_edges(self, edge, gt_edge):
        assert gt_edge.shape == edge.shape

        true_positive = np.zeros_like(edge)
        false_negative = gt_edge.copy()

        # Count true positive
        if self.shift_compensation:
            window_range = 1
        else:
            window_range = 0

        window = sorted(range(-window_range, window_range + 1), key=abs)  # Place zero at first place

        for i in window:
            for j in window:
                gt_ = np.roll(false_negative, i, axis=0)
                gt_ = np.roll(gt_, j, axis=1)

                ad = edge * gt_ * np.logical_not(true_positive)

                np.logical_or(true_positive, ad, out=true_positive)
                if self.penalize_wider_edges:
                    # Unmark already used edges
                    ad = np.roll(ad, -j, axis=1)
                    ad = np.roll(ad, -i, axis=0)
                    np.logical_and(false_negative, np.logical_not(ad), out=false_negative)

        if not self.penalize_wider_edges:
            false_negative = gt_edge * np.logical_not(true_positive)

        assert not np.logical_and(true_positive, false_negative).any()

        return true_positive, false_negative

    def f1_matches(self, edge, true_positive, false_negative):
        tp = np.sum(true_positive)
        fp = np.sum(edge) - tp
        fn = np.sum(false_negative)

        if tp + fp == 0 or tp + fn == 0:
            f1 = 0
        else:
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)

            f1 = 2 * prec * recall / (prec + recall)

        return f1

    def generate_visualization(self, edge, true_positive, false_negative):
        """
        Form f1 data into RGB visualization

        - False Negative - blue
        - False Positive - red
        - False Negative - white
        - False Negative - black

        Args:
            edge (np.array): edge mask of image
            true_positive (np.array): true positive mask from `match_edges`
            false_negative (np.array): false negative mask from `match_edges`
        """
        false_positive = edge - true_positive

        # Using BGR as in OpenCV
        blue = np.array([1, 0, 0])
        red = np.array([0, 0, 1])
        white = np.array([1, 1, 1])

        return false_negative[..., None] * blue[None, None] \
               + false_positive[..., None] * red[None, None] \
               + true_positive[..., None] * white[None, None]
