import cv2
import numpy as np

from erqa import ERQA


def test_erqa_v1_0():
    metric = ERQA(version='1.0')

    gt_sample = cv2.imread('samples/gt.png')

    espcn_sample = cv2.imread('samples/espcn.png')
    assert np.isclose(metric(espcn_sample, gt_sample), 0.5615558350576811)

    dbvsr_sample = cv2.imread('samples/dbvsr.png')
    assert np.isclose(metric(dbvsr_sample, gt_sample), 0.7430773590252197)


def test_erqa_v1_1():
    metric = ERQA(version='1.1')

    gt_sample = cv2.imread('samples/gt.png')

    espcn_sample = cv2.imread('samples/espcn.png')
    assert np.isclose(metric(espcn_sample, gt_sample), 0.5796256145601124)

    dbvsr_sample = cv2.imread('samples/dbvsr.png')
    assert np.isclose(metric(dbvsr_sample, gt_sample), 0.7801260659992584)

def test_erqa_vis():
    metric = ERQA(version='1.1')

    gt_sample = cv2.imread('samples/gt.png')

    espcn_sample = cv2.imread('samples/espcn.png')
    espcn_vis = cv2.imread('samples/espcn_vis.png')
    assert np.array_equal((255 * metric(espcn_sample, gt_sample, return_vis=True)[1]).astype(np.uint8), espcn_vis)

    dbvsr_sample = cv2.imread('samples/dbvsr.png')
    dbvsr_vis = cv2.imread('samples/dbvsr_vis.png')
    assert np.array_equal((255 * metric(dbvsr_sample, gt_sample, return_vis=True)[1]).astype(np.uint8), dbvsr_vis)
