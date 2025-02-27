
import os

import cv2
import numpy as np
import argparse

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Trimap')
    parser.add_argument('--mskDir', type=str, required=True, help="masks directory")
    parser.add_argument('--saveDir', type=str, required=True, help="where trimap result save to")
    # parser.add_argument('--list', type=str, required=True, help="list of images id")
    parser.add_argument('--size', type=int, required=False, default=10, help="kernel size")
    args = parser.parse_args()
    print(args)
    return args


def erode_dilate(msk, struc="RECT", size=(10, 10)):
    Y, X = msk.shape[:2]
    SZ = (X+Y)//70
    size = (SZ, SZ)
    if struc == "RECT":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif struc == "CORSS":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    # msk = msk.astype(np.float32)
    msk = msk / 255
    # msk = msk.astype(np.uint8)

    # val in 0 or 255

    dilated = cv2.dilate(msk, kernel, iterations=1) * 255
    eroded = cv2.erode(msk, kernel, iterations=1) * 255

    cnt1 = len(np.where(msk >= 0)[0])
    cnt2 = len(np.where(msk == 0)[0])
    cnt3 = len(np.where(msk == 1)[0])
    # print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
    assert (cnt1 == cnt2 + cnt3)

    cnt1 = len(np.where(dilated >= 0)[0])
    cnt2 = len(np.where(dilated == 0)[0])
    cnt3 = len(np.where(dilated == 255)[0])
    # print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
    assert (cnt1 == cnt2 + cnt3)

    cnt1 = len(np.where(eroded >= 0)[0])
    cnt2 = len(np.where(eroded == 0)[0])
    cnt3 = len(np.where(eroded == 255)[0])
    # print("all:{} bg:{} fg:{}".format(cnt1, cnt2, cnt3))
    assert (cnt1 == cnt2 + cnt3)

    res = dilated.copy()
    # res[((dilated == 255) & (msk == 0))] = 128
    res[((dilated == 255) & (eroded == 0))] = 128

    return res


def main():
    args = get_args()
    masks = os.listdir(args.mskDir)
    print("Images Count: {}".format(len(masks)))
    if not os.path.exists(args.saveDir):
        os.makedirs(args.saveDir, exist_ok=True)
    for name in tqdm(masks):
        msk = cv2.imread(os.path.join(args.mskDir, name), 0)
        assert msk is not None, os.path.join(args.mskDir, name)
        trimap = erode_dilate(msk, size=(args.size, args.size))
        cv2.imwrite(os.path.join(args.saveDir, name), trimap)


if __name__ == "__main__":
    main()
