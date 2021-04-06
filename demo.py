from __future__ import print_function

import argparse

import cv2 as cv
import numpy as np

morph_size = 0
max_operator = 4
max_elem = 3
max_kernel_size = 21
max_lowThreshold = 100
ratio = 3
kernel_size = 3
title_trackbar_linear_filter = 'Linear Filter'
title_trackbar_gaussian_filter = 'Gaussian Filter'
title_trackbar_canny_edges = 'Canny Edges'
title_trackbar_painter = 'Painter'
title_window = 'Image Processing'


class Demo:
    def __init__(self, src):
        self.src = src
        self.output = None

    def linear_filter(self, val):
        ddepth = -1
        ind = val
        kernel_size = 3 + 2 * (ind % 5)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= (kernel_size * kernel_size)

        self.output = cv.filter2D(self.src, ddepth, kernel)
        cv.imshow(title_window, self.output)

    def gaussian_blur(self, val):
        self.output = cv.GaussianBlur(self.src, (5, 5), val)
        cv.imshow(title_window, self.output)

    def canny_edge_detection(self, val):
        low_threshold = val
        src_gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)
        img_blur = cv.blur(src_gray, (3, 3))
        detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
        mask = detected_edges != 0
        self.output = self.src * (mask[:, :, None].astype(self.src.dtype))
        cv.imshow(title_window, self.output)

    def dodgeV2(self, val):
        img_gray = cv.cvtColor(self.src, cv.COLOR_BGRA2GRAY)
        img_invert = cv.bitwise_not(img_gray)
        img_smoothing = cv.GaussianBlur(img_invert, (5, 5), val)
        self.output = cv.divide(img_gray, 255 - img_smoothing, scale=256)
        cv.imshow(title_window, self.output)


parser = argparse.ArgumentParser(description='Code for More Morphology Transformations tutorial.')
parser.add_argument('--input', help='Path to input image.', default='sg.jpg')
args = parser.parse_args()
src = cv.imread(args.input)
d = Demo(src)
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
cv.namedWindow(title_window)
cv.createTrackbar(title_trackbar_linear_filter, title_window, 0, max_operator, d.linear_filter)
cv.createTrackbar(title_trackbar_gaussian_filter, title_window, 0, max_elem, d.gaussian_blur)
cv.createTrackbar(title_trackbar_canny_edges, title_window, 0, max_lowThreshold, d.canny_edge_detection)
cv.createTrackbar(title_trackbar_painter, title_window, 0, max_lowThreshold, d.dodgeV2)

src =d.linear_filter(0)
d.gaussian_blur(0)
d.canny_edge_detection(0)
d.dodgeV2(0)
cv.waitKey(0)
