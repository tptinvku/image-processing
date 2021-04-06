import sys
import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

class Task:
    def __init__(self, argv, ui):
        self.imageName = argv[0] if len(argv) > 0 else 'input.jpg'
        self.ui = ui
        self.src = cv.imread(cv.samples.findFile(self.imageName), cv.IMREAD_COLOR)
        self.ddepth = -1

        if self.src is None:
            print('Error opening image!')
            print('Usage: filter2D.py [image_name -- default lena.jpg] \n')
            return -1

    def event_catch(self):
        self.ui.btn_linear_filter.clicked.connect(
            lambda: self.linear_filter()
        )
        self.ui.btn_gaussian_blur.clicked.connect(
            lambda: self.gaussian_blur()
        )
        self.ui.btn_canny_edge.clicked.connect(
            lambda: self.canny_edge_detection()
        )
        self.ui.btn_paint.clicked.connect(
            lambda: self.dodgeV2()
        )

    def linear_filter(self):
        ind = 4
        kernel_size = 3 + 5 * (ind % 5)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel /= (kernel_size * kernel_size)

        dst = cv.filter2D(self.src, self.ddepth, kernel)
        cv.imwrite("linear_filter.jpg", dst)
        self.ui.img_output.setPixmap(QtGui.QPixmap("linear_filter.jpg"))
        self.ui.img_output.setScaledContents(True)
        self.ui.img_output.setWordWrap(True)

    def gaussian_blur(self):
        blur = cv.GaussianBlur(self.src, (5, 5), 0)
        cv.imwrite("gaussian_blur.jpg", blur)
        self.ui.img_output.setPixmap(QtGui.QPixmap("gaussian_blur.jpg"))
        self.ui.img_output.setScaledContents(True)
        self.ui.img_output.setWordWrap(True)

    def canny_edge_detection(self):
        edges = cv.Canny(self.src, 100, 200)
        cv.imwrite("canny_edge.jpg", edges)
        self.ui.img_output.setPixmap(QtGui.QPixmap("canny_edge.jpg"))
        self.ui.img_output.setScaledContents(True)
        self.ui.img_output.setWordWrap(True)

    def dodgeV2(self):
        img_gray = cv.imread("input.jpg", cv.COLOR_BGRA2GRAY)
        img_invert = cv.bitwise_not(img_gray)
        img_smoothing = cv.GaussianBlur(img_invert, (5, 5), 0)
        picture = cv.divide(img_gray, 255 - img_smoothing, scale=256)
        cv.imwrite("picture.jpg", picture)
        self.ui.img_output.setPixmap(QtGui.QPixmap("picture.jpg"))
        self.ui.img_output.setScaledContents(True)
        self.ui.img_output.setWordWrap(True)
