# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(618, 555)
        Form.setMaximumSize(QtCore.QSize(618, 700))
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.widget_2 = QtWidgets.QWidget(Form)
        self.widget_2.setObjectName("widget_2")
        self.formLayout = QtWidgets.QFormLayout(self.widget_2)
        self.formLayout.setObjectName("formLayout")
        self.btn_linear_filter = QtWidgets.QPushButton(self.widget_2)
        self.btn_linear_filter.setObjectName("btn_linear_filter")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.btn_linear_filter)
        self.btn_gaussian_blur = QtWidgets.QPushButton(self.widget_2)
        self.btn_gaussian_blur.setObjectName("btn_gaussian_blur")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.btn_gaussian_blur)
        self.btn_canny_edge = QtWidgets.QPushButton(self.widget_2)
        self.btn_canny_edge.setObjectName("btn_linear_filter")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.btn_canny_edge)
        self.btn_paint = QtWidgets.QPushButton(self.widget_2)
        self.btn_paint.setObjectName("btn_linear_filter")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.btn_paint)
        self.horizontalLayout.addWidget(self.widget_2)
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setMaximumSize(QtCore.QSize(600, 700))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setObjectName("gridLayout")
        self.img_output = QtWidgets.QLabel(self.widget)
        self.img_output.setMaximumSize(QtCore.QSize(300, 500))
        self.img_output.setSizeIncrement(QtCore.QSize(0, 0))
        self.img_output.setFrameShape(QtWidgets.QFrame.Box)
        self.img_output.setText("")
        self.img_output.setPixmap(QtGui.QPixmap("input.jpg"))
        self.img_output.setScaledContents(True)
        self.img_output.setWordWrap(True)
        self.img_output.setObjectName("img_output")
        self.gridLayout.addWidget(self.img_output, 0, 1, 1, 1)
        self.img_origin = QtWidgets.QLabel(self.widget)
        self.img_origin.setEnabled(True)
        self.img_origin.setMaximumSize(QtCore.QSize(300, 500))
        self.img_origin.setFrameShape(QtWidgets.QFrame.Box)
        self.img_origin.setText("")
        self.img_origin.setPixmap(QtGui.QPixmap("input.jpg"))
        self.img_origin.setScaledContents(True)
        self.img_origin.setWordWrap(True)
        self.img_origin.setObjectName("img_origin")
        self.gridLayout.addWidget(self.img_origin, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 1, 1, 1)
        self.horizontalLayout.addWidget(self.widget)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Image Processing"))
        self.btn_linear_filter.setText(_translate("Form", "Linear Filter"))
        self.btn_gaussian_blur.setText(_translate("Form", "Gaussian Blur"))
        self.btn_canny_edge.setText(_translate("Form", "Canny Edge"))
        self.btn_paint.setText(_translate("Form", "Paint"))
        self.label_3.setText(_translate("Form", "Origin"))
        self.label_4.setText(_translate("Form", "Ouput"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    from task import Task
    t = Task(sys.argv[1:], ui)
    controller = t.event_catch()
    sys.exit(app.exec_())
