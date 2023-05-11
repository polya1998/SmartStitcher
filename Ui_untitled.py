# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\lybstitch\smart_stitch\3.18\untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1216, 745)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.InputJsonButton = QtWidgets.QPushButton(self.centralwidget)
        self.InputJsonButton.setObjectName("InputJsonButton")
        self.verticalLayout.addWidget(self.InputJsonButton)
        self.CeateJsonButton = QtWidgets.QPushButton(self.centralwidget)
        self.CeateJsonButton.setObjectName("CeateJsonButton")
        self.verticalLayout.addWidget(self.CeateJsonButton)
        self.print_jsonButton = QtWidgets.QPushButton(self.centralwidget)
        self.print_jsonButton.setObjectName("print_jsonButton")
        self.verticalLayout.addWidget(self.print_jsonButton)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.JsonInfoBox = QtWidgets.QGroupBox(self.centralwidget)
        self.JsonInfoBox.setObjectName("JsonInfoBox")
        self.gridLayout = QtWidgets.QGridLayout(self.JsonInfoBox)
        self.gridLayout.setObjectName("gridLayout")
        self.output_folderButton = QtWidgets.QPushButton(self.JsonInfoBox)
        self.output_folderButton.setObjectName("output_folderButton")
        self.gridLayout.addWidget(self.output_folderButton, 1, 0, 1, 1)
        self.input_folderButton = QtWidgets.QPushButton(self.JsonInfoBox)
        self.input_folderButton.setObjectName("input_folderButton")
        self.gridLayout.addWidget(self.input_folderButton, 0, 0, 1, 1)
        self.overlapYEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.overlapYEdit.setObjectName("overlapYEdit")
        self.gridLayout.addWidget(self.overlapYEdit, 6, 2, 1, 3)
        self.label_10 = QtWidgets.QLabel(self.JsonInfoBox)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 10, 5, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.JsonInfoBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 5, 0, 1, 1)
        self.wildnameEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.wildnameEdit.setObjectName("wildnameEdit")
        self.gridLayout.addWidget(self.wildnameEdit, 9, 0, 1, 4)
        self.label_6 = QtWidgets.QLabel(self.JsonInfoBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 8, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.JsonInfoBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 8, 3, 1, 3)
        self.high_res_input_folder_Button = QtWidgets.QPushButton(self.JsonInfoBox)
        self.high_res_input_folder_Button.setObjectName("high_res_input_folder_Button")
        self.gridLayout.addWidget(self.high_res_input_folder_Button, 0, 7, 1, 1)
        self.z_lengthEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.z_lengthEdit.setObjectName("z_lengthEdit")
        self.gridLayout.addWidget(self.z_lengthEdit, 3, 5, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.JsonInfoBox)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 1, 7, 1, 2)
        self.output_path = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.output_path.setObjectName("output_path")
        self.gridLayout.addWidget(self.output_path, 1, 2, 1, 5)
        self.highest_vixel_sizeEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.highest_vixel_sizeEdit.setObjectName("highest_vixel_sizeEdit")
        self.gridLayout.addWidget(self.highest_vixel_sizeEdit, 1, 9, 1, 1)
        self.overlapXEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.overlapXEdit.setMinimumSize(QtCore.QSize(205, 0))
        self.overlapXEdit.setObjectName("overlapXEdit")
        self.gridLayout.addWidget(self.overlapXEdit, 6, 0, 1, 1)
        self.resgroupBox = QtWidgets.QGroupBox(self.JsonInfoBox)
        self.resgroupBox.setObjectName("resgroupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.resgroupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.all_stitch4 = QtWidgets.QCheckBox(self.resgroupBox)
        self.all_stitch4.setObjectName("all_stitch4")
        self.gridLayout_2.addWidget(self.all_stitch4, 3, 1, 1, 1)
        self.res1_label = QtWidgets.QLabel(self.resgroupBox)
        self.res1_label.setObjectName("res1_label")
        self.gridLayout_2.addWidget(self.res1_label, 0, 0, 1, 1)
        self.part_stitch3 = QtWidgets.QCheckBox(self.resgroupBox)
        self.part_stitch3.setObjectName("part_stitch3")
        self.gridLayout_2.addWidget(self.part_stitch3, 2, 2, 1, 1)
        self.all_stitch2 = QtWidgets.QCheckBox(self.resgroupBox)
        self.all_stitch2.setObjectName("all_stitch2")
        self.gridLayout_2.addWidget(self.all_stitch2, 1, 1, 1, 1)
        self.res5_label = QtWidgets.QLabel(self.resgroupBox)
        self.res5_label.setObjectName("res5_label")
        self.gridLayout_2.addWidget(self.res5_label, 4, 0, 1, 1)
        self.all_stitch6 = QtWidgets.QCheckBox(self.resgroupBox)
        self.all_stitch6.setObjectName("all_stitch6")
        self.gridLayout_2.addWidget(self.all_stitch6, 5, 1, 1, 1)
        self.part_stitch4 = QtWidgets.QCheckBox(self.resgroupBox)
        self.part_stitch4.setObjectName("part_stitch4")
        self.gridLayout_2.addWidget(self.part_stitch4, 3, 2, 1, 1)
        self.res6_label = QtWidgets.QLabel(self.resgroupBox)
        self.res6_label.setObjectName("res6_label")
        self.gridLayout_2.addWidget(self.res6_label, 5, 0, 1, 1)
        self.res4_label = QtWidgets.QLabel(self.resgroupBox)
        self.res4_label.setObjectName("res4_label")
        self.gridLayout_2.addWidget(self.res4_label, 3, 0, 1, 1)
        self.res3_label = QtWidgets.QLabel(self.resgroupBox)
        self.res3_label.setObjectName("res3_label")
        self.gridLayout_2.addWidget(self.res3_label, 2, 0, 1, 1)
        self.part_stitch5 = QtWidgets.QCheckBox(self.resgroupBox)
        self.part_stitch5.setObjectName("part_stitch5")
        self.gridLayout_2.addWidget(self.part_stitch5, 4, 2, 1, 1)
        self.all_stitch5 = QtWidgets.QCheckBox(self.resgroupBox)
        self.all_stitch5.setObjectName("all_stitch5")
        self.gridLayout_2.addWidget(self.all_stitch5, 4, 1, 1, 1)
        self.part_stitch2 = QtWidgets.QCheckBox(self.resgroupBox)
        self.part_stitch2.setObjectName("part_stitch2")
        self.gridLayout_2.addWidget(self.part_stitch2, 1, 2, 1, 1)
        self.part_stitch1 = QtWidgets.QCheckBox(self.resgroupBox)
        self.part_stitch1.setObjectName("part_stitch1")
        self.gridLayout_2.addWidget(self.part_stitch1, 0, 2, 1, 1)
        self.all_stitch3 = QtWidgets.QCheckBox(self.resgroupBox)
        self.all_stitch3.setObjectName("all_stitch3")
        self.gridLayout_2.addWidget(self.all_stitch3, 2, 1, 1, 1)
        self.part_stitch6 = QtWidgets.QCheckBox(self.resgroupBox)
        self.part_stitch6.setObjectName("part_stitch6")
        self.gridLayout_2.addWidget(self.part_stitch6, 5, 2, 1, 1)
        self.res2_label = QtWidgets.QLabel(self.resgroupBox)
        self.res2_label.setObjectName("res2_label")
        self.gridLayout_2.addWidget(self.res2_label, 1, 0, 1, 1)
        self.all_stitch1 = QtWidgets.QCheckBox(self.resgroupBox)
        self.all_stitch1.setObjectName("all_stitch1")
        self.gridLayout_2.addWidget(self.all_stitch1, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.resgroupBox, 2, 7, 9, 3)
        self.y_lengthEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.y_lengthEdit.setObjectName("y_lengthEdit")
        self.gridLayout.addWidget(self.y_lengthEdit, 3, 2, 1, 3)
        self.vixel_sizeEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.vixel_sizeEdit.setObjectName("vixel_sizeEdit")
        self.gridLayout.addWidget(self.vixel_sizeEdit, 9, 4, 1, 3)
        self.input_path = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.input_path.setObjectName("input_path")
        self.gridLayout.addWidget(self.input_path, 0, 2, 1, 5)
        self.high_res_input_path = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.high_res_input_path.setObjectName("high_res_input_path")
        self.gridLayout.addWidget(self.high_res_input_path, 0, 8, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.JsonInfoBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 5, 3, 1, 3)
        self.overlapZEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.overlapZEdit.setObjectName("overlapZEdit")
        self.gridLayout.addWidget(self.overlapZEdit, 6, 5, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.JsonInfoBox)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 10, 0, 1, 3)
        self.collabel = QtWidgets.QLabel(self.JsonInfoBox)
        self.collabel.setObjectName("collabel")
        self.gridLayout.addWidget(self.collabel, 12, 4, 1, 1)
        self.collineEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.collineEdit.setObjectName("collineEdit")
        self.gridLayout.addWidget(self.collineEdit, 12, 5, 1, 1)
        self.rowlabel = QtWidgets.QLabel(self.JsonInfoBox)
        self.rowlabel.setObjectName("rowlabel")
        self.gridLayout.addWidget(self.rowlabel, 12, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.JsonInfoBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 4, 5, 1, 1)
        self.rowlineEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.rowlineEdit.setObjectName("rowlineEdit")
        self.gridLayout.addWidget(self.rowlineEdit, 12, 3, 1, 1)
        self.x_lengthEdit = QtWidgets.QLineEdit(self.JsonInfoBox)
        self.x_lengthEdit.setObjectName("x_lengthEdit")
        self.gridLayout.addWidget(self.x_lengthEdit, 3, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.JsonInfoBox)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 7, 5, 1, 1)
        self.horizontalLayout.addWidget(self.JsonInfoBox)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.MIP_high_res_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.MIP_high_res_pushButton.setObjectName("MIP_high_res_pushButton")
        self.gridLayout_3.addWidget(self.MIP_high_res_pushButton, 1, 5, 1, 1)
        self.high_res_x_y_shift_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.high_res_x_y_shift_pushButton.setObjectName("high_res_x_y_shift_pushButton")
        self.gridLayout_3.addWidget(self.high_res_x_y_shift_pushButton, 1, 6, 1, 1)
        self.MIPpreviewButton = QtWidgets.QPushButton(self.centralwidget)
        self.MIPpreviewButton.setObjectName("MIPpreviewButton")
        self.gridLayout_3.addWidget(self.MIPpreviewButton, 1, 2, 1, 1)
        self.HR_stitched_MIP_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.HR_stitched_MIP_pushButton.setObjectName("HR_stitched_MIP_pushButton")
        self.gridLayout_3.addWidget(self.HR_stitched_MIP_pushButton, 2, 5, 1, 1)
        self.HR_z_slice_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.HR_z_slice_pushButton.setObjectName("HR_z_slice_pushButton")
        self.gridLayout_3.addWidget(self.HR_z_slice_pushButton, 3, 1, 1, 1)
        self.high_res_op_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.high_res_op_pushButton.setObjectName("high_res_op_pushButton")
        self.gridLayout_3.addWidget(self.high_res_op_pushButton, 2, 0, 1, 1)
        self.low_res_z_shift_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.low_res_z_shift_pushButton.setObjectName("low_res_z_shift_pushButton")
        self.gridLayout_3.addWidget(self.low_res_z_shift_pushButton, 1, 4, 1, 1)
        self.LR_shift_floatToint_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.LR_shift_floatToint_pushButton.setObjectName("LR_shift_floatToint_pushButton")
        self.gridLayout_3.addWidget(self.LR_shift_floatToint_pushButton, 2, 3, 1, 1)
        self.LR_z_slice_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.LR_z_slice_pushButton.setObjectName("LR_z_slice_pushButton")
        self.gridLayout_3.addWidget(self.LR_z_slice_pushButton, 3, 0, 1, 1)
        self.make_slice_HR_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.make_slice_HR_pushButton.setObjectName("make_slice_HR_pushButton")
        self.gridLayout_3.addWidget(self.make_slice_HR_pushButton, 2, 7, 1, 1)
        self.low_res_op_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.low_res_op_pushButton.setObjectName("low_res_op_pushButton")
        self.gridLayout_3.addWidget(self.low_res_op_pushButton, 2, 1, 1, 1)
        self.output_terafly_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.output_terafly_pushButton.setObjectName("output_terafly_pushButton")
        self.gridLayout_3.addWidget(self.output_terafly_pushButton, 3, 2, 1, 1)
        self.xy_shift_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.xy_shift_pushButton.setObjectName("xy_shift_pushButton")
        self.gridLayout_3.addWidget(self.xy_shift_pushButton, 1, 3, 1, 1)
        self.LR_stitched_MIP_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.LR_stitched_MIP_pushButton.setObjectName("LR_stitched_MIP_pushButton")
        self.gridLayout_3.addWidget(self.LR_stitched_MIP_pushButton, 2, 4, 1, 1)
        self.test_threadpushButton = QtWidgets.QPushButton(self.centralwidget)
        self.test_threadpushButton.setObjectName("test_threadpushButton")
        self.gridLayout_3.addWidget(self.test_threadpushButton, 3, 3, 1, 1)
        self.HR_shift_floatToint_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.HR_shift_floatToint_pushButton.setObjectName("HR_shift_floatToint_pushButton")
        self.gridLayout_3.addWidget(self.HR_shift_floatToint_pushButton, 2, 2, 1, 1)
        self.make_slice_LR_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.make_slice_LR_pushButton.setObjectName("make_slice_LR_pushButton")
        self.gridLayout_3.addWidget(self.make_slice_LR_pushButton, 2, 6, 1, 1)
        self.high_res_z_shift_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.high_res_z_shift_pushButton.setObjectName("high_res_z_shift_pushButton")
        self.gridLayout_3.addWidget(self.high_res_z_shift_pushButton, 1, 7, 1, 1)
        self.MipButton = QtWidgets.QPushButton(self.centralwidget)
        self.MipButton.setObjectName("MipButton")
        self.gridLayout_3.addWidget(self.MipButton, 1, 1, 1, 1)
        self.ArrangementButton = QtWidgets.QPushButton(self.centralwidget)
        self.ArrangementButton.setObjectName("ArrangementButton")
        self.gridLayout_3.addWidget(self.ArrangementButton, 1, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_3)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.update_bar = QtWidgets.QProgressBar(self.centralwidget)
        self.update_bar.setMaximumSize(QtCore.QSize(1132, 16777215))
        self.update_bar.setProperty("value", 24)
        self.update_bar.setObjectName("update_bar")
        self.verticalLayout_3.addWidget(self.update_bar)
        self.progressBar2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar2.setProperty("value", 24)
        self.progressBar2.setObjectName("progressBar2")
        self.verticalLayout_3.addWidget(self.progressBar2)
        self.jsonBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.jsonBrowser.setObjectName("jsonBrowser")
        self.verticalLayout_3.addWidget(self.jsonBrowser)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1216, 23))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionopen = QtWidgets.QAction(MainWindow)
        self.actionopen.setObjectName("actionopen")
        self.menu.addAction(self.actionopen)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.input_path, self.output_path)
        MainWindow.setTabOrder(self.output_path, self.x_lengthEdit)
        MainWindow.setTabOrder(self.x_lengthEdit, self.y_lengthEdit)
        MainWindow.setTabOrder(self.y_lengthEdit, self.overlapXEdit)
        MainWindow.setTabOrder(self.overlapXEdit, self.overlapYEdit)
        MainWindow.setTabOrder(self.overlapYEdit, self.wildnameEdit)
        MainWindow.setTabOrder(self.wildnameEdit, self.vixel_sizeEdit)
        MainWindow.setTabOrder(self.vixel_sizeEdit, self.high_res_input_folder_Button)
        MainWindow.setTabOrder(self.high_res_input_folder_Button, self.high_res_input_path)
        MainWindow.setTabOrder(self.high_res_input_path, self.highest_vixel_sizeEdit)
        MainWindow.setTabOrder(self.highest_vixel_sizeEdit, self.all_stitch1)
        MainWindow.setTabOrder(self.all_stitch1, self.part_stitch1)
        MainWindow.setTabOrder(self.part_stitch1, self.all_stitch2)
        MainWindow.setTabOrder(self.all_stitch2, self.part_stitch2)
        MainWindow.setTabOrder(self.part_stitch2, self.all_stitch3)
        MainWindow.setTabOrder(self.all_stitch3, self.part_stitch3)
        MainWindow.setTabOrder(self.part_stitch3, self.all_stitch4)
        MainWindow.setTabOrder(self.all_stitch4, self.part_stitch4)
        MainWindow.setTabOrder(self.part_stitch4, self.all_stitch5)
        MainWindow.setTabOrder(self.all_stitch5, self.part_stitch5)
        MainWindow.setTabOrder(self.part_stitch5, self.all_stitch6)
        MainWindow.setTabOrder(self.all_stitch6, self.part_stitch6)
        MainWindow.setTabOrder(self.part_stitch6, self.MIPpreviewButton)
        MainWindow.setTabOrder(self.MIPpreviewButton, self.xy_shift_pushButton)
        MainWindow.setTabOrder(self.xy_shift_pushButton, self.low_res_z_shift_pushButton)
        MainWindow.setTabOrder(self.low_res_z_shift_pushButton, self.high_res_x_y_shift_pushButton)
        MainWindow.setTabOrder(self.high_res_x_y_shift_pushButton, self.jsonBrowser)
        MainWindow.setTabOrder(self.jsonBrowser, self.InputJsonButton)
        MainWindow.setTabOrder(self.InputJsonButton, self.CeateJsonButton)
        MainWindow.setTabOrder(self.CeateJsonButton, self.print_jsonButton)
        MainWindow.setTabOrder(self.print_jsonButton, self.input_folderButton)
        MainWindow.setTabOrder(self.input_folderButton, self.output_folderButton)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.InputJsonButton.setText(_translate("MainWindow", "InputJson"))
        self.CeateJsonButton.setText(_translate("MainWindow", "saveJson"))
        self.print_jsonButton.setText(_translate("MainWindow", "print json"))
        self.JsonInfoBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.output_folderButton.setText(_translate("MainWindow", "Output"))
        self.input_folderButton.setText(_translate("MainWindow", "Input"))
        self.label_10.setText(_translate("MainWindow", "vixel_size"))
        self.label_3.setText(_translate("MainWindow", "       x_length"))
        self.label_6.setText(_translate("MainWindow", "       overlapX"))
        self.label_7.setText(_translate("MainWindow", "overlapY"))
        self.high_res_input_folder_Button.setText(_translate("MainWindow", "high_res"))
        self.label_11.setText(_translate("MainWindow", "highest_vixel_size:"))
        self.resgroupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.all_stitch4.setText(_translate("MainWindow", "all_stitch"))
        self.res1_label.setText(_translate("MainWindow", "TextLabel"))
        self.part_stitch3.setText(_translate("MainWindow", "part_stitch"))
        self.all_stitch2.setText(_translate("MainWindow", "all_stitch"))
        self.res5_label.setText(_translate("MainWindow", "TextLabel"))
        self.all_stitch6.setText(_translate("MainWindow", "all_stitch"))
        self.part_stitch4.setText(_translate("MainWindow", "part_stitch"))
        self.res6_label.setText(_translate("MainWindow", "TextLabel"))
        self.res4_label.setText(_translate("MainWindow", "TextLabel"))
        self.res3_label.setText(_translate("MainWindow", "TextLabel"))
        self.part_stitch5.setText(_translate("MainWindow", "part_stitch"))
        self.all_stitch5.setText(_translate("MainWindow", "all_stitch"))
        self.part_stitch2.setText(_translate("MainWindow", "part_stitch"))
        self.part_stitch1.setText(_translate("MainWindow", "part_stitch"))
        self.all_stitch3.setText(_translate("MainWindow", "all_stitch"))
        self.part_stitch6.setText(_translate("MainWindow", "part_stitch"))
        self.res2_label.setText(_translate("MainWindow", "TextLabel"))
        self.all_stitch1.setText(_translate("MainWindow", "all_stitch"))
        self.label_4.setText(_translate("MainWindow", "y_length"))
        self.label_9.setText(_translate("MainWindow", "wildname"))
        self.collabel.setText(_translate("MainWindow", "  col:"))
        self.rowlabel.setText(_translate("MainWindow", "row:"))
        self.label_5.setText(_translate("MainWindow", "      z_length"))
        self.label_8.setText(_translate("MainWindow", "       overlapZ"))
        self.MIP_high_res_pushButton.setText(_translate("MainWindow", "6.createMIP_high_res"))
        self.high_res_x_y_shift_pushButton.setText(_translate("MainWindow", "7.high_res_x_y_shift"))
        self.MIPpreviewButton.setText(_translate("MainWindow", "3.MIP preview"))
        self.HR_stitched_MIP_pushButton.setText(_translate("MainWindow", "13.HR-stitched-MIP"))
        self.HR_z_slice_pushButton.setText(_translate("MainWindow", "17.HR-z_slice-result"))
        self.high_res_op_pushButton.setText(_translate("MainWindow", "9.high_res_optimazation"))
        self.low_res_z_shift_pushButton.setText(_translate("MainWindow", "5.low_res_z_shift"))
        self.LR_shift_floatToint_pushButton.setText(_translate("MainWindow", "12.LR-shift_floatToint"))
        self.LR_z_slice_pushButton.setText(_translate("MainWindow", "16.LR-z_slice-result"))
        self.make_slice_HR_pushButton.setText(_translate("MainWindow", "15.make_slice_HR"))
        self.low_res_op_pushButton.setText(_translate("MainWindow", "10.low_res_optimazation"))
        self.output_terafly_pushButton.setText(_translate("MainWindow", "18.output_terafly"))
        self.xy_shift_pushButton.setText(_translate("MainWindow", "4.low_res_x_y_shift"))
        self.LR_stitched_MIP_pushButton.setText(_translate("MainWindow", "13.LR-stitched-MIP"))
        self.test_threadpushButton.setText(_translate("MainWindow", "test_thread"))
        self.HR_shift_floatToint_pushButton.setText(_translate("MainWindow", "11.HR-shift_floatToint"))
        self.make_slice_LR_pushButton.setText(_translate("MainWindow", "14.make_slice_LR"))
        self.high_res_z_shift_pushButton.setText(_translate("MainWindow", "8.high_res_z_shift"))
        self.MipButton.setText(_translate("MainWindow", "2.createMIP_low_res"))
        self.ArrangementButton.setText(_translate("MainWindow", "1.Arrangement"))
        self.menu.setTitle(_translate("MainWindow", "file"))
        self.actionopen.setText(_translate("MainWindow", "open"))