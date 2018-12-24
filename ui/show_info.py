# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'show_info.ui'
# Created by: PyQt5 UI code generator 5.9
# WARNING! All changes made in this file will be lost!
from PyQt5 import QtWidgets,QtCore

from PyQt5.QtWidgets import QHeaderView
class Ui_ShowInfoWindow(QtWidgets.QMainWindow):
    def setupUi(self, ShowInfoWindow):
        ShowInfoWindow.setObjectName("ShowInfoWindow")
        ShowInfoWindow.resize(1000, 350)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(ShowInfoWindow.sizePolicy().hasHeightForWidth())
        ShowInfoWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(ShowInfoWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setShowGrid(False)
        self.tableWidget.setRowCount(9)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()
        self.verticalLayout.addWidget(self.tableWidget)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        ShowInfoWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ShowInfoWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 309, 26))
        self.menubar.setObjectName("menubar")
        ShowInfoWindow.setMenuBar(self.menubar)

        self.retranslateUi(ShowInfoWindow)
        QtCore.QMetaObject.connectSlotsByName(ShowInfoWindow)
    def retranslateUi(self, ShowInfoWindow):
        _translate = QtCore.QCoreApplication.translate
        ShowInfoWindow.setWindowTitle(_translate("ShowInfoWindow", "统计结果"))
        self.pushButton_2.setText(_translate("ShowInfoWindow", "箱型图"))
        self.pushButton.setText(_translate("ShowInfoWindow", "保存"))
