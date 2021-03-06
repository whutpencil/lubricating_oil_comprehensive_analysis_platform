# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_interface1.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

class ComboCheckBox(QComboBox):
    def __init__(self):  # items==[str,str...]
        #super(ComboCheckBox, self).__init__()
        QComboBox.__init__(self)
        self.items = ['全选']
        self.row_num = 1
        self.Selectedrow_num = 0
        self.qCheckBox = []
        self.qLineEdit = QLineEdit()
        self.qLineEdit.setReadOnly(True)
        self.qListWidget = QListWidget()
        self.addQCheckBox(0)
        self.setModel(self.qListWidget.model())
        self.setView(self.qListWidget)
        self.setLineEdit(self.qLineEdit)

    def addItems(self, items):
        self.items = items
        self.items.insert(0, '全选')
        self.row_num = len(self.items)
        self.Selectedrow_num = 0
        self.qCheckBox = []
        self.qLineEdit = QLineEdit()
        self.qLineEdit.setReadOnly(True)
        self.qListWidget = QListWidget()
        self.addQCheckBox(0)
        self.qCheckBox[0].stateChanged.connect(self.All)
        for i in range(1, self.row_num):
            self.addQCheckBox(i)
            self.qCheckBox[i].stateChanged.connect(self.show)
        self.setModel(self.qListWidget.model())
        self.setView(self.qListWidget)
        self.setLineEdit(self.qLineEdit)

    def addQCheckBox(self, i):
        self.qCheckBox.append(QCheckBox())
        qItem = QListWidgetItem(self.qListWidget)
        self.qCheckBox[i].setText(self.items[i])
        self.qListWidget.setItemWidget(qItem, self.qCheckBox[i])

    def Selectlist(self):
        Outputlist = []
        for i in range(1, self.row_num):
            if self.qCheckBox[i].isChecked() == True:
                Outputlist.append(self.qCheckBox[i].text())
        self.Selectedrow_num = len(Outputlist)
        return Outputlist

    def show(self):
        show = ''
        Outputlist = self.Selectlist()
        self.qLineEdit.setReadOnly(False)
        self.qLineEdit.clear()
        for i in Outputlist:
            show += i + ';'
        if self.Selectedrow_num == 0:
            self.qCheckBox[0].setCheckState(0)
        elif self.Selectedrow_num == self.row_num - 1:
            self.qCheckBox[0].setCheckState(2)
        else:
            self.qCheckBox[0].setCheckState(1)
        self.qLineEdit.setText(show)
        self.qLineEdit.setReadOnly(True)

    def All(self, status):
        if status == 2:
            for i in range(1, self.row_num):
                self.qCheckBox[i].setChecked(True)
        elif status == 1:
            if self.Selectedrow_num == 0:
                self.qCheckBox[0].setCheckState(2)
        elif status == 0:
            self.clear()


    def clear(self):
        for i in range(self.row_num):
            self.qCheckBox[i].setChecked(False)
class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(900, 950)
        mainWindow.setMouseTracking(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/logo.png"), QtGui.QIcon.Selected, QtGui.QIcon.Off)
        mainWindow.setWindowIcon(icon)
        mainWindow.setIconSize(QtCore.QSize(40, 40))
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_4.addWidget(self.label_4)
        self.line_2 = QtWidgets.QFrame(self.tab)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_4.addWidget(self.line_2)
        self.tableWidget = QtWidgets.QTableWidget(self.tab)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.verticalLayout_4.addWidget(self.tableWidget)
        self.horizontalLayout_2.addLayout(self.verticalLayout_4)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_3 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)

        # 测试用
        self.comboCheckBox = ComboCheckBox()
        self.verticalLayout.addWidget(self.comboCheckBox)

        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_2.setSpacing(7)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_format_adjust = QtWidgets.QPushButton(self.tab)
        self.pushButton_format_adjust.setObjectName("pushButton_format_adjust")
        self.gridLayout_2.addWidget(self.pushButton_format_adjust, 0, 0, 1, 1)
        self.pushButton_drop_params = QtWidgets.QPushButton(self.tab)
        self.pushButton_drop_params.setObjectName("pushButton_drop_params")
        self.gridLayout_2.addWidget(self.pushButton_drop_params, 2, 0, 1, 1)
        self.pushButton_restore_data = QtWidgets.QPushButton(self.tab)
        self.pushButton_restore_data.setObjectName("pushButton_restore_data")
        self.gridLayout_2.addWidget(self.pushButton_restore_data, 2, 1, 1, 1)
        self.pushButton_show_info = QtWidgets.QPushButton(self.tab)
        self.pushButton_show_info.setObjectName("pushButton_show_info")
        self.gridLayout_2.addWidget(self.pushButton_show_info, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_3.setSpacing(5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label_5 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout.addWidget(self.label_5)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pushButton_10 = QtWidgets.QPushButton(self.tab)
        self.pushButton_10.setObjectName("pushButton_10")
        self.gridLayout_3.addWidget(self.pushButton_10, 1, 0, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.tab)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_3.addWidget(self.pushButton_3, 2, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.tab)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_3.addWidget(self.pushButton_4, 2, 1, 1, 1)
        self.pushButton_savefile = QtWidgets.QPushButton(self.tab)
        self.pushButton_savefile.setObjectName("pushButton_savefile")
        self.gridLayout_3.addWidget(self.pushButton_savefile, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_3)
        self.label_6 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.pushButton_draw_hist = QtWidgets.QPushButton(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_draw_hist.sizePolicy().hasHeightForWidth())
        self.pushButton_draw_hist.setSizePolicy(sizePolicy)
        self.pushButton_draw_hist.setAutoDefault(False)
        self.pushButton_draw_hist.setDefault(False)
        self.pushButton_draw_hist.setFlat(False)
        self.pushButton_draw_hist.setObjectName("pushButton_draw_hist")
        self.gridLayout_4.addWidget(self.pushButton_draw_hist, 0, 0, 1, 1)
        self.pushButton_draw_trend = QtWidgets.QPushButton(self.tab)
        self.pushButton_draw_trend.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_draw_trend.sizePolicy().hasHeightForWidth())
        self.pushButton_draw_trend.setSizePolicy(sizePolicy)
        self.pushButton_draw_trend.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pushButton_draw_trend.setMouseTracking(False)
        self.pushButton_draw_trend.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_draw_trend.setAutoFillBackground(True)
        self.pushButton_draw_trend.setObjectName("pushButton_draw_trend")
        self.gridLayout_4.addWidget(self.pushButton_draw_trend, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_4)
        self.label_2 = QtWidgets.QLabel(self.tab)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.textBrowser = QtWidgets.QTextBrowser(self.tab)
        self.textBrowser.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout.setStretch(0, 1)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.horizontalLayout_2.setStretch(0, 5)
        self.horizontalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.label_8 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.verticalLayout_2.addWidget(self.label_8)
        self.line_5 = QtWidgets.QFrame(self.tab)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout_2.addWidget(self.line_5)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.verticalLayout_2.addLayout(self.gridLayout_5)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setEnabled(True)
        self.label.setMinimumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.textEdit = QtWidgets.QTextEdit(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy)
        self.textEdit.setMaximumSize(QtCore.QSize(16777215, 30))
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayout_4.addWidget(self.textEdit)
        self.pushButton_openfile = QtWidgets.QPushButton(self.tab)
        self.pushButton_openfile.setFlat(False)
        self.pushButton_openfile.setObjectName("pushButton_openfile")
        self.horizontalLayout_4.addWidget(self.pushButton_openfile)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.verticalLayout_2.setStretch(3, 10)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(110, 140, 671, 331))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_13 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.verticalLayout_5.addWidget(self.label_13)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_7 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_6.addWidget(self.label_7)
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget_7)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_6.addWidget(self.comboBox)
        self.verticalLayout_5.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_9 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_7.addWidget(self.label_9)
        self.comboBox_2 = QtWidgets.QComboBox(self.horizontalLayoutWidget_7)
        self.comboBox_2.setObjectName("comboBox_2")
        self.horizontalLayout_7.addWidget(self.comboBox_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_10 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_8.addWidget(self.label_10)
        self.comboBox_3 = QtWidgets.QComboBox(self.horizontalLayoutWidget_7)
        self.comboBox_3.setObjectName("comboBox_3")
        self.horizontalLayout_8.addWidget(self.comboBox_3)
        self.verticalLayout_5.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_11 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_9.addWidget(self.label_11)
        self.comboBox_4 = QtWidgets.QComboBox(self.horizontalLayoutWidget_7)
        self.comboBox_4.setObjectName("comboBox_4")
        self.horizontalLayout_9.addWidget(self.comboBox_4)
        self.verticalLayout_5.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_12 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_10.addWidget(self.label_12)
        self.comboBox_5 = QtWidgets.QComboBox(self.horizontalLayoutWidget_7)
        self.comboBox_5.setObjectName("comboBox_5")
        self.horizontalLayout_10.addWidget(self.comboBox_5)
        self.verticalLayout_5.addLayout(self.horizontalLayout_10)
        self.verticalLayout_5.setStretch(0, 1)
        self.verticalLayout_5.setStretch(1, 1)
        self.verticalLayout_5.setStretch(2, 1)
        self.verticalLayout_5.setStretch(3, 1)
        self.verticalLayout_5.setStretch(4, 1)
        self.verticalLayout_5.setStretch(5, 1)
        self.horizontalLayout_11.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_14 = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(10)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.verticalLayout_6.addWidget(self.label_14)
        self.textEdit_2 = QtWidgets.QTextBrowser(self.horizontalLayoutWidget_7)
        self.textEdit_2.setObjectName("textBrowser_2")
        self.verticalLayout_6.addWidget(self.textEdit_2)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("color: rgb(0, 0, 255);")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_12.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_12.addWidget(self.pushButton_2)
        self.checkBox = QtWidgets.QCheckBox(self.horizontalLayoutWidget_7)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(12)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_12.addWidget(self.checkBox)
        self.verticalLayout_6.addLayout(self.horizontalLayout_12)
        self.verticalLayout_6.setStretch(0, 1)
        self.verticalLayout_6.setStretch(1, 5)
        self.verticalLayout_6.setStretch(2, 1)
        self.horizontalLayout_11.addLayout(self.verticalLayout_6)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout)
        mainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(mainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 900, 26))
        self.menuBar.setObjectName("menuBar")
        self.menu = QtWidgets.QMenu(self.menuBar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menuBar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menuBar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menuBar)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menuBar)
        self.menu_5.setObjectName("menu_5")
        self.menu_6 = QtWidgets.QMenu(self.menuBar)
        self.menu_6.setObjectName("menu_6")
        self.menu_7 = QtWidgets.QMenu(self.menuBar)
        self.menu_7.setObjectName("menu_7")
        self.menu_8 = QtWidgets.QMenu(self.menuBar)
        self.menu_8.setObjectName("menu_8")
        mainWindow.setMenuBar(self.menuBar)
        self.action_normalization_data = QtWidgets.QAction(mainWindow)
        self.action_normalization_data.setObjectName("action_normalization_data")
        self.action_covariance_matrix = QtWidgets.QAction(mainWindow)
        self.action_covariance_matrix.setObjectName("action_covariance_matrix")
        self.action_covariance_matrix_heatmap = QtWidgets.QAction(mainWindow)
        self.action_covariance_matrix_heatmap.setObjectName("action_covariance_matrix_heatmap")
        self.action_eigenvalue = QtWidgets.QAction(mainWindow)
        self.action_eigenvalue.setObjectName("action_eigenvalue")
        self.action_eigenvector = QtWidgets.QAction(mainWindow)
        self.action_eigenvector.setObjectName("action_eigenvector")
        self.action_scree_plot = QtWidgets.QAction(mainWindow)
        self.action_scree_plot.setObjectName("action_scree_plot")
        self.action_total_variance_explained = QtWidgets.QAction(mainWindow)
        self.action_total_variance_explained.setObjectName("action_total_variance_explained")
        self.action_factorial_load_matrix = QtWidgets.QAction(mainWindow)
        self.action_factorial_load_matrix.setObjectName("action_factorial_load_matrix")
        self.action_9 = QtWidgets.QAction(mainWindow)
        self.action_9.setObjectName("action_9")
        self.action_openfile = QtWidgets.QAction(mainWindow)
        self.action_openfile.setObjectName("action_openfile")
        self.action_savefile = QtWidgets.QAction(mainWindow)
        self.action_savefile.setObjectName("action_savefile")
        self.action_exit = QtWidgets.QAction(mainWindow)
        self.action_exit.setObjectName("action_exit")
        self.action_select_rangedata = QtWidgets.QAction(mainWindow)
        self.action_select_rangedata.setObjectName("action_select_rangedata")
        self.action_stationary_detaction = QtWidgets.QAction(mainWindow)
        self.action_stationary_detaction.setObjectName("action_stationary_detaction")
        self.action_white_noise = QtWidgets.QAction(mainWindow)
        self.action_white_noise.setObjectName("action_white_noise")
        self.action_plot_acfandpacf = QtWidgets.QAction(mainWindow)
        self.action_plot_acfandpacf.setObjectName("action_plot_acfandpacf")
        self.action_model_order_training = QtWidgets.QAction(mainWindow)
        self.action_model_order_training.setObjectName("action_model_order_training")
        self.action_grid = QtWidgets.QAction(mainWindow)
        self.action_grid.setObjectName("action_grid")
        self.action_19 = QtWidgets.QAction(mainWindow)
        self.action_19.setObjectName("action_19")
        self.action_bins = QtWidgets.QAction(mainWindow)
        self.action_bins.setObjectName("action_bins")
        self.action_model_residual_diagnosis = QtWidgets.QAction(mainWindow)
        self.action_model_residual_diagnosis.setObjectName("action_model_residual_diagnosis")
        self.action_output_model_report = QtWidgets.QAction(mainWindow)
        self.action_output_model_report.setObjectName("action_output_model_report")
        self.action_arima_model_predict = QtWidgets.QAction(mainWindow)
        self.action_arima_model_predict.setObjectName("action_arima_model_predict")
        self.menu.addAction(self.action_openfile)
        self.menu.addAction(self.action_savefile)
        self.menu.addAction(self.action_exit)
        self.menu_2.addAction(self.action_normalization_data)
        self.menu_2.addAction(self.action_covariance_matrix)
        self.menu_2.addAction(self.action_covariance_matrix_heatmap)
        self.menu_2.addAction(self.action_eigenvalue)
        self.menu_2.addAction(self.action_eigenvector)
        self.menu_2.addAction(self.action_scree_plot)
        self.menu_2.addAction(self.action_total_variance_explained)
        self.menu_2.addAction(self.action_factorial_load_matrix)
        self.menu_3.addAction(self.action_select_rangedata)
        self.menu_3.addAction(self.action_stationary_detaction)
        self.menu_3.addAction(self.action_white_noise)
        self.menu_3.addAction(self.action_plot_acfandpacf)
        self.menu_3.addAction(self.action_model_order_training)
        self.menu_3.addAction(self.action_model_residual_diagnosis)
        self.menu_3.addAction(self.action_output_model_report)
        self.menu_3.addAction(self.action_arima_model_predict)
        self.menu_7.addAction(self.action_grid)
        self.menu_7.addAction(self.action_bins)
        self.menu_8.addAction(self.action_19)
        self.menuBar.addAction(self.menu.menuAction())
        self.menuBar.addAction(self.menu_2.menuAction())
        self.menuBar.addAction(self.menu_3.menuAction())
        self.menuBar.addAction(self.menu_4.menuAction())
        self.menuBar.addAction(self.menu_5.menuAction())
        self.menuBar.addAction(self.menu_6.menuAction())
        self.menuBar.addAction(self.menu_7.menuAction())
        self.menuBar.addAction(self.menu_8.menuAction())

        self.retranslateUi(mainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "油液理化性能离线分析系统"))
        self.label_4.setText(_translate("mainWindow", "数据区"))
        self.label_3.setText(_translate("mainWindow", "指标与数据清洗"))
        self.pushButton_format_adjust.setToolTip(_translate("mainWindow", "请输入要去除的符号，便于数值化分析"))
        self.pushButton_format_adjust.setText(_translate("mainWindow", "数据格式调整"))
        self.pushButton_drop_params.setText(_translate("mainWindow", "删除选中指标"))
        self.pushButton_restore_data.setToolTip(_translate("mainWindow", "若数据量较大，则恢复原始数据的过程较慢"))
        self.pushButton_restore_data.setText(_translate("mainWindow", "恢复原始数据"))
        self.pushButton_show_info.setText(_translate("mainWindow", "指标信息分布"))
        self.label_5.setText(_translate("mainWindow", "数据分析处理"))
        self.pushButton_10.setText(_translate("mainWindow", "单指标三线值"))
        self.pushButton_3.setText(_translate("mainWindow", "一键算法对比"))
        self.pushButton_4.setText(_translate("mainWindow", "算法误差分析"))
        self.pushButton_savefile.setToolTip(_translate("mainWindow", "导出数据区的数据"))
        self.pushButton_savefile.setText(_translate("mainWindow", "导出计算结果"))
        self.label_6.setText(_translate("mainWindow", "数据可视化"))
        self.pushButton_draw_hist.setToolTip(_translate("mainWindow", "由于图幅限制，最多选择两个指标！"))
        self.pushButton_draw_hist.setText(_translate("mainWindow", "绘制直方图"))
        self.pushButton_draw_trend.setToolTip(_translate("mainWindow", "若超过2个指标，请选择数值范围相近的列"))
        self.pushButton_draw_trend.setText(_translate("mainWindow", "绘制走势图"))
        self.label_2.setText(_translate("mainWindow", "操作记录："))
        self.label_8.setText(_translate("mainWindow", "作图区"))
        self.label.setText(_translate("mainWindow", "文件路径:"))
        self.pushButton_openfile.setText(_translate("mainWindow", "打开"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("mainWindow", "数据分析"))
        self.label_13.setText(_translate("mainWindow", "串口设置："))
        self.label_7.setText(_translate("mainWindow", "串口号"))
        self.label_9.setText(_translate("mainWindow", "波特率"))
        self.label_10.setText(_translate("mainWindow", "数据位"))
        self.label_11.setText(_translate("mainWindow", "停止位"))
        self.label_12.setText(_translate("mainWindow", "校验位"))
        self.label_14.setText(_translate("mainWindow", "接收数据区："))
        self.pushButton.setText(_translate("mainWindow", "打开串口"))
        self.pushButton_2.setText(_translate("mainWindow", "刷新串口设备"))
        self.checkBox.setText(_translate("mainWindow", "HEX显示"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("mainWindow", "设置"))
        self.menu.setTitle(_translate("mainWindow", "文件"))
        self.menu_2.setTitle(_translate("mainWindow", "主成分分析"))
        self.menu_3.setTitle(_translate("mainWindow", "ARIMA模型"))
        self.menu_4.setTitle(_translate("mainWindow", "控制限"))
        self.menu_5.setTitle(_translate("mainWindow", "函数拟合"))
        self.menu_6.setTitle(_translate("mainWindow", "神经网络算法"))
        self.menu_7.setTitle(_translate("mainWindow", "制图参数"))
        self.menu_8.setTitle(_translate("mainWindow", "帮助"))
        self.action_normalization_data.setText(_translate("mainWindow", "数据标准化"))
        self.action_covariance_matrix.setText(_translate("mainWindow", "相关系数矩阵"))
        self.action_covariance_matrix_heatmap.setText(_translate("mainWindow", "相关系数热力图"))
        self.action_eigenvalue.setText(_translate("mainWindow", "特征值"))
        self.action_eigenvector.setText(_translate("mainWindow", "特征向量"))
        self.action_scree_plot.setText(_translate("mainWindow", "碎石图"))
        self.action_total_variance_explained.setText(_translate("mainWindow", "总方差解释"))
        self.action_factorial_load_matrix.setText(_translate("mainWindow", "因子载荷矩阵"))
        self.action_9.setText(_translate("mainWindow", "碎石图"))
        self.action_openfile.setText(_translate("mainWindow", "打开"))
        self.action_openfile.setShortcut(_translate("mainWindow", "Ctrl+O"))
        self.action_savefile.setText(_translate("mainWindow", "保存"))
        self.action_savefile.setShortcut(_translate("mainWindow", "Ctrl+S"))
        self.action_exit.setText(_translate("mainWindow", "退出"))
        self.action_select_rangedata.setText(_translate("mainWindow", "数据段选取"))
        self.action_stationary_detaction.setText(_translate("mainWindow", "平稳性检测"))
        self.action_white_noise.setText(_translate("mainWindow", "白噪声检测"))
        self.action_plot_acfandpacf.setText(_translate("mainWindow", "自相关和偏相关图"))
        self.action_model_order_training.setText(_translate("mainWindow", "模型阶数训练"))
        self.action_grid.setText(_translate("mainWindow", "网格线√"))
        self.action_19.setText(_translate("mainWindow", "网格线"))
        self.action_bins.setText(_translate("mainWindow", "直方图条数"))
        self.action_model_residual_diagnosis.setText(_translate("mainWindow", "模型残差诊断图"))
        self.action_output_model_report.setText(_translate("mainWindow", "输出模型报告"))
        self.action_arima_model_predict.setText(_translate("mainWindow", "模型预测图"))

