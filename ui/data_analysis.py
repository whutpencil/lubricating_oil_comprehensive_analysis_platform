#!/usr/bin/python3
# -*- coding:utf-8 -*-
#
# #Author:Hu Biao
#
# @Time :2018/12/19 14:54

import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import time
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import QTimer
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
sns.set_style('white')
matplotlib.use("Qt5Agg")  # 声明使用QT5
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from ui.show_info import Ui_ShowInfoWindow
from ui.main_interface1 import Ui_mainWindow
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import traceback

class Mainwindow(QMainWindow,Ui_mainWindow):
    def __init__(self, parent=None):
        super(Mainwindow,self).__init__(parent)
        self.figure = plt.figure(figsize=(12, 10), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.setupUi(self)
        self.gridLayout_5.addWidget(self.canvas)  ##添加画布
        self.horizontalLayout_4.addWidget(self.toolbar) ##添加toolbar

        # 串口无效
        self.ser = None
        self.receive_num = 0
        # 记录最后发送的回车字符的变量
        self.rcv_enter = ''
        # 显示发送与接收的字符数量
        #dis = '  接收:' + '{:d}'.format(self.receive_num)
        #self.statusBar.showMessage(dis)
        # 刷新一下串口的列表
        self.refresh()

        # 波特率
        self.comboBox_2.addItem('115200')
        self.comboBox_2.addItem('57600')
        self.comboBox_2.addItem('56000')
        self.comboBox_2.addItem('38400')
        self.comboBox_2.addItem('19200')
        self.comboBox_2.addItem('14400')
        self.comboBox_2.addItem('9600')
        self.comboBox_2.addItem('4800')
        self.comboBox_2.addItem('2400')
        self.comboBox_2.addItem('1200')
        # 数据位
        self.comboBox_3.addItem('8')
        self.comboBox_3.addItem('7')
        self.comboBox_3.addItem('6')
        self.comboBox_3.addItem('5')
        # 停止位
        self.comboBox_4.addItem('1')
        self.comboBox_4.addItem('1.5')
        self.comboBox_4.addItem('2')
        # 校验位
        self.comboBox_5.addItem('NONE')
        self.comboBox_5.addItem('ODD')
        self.comboBox_5.addItem('EVEN')
        # 对testEdit进行事件过滤
        self.textEdit_2.installEventFilter(self)
        self.timer = QTimer(self)  ##实例化一个定时器
        self.timer.timeout.connect(self.recv)  ##定时器调用读取串口接收数据

        # 窗口样式表文件读取
        # sshFile = "three.qss"
        # with open(sshFile, "r") as fh:
        #     self.setStyleSheet(fh.read())

        ####参数意义
        #self.input_table  #打开文件后的输入数据，index为日期
        #self.self.path  #文件路径
        #self.info   #data.describe()信息
        #self.params_list #选中指标信息
        #self.params_list_tostr  #选中指标连成一个字符串
        #self.select_count  #选中指标个数
        ###参数初始化
        self.plot_grid = True  #网格线
        self.diff = None

        ###################初始化show_info窗口#####################
        self.show_info_window = QtWidgets.QMainWindow()
        self.ShowInfoWindow = Ui_ShowInfoWindow()
        self.ShowInfoWindow.setupUi(self.show_info_window)
        self.ShowInfoWindow.pushButton_2.clicked.connect(self.draw_boxplot)
        self.ShowInfoWindow.pushButton.clicked.connect(self.show_info_write_to_file)

        #####################以下是串口绑定事件####################
        self.pushButton_serial_openoff.clicked.connect(self.open_close)  ##打开关闭串口按钮
        self.pushButton_serial_refresh.clicked.connect(self.refresh)     ##刷新串口外设按钮
        self.pushButton_serial_clean.clicked.connect(self.clear)  ##刷新串口外设按钮
        self.comboBox_2.activated.connect(self.baud_modify)       ##波特率修改
        self.comboBox.activated.connect(self.com_modify)         ##串口号修改
        self.open_close(True)                                ##执行一下打开串口
        self.pushButton_serial_openoff.setChecked(True)      ##将按钮初始化值为True
        #####################以下是绑定事件########################
        self.pushButton_openfile.clicked.connect(self.openfile) ##绑定打开按钮
        self.pushButton_show_info.clicked.connect(self.show_info) ##绑定指标分析按钮pushButton
        self.pushButton_savefile.clicked.connect(self.main_write_to_file)  ##绑定导出计算结果按钮
        self.pushButton_draw_trend.clicked.connect(self.draw_picture)   ##绘制走势图
        self.pushButton_drop_params.clicked.connect(self.drop_params)   ##删除指标
        self.pushButton_format_adjust.clicked.connect(self.format_adjust)   ##数据格式调整
        self.pushButton_restore_data.clicked.connect(self.restore_data)  ##恢复数据
        self.pushButton_draw_hist.clicked.connect(self.draw_distplot)  ##绘制直方图

        ###绑定主成分分析
        self.action_normalization_data.triggered.connect(self.data_standardization)
        self.action_covariance_matrix.triggered.connect(self.covariance_matrix)
        self.action_covariance_matrix_heatmap.triggered.connect(self.covariance_matrix_heatmap)
        self.action_eigenvalue.triggered.connect(self.eigenvalue)
        self.action_eigenvector.triggered.connect(self.eigenvector)
        self.action_scree_plot.triggered.connect(self.scree_plot)
        self.action_total_variance_explained.triggered.connect(self.total_variance_explained)
        self.action_factorial_load_matrix.triggered.connect(self.factorial_load_matrix)

        ###绑定其他按钮
        self.action_openfile.triggered.connect(self.openfile) ##绑定打开按钮
        self.action_savefile.triggered.connect(self.main_write_to_file)  ##绑定保存按钮
        self.action_exit.triggered.connect(QCoreApplication.instance().quit)   ##绑定退出按钮
        self.action_grid.triggered.connect(self.turn_on_off_grid)  ##绑定网格线打开按钮

        ###绑定ARIMA模型
        self.action_select_rangedata.triggered.connect(self.select_range_data)  ##绑定选取数据段按钮
        self.action_stationary_detaction.triggered.connect(self.stationary_detection)  ##平稳检测
        self.action_white_noise.triggered.connect(self.white_noise_detection)   ##白噪声检测
        self.action_plot_acfandpacf.triggered.connect(self.plot_acfandpacf)    ##自相关和偏相关
        self.action_model_order_training.triggered.connect(self.arima_chose_pq)  ##模型阶数训练
        self.action_model_residual_diagnosis.triggered.connect(self.arima_model_diagnosis)  ##模型诊断
        self.action_output_model_report.triggered.connect(self.arima_model_summary)    ##输出模型报告
        self.action_arima_model_predict.triggered.connect(self.arima_result_visualization)  ##arima预测数据


    ###主成分分析模块==========
    def data_standardization(self): ##数据标准化
        self.check_check_box()
        try:
            if self.select_count < 4:
                print('请选择一个指标')
                QMessageBox.warning(self, "注意", "请选择至少4个指标!")
            else:
                std = StandardScaler()
                self.data_std=std.fit_transform(self.input_table[self.params_list])
                self.data_std_pd=pd.DataFrame(self.data_std,columns=self.params_list)
                self.write_to_table(self.data_std_pd)
                self.label_4.setText("标准化后的数据")
                self.display('对'+self.params_list_tostr+'进行标准化')

        except:
            return
    def covariance_matrix(self): ##相关系数矩阵
        try:
            self.covariance_matrix=np.corrcoef(self.data_std.T)
        except:
            QMessageBox.warning(self,"警告","无法求出协方差矩阵！")
            return
        self.covariance_matrix_pd=pd.DataFrame(self.covariance_matrix,index=self.params_list,columns=self.params_list)
        self.write_to_table(self.covariance_matrix_pd)
        self.label_4.setText("相关系数矩阵")
        self.display('求' + self.params_list_tostr + '的相关系数矩阵')
    def covariance_matrix_heatmap(self):     ##相关系数热力图
        f, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(self.covariance_matrix_pd, cmap="YlGnBu", cbar=True, annot=True,
                    square=True, fmt='.2f', annot_kws={'size': 10},ax=ax)
        #ax.xaxis.tick_top() 将x轴放上面显示
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
        ax.invert_yaxis()
        ax.set_title("相关系数热力图",fontsize=20)
        f.show()
        self.display('作' + self.params_list_tostr + '相关系数热力图')
        return
    def eigenvalue(self): ##求特征值
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=len(self.params_list))
        self.eigenvector=self.pca.fit_transform(self.data_std)
        self.eigenvalue=self.pca.explained_variance_
        #self.eigenvalue,self.eigenvector=np.linalg.eig(self.covariance_matrix)   #因为求特征值不准，所以不用了
        self.eigenvalue_pd = pd.DataFrame(self.eigenvalue,columns=['特征值'])
        self.write_to_table(self.eigenvalue_pd)
        self.label_4.setText("特征值")
        self.display('求' + self.params_list_tostr + '相关系数矩阵的特征值')
    def eigenvector(self): ##求特征向量
        index = []
        for i in range(1, self.eigenvector.shape[1] + 1):
            index.append(str(i))
        self.eigenvector_pd=pd.DataFrame(self.eigenvector,columns=index)
        self.write_to_table(self.eigenvector_pd.round(4))
        self.label_4.setText("特征向量")
        self.display('求' + self.params_list_tostr + '相关系数矩阵的特征向量')
    def scree_plot(self):      ##碎石图
        f,ax = plt.subplots(figsize=(12, 9))
        ax.plot(self.eigenvalue, 'ro-', markersize=8)
        ax.set_title("Scree Plot", fontsize=20)
        ax.set_xlabel('pca_nums', fontsize=12)
        ax.set_ylabel('values', fontsize=12)
        ax.grid(True, linestyle='--', color='black', linewidth=0.5)
        f.show()
        self.display('作碎石图')
        return
    def total_variance_explained(self):     ##总方差解释
        import traceback
        eigvalue_num=len(self.params_list)
        self.eig_pairs = [(np.abs(self.eigenvalue[i]), self.eigenvector[:,i]) for i in range(eigvalue_num)]
        self.eig_pairs.sort(reverse=True)
        total_eigvalue=sum(self.eigenvalue)
        table=[([0.0] * 3) for i in range(eigvalue_num)]
        table=np.array(table)
        accumulate = 0.0
        for i in range(eigvalue_num):
            #各成分特征值
            table[i][0]=self.eig_pairs[i][0]
            #方差占比
            table[i][1]=self.eig_pairs[i][0]/total_eigvalue
            #累积方差
            accumulate += table[i][1]
            table[i][2]=accumulate
        # np.around(table[1:3],decimals=3)
        columns=['特征值','方差的%','累积%']
        table_pd=pd.DataFrame(table,columns=columns)
        self.table_pd=table_pd.round(3)
        self.write_to_table(self.table_pd)
        self.label_4.setText("解释的总方差")
        self.display('求解释总方差')
        return
    def factorial_load_matrix(self):     ##因子载荷矩阵
        try:
            pca_num,ok = QInputDialog.getText(self, "主成分个数选取", "输入选择的主成分个数:", QLineEdit.Normal)
            pca_num=int(pca_num)
            if ok and (1 < pca_num) & (pca_num <= len(self.params_list)):
                col = []
                columns=[]
                for i in range(pca_num):
                    from math import sqrt
                    col.append(list(sqrt(self.eigenvalue[0])*self.eigenvector[1]))
                    columns.append("factor"+str(i+1))
                component_matrix = pd.DataFrame(col).T
                component_matrix.columns = columns
                self.write_to_table(component_matrix)
                self.label_4.setText(str(pca_num)+"个主成分的因子载荷矩阵")
                self.display('求'+str(pca_num)+'个主成分的因子载荷矩阵')
                return
        except:
            return

    ####ARIMA模块==============
    def stationary_detection(self):
        self.check_check_box()
        try:
            if self.select_count != 1:
                QMessageBox.warning(self, "注意", "只能选择1个指标!")
                return
            else:
                self.diff = 0
                self.diff_data = self.input_table_select[self.params_list[0]]
                adftest=adfuller(self.diff_data)
                while adftest[1] > 0.05:
                    self.diff += 1
                    self.diff_data=self.diff_data.diff(1).dropna()
                    adftest = adfuller(self.diff_data, autolag='AIC')
                self.display(u'原始序列经过%s阶差分后归于平稳，p值为%s' %(self.diff, adftest[1]))
                plt.clf()
                ax = self.figure.add_subplot(111)
                plt.tight_layout(pad=0.1, w_pad=0.1)
                plt.subplots_adjust(right=0.93, left=0.07, bottom=0.13)
                #.plot(ax=ax)
                self.diff_data.plot(ax=ax)
                if self.plot_grid:
                    ax.grid(linestyle='--', color='black', linewidth=0.4)
                self.canvas.draw()
                self.label_8.setText(str(self.diff) + '阶差分后的时序图')
                self.display('绘制' + str(self.diff) + '阶差分后的时序图')
        except:
            return
    def white_noise_detection(self):
        try:
            # 白噪声检测
            [[lb], [p]] = acorr_ljungbox(self.diff_data, lags=1)
            if p < 0.05:
                self.display(u'处理后的序列为非白噪声序列(>0.05)，对应的p值为：%s' %p)
            else:
                self.display(u'处理后的序列该序列为白噪声序列(<0.05)，对应的p值为：%s' %p)
        except:
            return
    def plot_acfandpacf(self):
        try:
            f = plt.figure(figsize=(12, 9))
            ax1 = f.add_subplot(211)
            plot_acf(self.diff_data, lags=40, ax=ax1).show()
            ax2 = f.add_subplot(212)
            plot_pacf(self.diff_data, lags=40, ax=ax2).show()
            f.show()
            self.display("绘制自相关和偏自相关图")
        except:
            QMessageBox.warning(self, "注意", "无法绘制自相关和偏相关图!")
    def arima_chose_pq(self):
        if self.diff == None:
            self.select_range_data()
            self.stationary_detection()
        pq, ok = QInputDialog.getText(self, "是否输入p，q值", "若您通过自相关和偏相关图看出阶数，可直接输入p，q，以英文逗号隔开，时间较短。\n\n"
                                                          "      若要通过本软件自动寻找最优参数（耗费较长时间），可直接按确认键。",
                                         QLineEdit.Normal)
        ####因为日期可能不连续，故取消索引
        self.data_to_arima = self.input_table_select.reset_index()
        self.data_to_arima = self.data_to_arima[self.params_list[0]]
        try:
            if len(pq) == 0 and ok:
                # 一般阶数不超过length/10,这里取p<=8,q<=8
                bic_matrix = []
                for p in range(9):
                    temp = []
                    for q in range(9):
                        try:
                            temp.append(ARIMA(self.data_to_arima, (p, self.diff, q)).fit().bic)
                        except:
                            temp.append(None)
                        bic_matrix.append(temp)
                bic_matrix = pd.DataFrame(bic_matrix)  # 将其转换成Dataframe数据结构
                p, q = bic_matrix.stack().idxmin()  # 先使用stack展平，然后使用idxmin找出最小值的位置
                self.display(u'BIC最小的p值和q值：%s,%s' % (p, q))  # BIC最小的p值和q值
                # 所以可以建立ARIMA模型，ARIMA(p,self.diff,q)
                self.ARIMA_model = ARIMA(self.data_to_arima, (p, self.diff, q)).fit()
                # print(self.ARIMA_model.summary().tables[1])
                return
            elif len(pq) != 0 and ok:
                try:
                    ps = int(pq.split(',')[0])
                    qs = int(pq.split(',')[1])
                    self.ARIMA_model = ARIMA(self.data_to_arima, (ps, self.diff, qs)).fit()
                    self.display(u'BIC值为：%s' %self.ARIMA_model.bic)  # BIC值为这个
                except:
                    traceback.print_exc()
                    return
            else:
                return;
        except:
            QMessageBox.warning("本次操作有误！")
            pass
    def arima_model_diagnosis(self):
        plt.clf()
        ax = self.figure.add_subplot(121)
        resid = self.ARIMA_model.resid.values.reshape(-1,1)
        qqplot(self.ARIMA_model.resid.values, line='q', ax=ax, fit=True)
        ax1 = self.figure.add_subplot(122)
        std = StandardScaler()
        resid_std = std.fit_transform(resid)
        ax1.plot(resid_std,'blue',label='标准化残差')
        ax1.axhline(y=0, linestyle='-', linewidth=1.6, color='red')
        ax1.legend(loc='best',fontsize=10)
        if self.plot_grid:
            ax1.grid(linestyle='--', color='black', linewidth=0.4)
            ax.grid(linestyle='--', color='black', linewidth=0.4)
        plt.tight_layout(pad=0.1, w_pad=0.1)
        plt.subplots_adjust(right=0.93, left=0.07, bottom=0.13)
        self.canvas.draw()
        self.label_8.setText("QQPlot图（左）                            标准化残差图（右）")
        self.display("绘制模型诊断图")
    def arima_model_summary(self):
        try:
            filepath = QFileDialog.getSaveFileName(self, "保存ARIMA模型报告", "arima模型报告", "txt文本(*.txt)")
            with open(filepath[0],'w') as f:
                f.write(str(self.ARIMA_model.summary()))
        except:
            self.display("保存模型过程操作有误！")
            return
    def arima_result_visualization(self):
        steps, ok = QInputDialog.getText(self, "步长输入", "请输入预测的步长："
                                        "（若想看模型拟合效果图，则直接按确认键）\n\n",QLineEdit.Normal)
        try:
            if ok and len(steps) == 0:
                steps = 0
            elif ok and len(steps) != 0:
                steps = int(steps)
            else:
                return
            plt.clf()
            ax = self.figure.add_subplot(111)
            plt.tight_layout(pad=0.1, w_pad=0.1)
            plt.subplots_adjust(right=0.93, left=0.07, bottom=0.13)
            self.ARIMA_model.plot_predict(1, len(self.data_to_arima) + steps, ax=ax)
            if self.plot_grid:
                ax.grid(linestyle='--', color='black', linewidth=0.4)
            self.canvas.draw()
            self.display("使用ARIMA预测%d步值"%steps)
            self.label_8.setText("ARIMA模型%d步预测效果图"%steps)
        except:
            traceback.print_exc()
            return

    ##打开文件
    def openfile(self):
        openfile_name= ''
        openfile_name = QFileDialog.getOpenFileName(self, '选择文件', '', 'Excel files(*.xlsx , *.xls)')
        if len(openfile_name[0])!=0:
            path_openfile_name = openfile_name[0] ###获取路径
            ##绑定文件路径
            self.textEdit.setPlainText(path_openfile_name)
            self.input_table = pd.read_excel(path_openfile_name)
            self.input_table.dropna(axis=1,how='all',inplace=True)
            name = self.input_table.columns[self.input_table.columns.str.contains('日期|时间')]
            if len(name)!=0:
                self.input_table[name[0]]=pd.to_datetime(self.input_table[name[0]])
                self.input_table.set_index(name[0],inplace=True)
            self.write_to_table(self.input_table)
            self.comboCheckBox.addItems(self.input_table.columns.tolist())
            self.label_4.setText("原始数据")
            self.path=openfile_name[0]
            self.display('打开文件')
            self.display_no_time(path_openfile_name)

        else:
            QMessageBox.warning(self, "注意", "未选择文件！")
            return
    #################导出数据的操作====================================
    def main_write_to_file(self):  ##主窗口导出数据
        self.write_to_file(self.tableWidget)
        self.display_no_time('保存主窗口的数据')
        return
    def show_info_write_to_file(self):  ##show_info窗口保存按钮导出数据
        self.write_to_file(self.ShowInfoWindow.tableWidget)
        self.display_no_time('保存统计信息窗口数据')
        return
    def write_to_file(self,tableWidget=None):
        filepath = QFileDialog.getSaveFileName(self, "保存数据", "untitle","Microsoft Excel 2016(*.xlsx)")
        if tableWidget == None:
            tableWidget = self.tableWidget
        output_table_rows = tableWidget.rowCount()
        output_table_columns = tableWidget.columnCount()
        output_file = [[0] * output_table_columns for _ in range(output_table_rows)]
        columns = []
        for index in range(output_table_columns):
            columns.append(tableWidget.horizontalHeaderItem(index).text())
        try:
            for i in range(output_table_rows):
                for j in range(tableWidget.columnCount()):
                    output_file[i][j] = tableWidget.item(i, j).text()
            output_file_pd=pd.DataFrame(output_file, columns=columns)
            output_file_pd.to_excel(filepath[0], index=False)
            self.display(filepath[0])
        except:
            return
    #################写入表格的操作====================================
    def write_to_table(self,data,tableWidget=None):
        ###===========读取表格，转换表格，===========================================
        input_table_rows = data.shape[0]
        input_table_columns = data.shape[1]
        ###======================给tablewidget设置行列表头============================
        if tableWidget==None:
            widgets=self.tableWidget
        else:
            widgets=tableWidget
        widgets.setColumnCount(input_table_columns)
        widgets.setRowCount(input_table_rows)
        input_table_header = data.columns.values.tolist()
        widgets.setHorizontalHeaderLabels(input_table_header)
        data=data.round(4)
        widgets.setVerticalHeaderLabels(data.index.astype(str).tolist())
        ###================遍历表格每个元素，同时添加到tablewidget中========================
        for i in range(input_table_columns):
            input_table_columns_values = data[data.columns[i]]
            # print(input_table_rows_values)
            if (input_table_columns_values.dtype == 'int64'):
                input_table_columns_values_array = np.array(input_table_columns_values, dtype='int64')
            else:
                input_table_columns_values_array = np.array(input_table_columns_values)
            input_table_columns_values_list = input_table_columns_values_array.tolist()
            for j in range(input_table_rows):
                input_table_items_list = input_table_columns_values_list[j]
                ###==============将遍历的元素添加到tablewidget中并显示=======================
                input_table_items = str(input_table_items_list)
                newItem = QTableWidgetItem(input_table_items)
                newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                widgets.setItem(j, i, newItem)
                ###================遍历表格每个元素，同时添加到tablewidget中========================

    ###恢复数据
    def restore_data(self):
        self.input_table = pd.read_excel(self.path)
        self.input_table.dropna(axis=1, how='all', inplace=True)
        name = self.input_table.columns[self.input_table.columns.str.contains('日期|时间')]
        if len(name) != 0:
            self.input_table[name[0]] = pd.to_datetime(self.input_table[name[0]])
            self.input_table.set_index(name[0], inplace=True)
        self.write_to_table(self.input_table)
        self.comboCheckBox.addItems(self.input_table.columns.tolist())
        self.label_4.setText("原始数据")
        self.display('恢复原始数据')
        return
    ###格式调整
    def format_adjust(self):
            self.input_table.dropna(axis=0, how='any', inplace=True)
            for column in self.input_table.columns:
                try:
                    self.input_table[column] = self.input_table[column].str.extract('(\d+\.\d+|\d+)')
                except:
                    continue
            self.write_to_table(self.input_table)
            self.display('清除数据中特殊字符')
            return
    ###删除指标
    def drop_params(self):
        self.check_check_box()
        self.input_table.drop(self.params_list,axis=1,inplace=True)
        self.write_to_table(self.input_table)
        self.comboCheckBox.addItems(self.input_table.columns.tolist())
        self.display('删除'+self.params_list_tostr+'指标')
        return
    ###显示统计信息
    def show_info(self):
        self.check_check_box()
        #弹消息窗版本，只能显示一个指标
        # if len(self.params_list) != 1:
        #     print('请选择一个指标')
        #     QMessageBox.warning(self, "注意", "只能选择一个指标!")
        # else:
        #     # bool_index = self.input_table.columns.str.contains(self.params_list[0])
        #     # index = np.argwhere(bool_index==True)[0]
        #     # index_name = self.input_table.columns[index]
        #     # info = str(self.input_table[index_name].describe())
        #     info = "-------------------------\n"+'统计'+\
        #            str(self.input_table[np.argwhere(self.input_table.columns.str.contains(self.params_list[0])==True)[0]].describe(include='all'))+\
        #            "\n-------------------------"
        #     QMessageBox.about(self, "统计信息", info):
        if self.select_count==0:
            QMessageBox.warning(self, "注意", "请至少选择一个参数！")
            return
        try:
            self.info = self.input_table[self.params_list].describe()
            self.info.insert(0,'统计量',['总数','平均值','标准差','最小值','25%处值','50%处值','75%处值','最大值'])
            self.write_to_table(self.info,tableWidget=self.ShowInfoWindow.tableWidget)
            self.show_info_window.show()
            self.display('显示统计信息')
        except:
            return
    ###看选择了几个指标
    def check_check_box(self):
        #self.params_list = []
        # for each_checkBox in [self.checkBox, self.checkBox_2, self.checkBox_3, self.checkBox_4, self.checkBox_5,
        #                       self.checkBox_6, self.checkBox_7, self.checkBox_8, self.checkBox_9, self.checkBox_10,
        #                       self.checkBox_11, self.checkBox_12, self.checkBox_13, self.checkBox_14,
        #                       self.checkBox_15]:
        #     if each_checkBox.isChecked():
        #         self.params_list.append(each_checkBox.text())
        #         print(each_checkBox.text())
        self.params_list=self.comboCheckBox.Selectlist()
        self.params_list_tostr=''
        for param in self.params_list:
            self.params_list_tostr+=param+' '
        self.params_list_tostr=self.params_list_tostr.strip()
        self.select_count=len(self.params_list)

    ##绘制直方图
    def draw_distplot(self):
        try:
            self.check_check_box()
            if (self.select_count == 0) | (self.select_count > 2):
                QMessageBox.information(self, "注意", "因图幅限制，只能选择1-2个指标！")
                return
            ##初始化plt.axis
            plt.clf()
            if self.select_count == 1:
                self.axis = self.figure.add_subplot(111)
                sns.distplot(self.input_table[self.params_list[0]],color='black',kde=False,)
            if self.select_count == 2:
                self.axis = self.figure.add_subplot(121)
                self.axis2 = self.figure.add_subplot(122)
                sns.distplot(self.input_table[self.params_list[0]],ax=self.axis,color='black',kde=False)
                sns.distplot(self.input_table[self.params_list[1]], ax=self.axis2,color='black',kde=False)
                self.axis2.grid(True, linestyle='--', color='black', linewidth=0.5)
            plt.tight_layout(pad=0.1, w_pad=0.1)
            plt.subplots_adjust(right=0.93, left=0.07, bottom=0.13)
            if self.plot_grid:
                self.axis.grid(linestyle='--', color='black', linewidth=0.5)
            self.axis.set_ylabel('频数')
            self.label_8.setText(self.params_list_tostr + "直方图")
            self.display('绘制' + self.params_list_tostr + '直方图')
            self.canvas.draw()
        except:
            return
    #pd.boxplot()
    def draw_boxplot(self):
        f, ax = plt.subplots(figsize=(12, 9))
        flag=QMessageBox.question(self, "请作出您的选择", "是否先将数据标准化？注意：若所选择指标之间数值差异过大，选是。",
                             QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Ok)
        if flag == QMessageBox.Ok:
            from sklearn.preprocessing import StandardScaler
            std = StandardScaler()
            data_std = std.fit_transform(self.input_table[self.params_list])
            data_box = pd.DataFrame(data_std, columns=self.params_list)
            ax.set_title("选中指标数据标准化后的箱型图", fontsize=15)
            ax.set_ylim([-3, 3])
            self.display('绘制选中指标数据标准化后的箱型图')
        else:
            data_box = self.input_table[self.params_list]
            ax.set_title("选中指标数据的箱型图", fontsize=15)
            self.display('绘制选中指标数据的箱型图')
        ##作图
        data_box[self.info.columns[1:]].boxplot(ax=ax)
        ax.grid(True, linestyle='--', color='black', linewidth=0.4)
        f.show()
        return
    def draw_predictplot(self,data):
        ##初始化plt.axis
        plt.clf()
        self.axis = self.figure.add_subplot(111)
        plt.tight_layout(pad=0.1, w_pad=0.1)
        plt.subplots_adjust(right=0.93, left=0.07, bottom=0.13)
        if self.plot_grid:
            self.axis.grid(linestyle='--', color='black', linewidth=0.4)
        self.axis.plot(self.input_table_select[self.params_list[0]], 'b', label='真实值')
        self.axis.plot(data, 'r', label='预测值')
        self.axis.set_ylabel(self.params_list[0])
        self.axis.legend(loc='best',fontsize=11)
        self.canvas.draw()
        self.label_8.setText(self.params_list_tostr + "预测效果图")
        self.display('绘制' + self.params_list_tostr + '预测效果图')
        return
    def draw_picture(self,data=None):
        if data == None:
            data=self.input_table
        ##初始化plt.axis
        plt.clf()
        self.axis = self.figure.add_subplot(111)
        plt.tight_layout(pad=0.1, w_pad=0.1)
        plt.subplots_adjust(right=0.93, left=0.07, bottom=0.13)
        if self.plot_grid:
            self.axis.grid(linestyle='--', color='black', linewidth=0.4)
        self.check_check_box()
        if self.input_table.index.dtype == 'datetime64[ns]':
            self.axis.set_xlabel("时间")
        if self.select_count == 0:
            print('最少选择1个参数')
            QMessageBox.information(self, "注意", "最少选择1个参数")
            return
        ##超过两个参数的图
        if (self.select_count == 1) | (self.select_count > 2):
            for param in self.params_list:
                self.axis.plot(self.input_table[param],label=param)
                self.axis.legend(loc='best', fontsize=11,frameon=True,edgecolor='black')
            self.canvas.draw()
            self.label_8.setText(self.params_list_tostr + "随时间变化图")
            self.display('绘制'+self.params_list_tostr+'单坐标趋势图')
            return
        if len(self.params_list) == 2:
            self.axis.plot(self.input_table[self.params_list[0]],'r',label=self.params_list[0])
            self.axis.set_ylabel(self.params_list[0])
            self.axis2=self.axis.twinx()
            self.axis2.plot(self.input_table[self.params_list[1]], 'b',label=self.params_list[1])
            self.axis2.set_ylabel(self.params_list[1])
            #self.axis2.legend(loc=1,bbox_to_anchor=(1,0.9),fontsize=11,frameon=True,edgecolor='black')
            handles1, labels1 = self.axis.get_legend_handles_labels()
            handles2, labels2 = self.axis2.get_legend_handles_labels()
            plt.legend(handles1 + handles2, labels1 + labels2, loc='best',fontsize=10,frameon=True,edgecolor='black')
            self.canvas.draw()
            self.label_8.setText(self.params_list_tostr + "随时间变化图")
            self.display('绘制' + self.params_list_tostr + '双坐标坐标趋势图')
            return
    def display(self, information):
        self.textBrowser.append(time.strftime("%H:%M:%S", time.localtime()))
        self.textBrowser.append(information)
        return
    def display_no_time(self,information):
        self.textBrowser.append(information)
        return
    def turn_on_off_grid(self):
        if self.plot_grid == True:
            self.plot_grid = False
            self.action_18.setText('网格线')
            self.display("关闭绘图网格线")
        else:
            self.plot_grid = True
            self.action_18.setText('网格线√')
            self.display("打开绘图网格线")
    def select_range_data(self):
        range, ok = QInputDialog.getText(self, "特定数据段选取", "若索引为时间，可输入开始与结尾的日期，最小精确到天。"
                                                          "格式为(年月日)：XXXX-XX-XX，XXXX-XX-XX。\n\n"
                                                          "          有无索引，均可输入所需数据段开始与结尾的索引数字。格式为：XX,XX。\n\n"
                                                        "                              注意:以英文逗号隔开！！\n\n"
                                                          "                       选择全部数据请直接按确认键", QLineEdit.Normal)
        try:
            if len(range) == 0:
                self.input_table_select = self.input_table
                self.display('选取了所有数据')
                return
            index = range.split(',')
            if (len(index[0]) != 10) | (len(index[0]) != 9):
                index[0] = int(index[0])
                index[1] = int(index[1])
            self.input_table_select=self.input_table[index[0]:index[1]]
            self.write_to_table(self.input_table_select)
            print(index[1])
            self.display("选取%d至%d共%d数据"%(index[0],index[1],index[1]-index[0]))
        except:
            return


    ####串口编程#######
    def refresh(self):
        # 查询可用的串口
        plist = list(serial.tools.list_ports.comports())
        if len(plist) <= 0:
            print("No used com!");
            #self.statusBar.showMessage('没有可用的串口')
        else:
            # 把所有的可用的串口输出到comboBox中去
            self.comboBox.clear()
            for i in range(0, len(plist)):
                plist_0 = list(plist[i])
                self.comboBox.addItem(str(plist_0[0]))
                # 重载窗口关闭事件
    def closeEvent(self, e):
        # 关闭定时器，停止读取接收数据
        self.timer.stop()
        # 关闭串口
        if self.ser != None:
            self.ser.close()
    def clear(self):  #清除窗口操作
        try:
            self.textEdit_2.clear()
            self.receive_num = 0
            dis = '接收:' + '{:d}'.format(self.receive_num)
            #self.statusBar.showMessage(dis)
        except:
            traceback.print_exc()
    def recv(self): # 串口接收数据处理
        try:
            num = self.ser.inWaiting()
        except:
            self.timer.stop()
            # 串口拔出错误，关闭定时器
            self.ser.close()
            self.ser = None
            # 设置为打开按钮状态
            self.pushButton_serial_openoff.setChecked(False)
            self.pushButton_serial_openoff.setText("打开串口")
            print('serial error!')
            return None
        if (num > 0):
            # 有时间会出现少读到一个字符的情况，还得进行读取第二次，所以多读一个
            data = self.ser.read(num)
            # 调试打印输出数据
            # print(data)
            num = len(data)
            # 十六进制显示
            if self.checkBox.checkState():
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '{:02X}'.format(data[i]) + ' '
            else:
                # 串口接收到的字符串为b'123',要转化成unicode字符串才能输出到窗口中去
                out_s = data.decode('iso-8859-1')
                if self.rcv_enter == '\r':
                    # 上次有回车未显示，与本次一起显示
                    out_s = '\r' + out_s
                    self.rcv_enter = ''
                if out_s[-1] == '\r':
                    # 如果末尾有回车，留下与下次可能出现的换行一起显示，解决textEdit控件分开2次输入回车与换行出现2次换行的问题
                    out_s = out_s[0:-1]
                    self.rcv_enter = '\r'
            # 先把光标移到到最后
            cursor = self.textEdit_2.textCursor()
            if (cursor != cursor.End):
                cursor.movePosition(cursor.End)
                self.textEdit_2.setTextCursor(cursor)
            # 把字符串显示到窗口中去
            self.textEdit_2.insertPlainText(out_s)
            # 统计接收字符的数量
            self.receive_num = self.receive_num + num
            dis = '接收:' + '{:d}'.format(self.receive_num)
            #self.statusBar.showMessage(dis)
            # 获取到text光标
            textCursor = self.textEdit_2.textCursor()
            # 滚动到底部
            textCursor.movePosition(textCursor.End)
            # 设置光标到text中去
            self.textEdit_2.setTextCursor(textCursor)
        else:
            # 此时回车后面没有收到换行，就把回车发出去
            if self.rcv_enter == '\r':
                # 先把光标移到到最后
                cursor = self.textEdit_2.textCursor()
                if (cursor != cursor.End):
                    cursor.movePosition(cursor.End)
                    self.textEdit_2.setTextCursor(cursor)
                self.textEdit_2.insertPlainText('\r')
                self.rcv_enter = ''
    def baud_modify(self): #波特率修改
        if self.ser != None:
            self.ser.baudrate = int(self.comboBox_2.currentText())
    def com_modify(self): #串口号修改
        if self.ser != None:
            self.ser.port = self.comboBox.currentText()
    def open_close(self,btn_sta):  ##打开关闭串口
        if btn_sta == True:
            try:
                # 输入参数'COM13',115200
                self.ser = serial.Serial(self.comboBox.currentText(), int(self.comboBox_2.currentText()), timeout=0.2)
            except:
                QMessageBox.critical(self, '提醒', '没有可用的串口或当前串口被占用')
                return None
            # 字符间隔超时时间设置
            self.ser.interCharTimeout = 0.001
            # 1ms的测试周期
            self.timer.start(2)
            self.pushButton_serial_openoff.setText("关闭串口")
            print('open')
        else:
            # 关闭定时器，停止接收数据
            # self.timer_send.stop()
            self.timer.stop()
            try:
                # 关闭串口
                self.ser.close()
            except:
                QMessageBox.critical(self, '提醒', '关闭串口失败')
                return None
            self.ser = None
            self.pushButton_serial_openoff.setText("打开串口")
            print('close!')

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = Mainwindow()
    ui.show()
    sys.exit(app.exec_())