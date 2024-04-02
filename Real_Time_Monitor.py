import sys
import json
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class DynamicMplCanvas(FigureCanvas):
    """一个动态更新的matplotlib画布，包括两个图表"""
    def __init__(self, *args, **kwargs):
        self.fig = Figure(figsize=(5, 8))  # 调整画布尺寸以容纳两个图表
        super(DynamicMplCanvas, self).__init__(self.fig)
        self.axes1 = self.fig.add_subplot(211)  # 第一个图表
        self.axes2 = self.fig.add_subplot(212)  # 第二个图表

    def update_figure(self):
        # 尝试读取并绘制第一个数据集
        try:
            with open('pearl/外汇交易分仓/动态监控.json', 'r') as file:
                data1 = json.load(file)
                if isinstance(data1, list):  # 确保数据是列表格式
                    self.axes1.cla()
                    self.axes1.plot(data1, 'r-')
                    self.axes1.set_ylabel('net value')  # 设置第一个图表的Y轴标签
                else:
                    raise ValueError("Data format is not supported")
        except Exception as e:
            1


        # 尝试读取并绘制第二个数据集
        try:
            with open('pearl/外汇交易分仓/奖励动态监控.json', 'r') as file:
                data2 = json.load(file)
                if isinstance(data2, list):  # 确保数据是列表格式
                    self.axes2.cla()
                    self.axes2.plot(data2, 'b-')
                    self.axes2.set_ylabel('reward')  # 设置第一个图表的Y轴标签
                else:
                    raise ValueError("Data format is not supported")
        except Exception as e:
            1


        self.draw()

class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real_time_monitor")
        self.main_widget = QWidget(self)
        layout = QVBoxLayout(self.main_widget)
        self.canvas = DynamicMplCanvas(self.main_widget, width=5, height=8, dpi=100)
        layout.addWidget(self.canvas)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.canvas.update_figure)
        self.timer.start(1000)

    def keyPressEvent(self, event):
        # 保留原有的键盘控制逻辑
        if event.key() == QtCore.Qt.Key_S:
            with open('提前终止训练.txt', 'w') as file:
                file.write('1')
            print('训练已提前终止，并已记录。')
            event.accept()

        elif event.key() == QtCore.Qt.Key_R:
            with open('提前终止训练.txt', 'w') as file:
                file.write('2')
            print('训练被重置，但不终止。')
            event.accept()

        elif event.key() == QtCore.Qt.Key_Up:
            with open('动态难度.txt', 'r') as file:
                content = file.read()
                difficulty = float(content)
                difficulty += 0.1
                print(f"订单难度上调到{difficulty}")
            with open('动态难度.txt', 'w') as file:
                file.write(str(difficulty))
            event.accept()

        elif event.key() == QtCore.Qt.Key_Down:
            with open('动态难度.txt', 'r') as file:
                content = file.read()
                difficulty = float(content)
                difficulty -= 0.1
                print(f"订单难度下调到{difficulty}")
            with open('动态难度.txt', 'w') as file:
                file.write(str(difficulty))
            event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(app.exec_())
