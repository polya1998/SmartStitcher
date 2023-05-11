from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

class BlinkingImage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置初始状态
        self._blink_on = False
        
        # 创建标签和按钮
        self._label = QLabel(self)
        self._button = QPushButton("Start", self)
        self._button.clicked.connect(self.start_blinking)
        
        # 加载图片
        self._pixmap = QPixmap("path/to/image.png")
        self._label.setPixmap(self._pixmap)
        
        # 创建垂直布局，并将标签和按钮添加到布局中
        layout = QVBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._button)
        self.setLayout(layout)
        
        # 创建定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.toggle_blink)
    
    def start_blinking(self):
        # 开始闪烁
        self._timer.start(500) # 每500毫秒切换一次
        
        # 更新按钮文字
        self._button.setText("Stop")
        self._button.clicked.disconnect()
        self._button.clicked.connect(self.stop_blinking)
    
    def stop_blinking(self):
        # 停止闪烁
        self._timer.stop()
        self._label.setPixmap(self._pixmap)
        
        # 更新按钮文字
        self._button.setText("Start")
        self._button.clicked.disconnect()
        self._button.clicked.connect(self.start_blinking)
    
    def toggle_blink(self):
        # 切换闪烁状态
        if self._blink_on:
            self._label.setPixmap(self._pixmap)
        else:
            self._label.setPixmap(QPixmap())
        self._blink_on = not self._blink_on

if __name__ == "__main__":
    app = QApplication([])
    window = BlinkingImage()
    window.show()
    app.exec_()
