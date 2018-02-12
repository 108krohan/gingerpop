from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QCheckBox, QGridLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QCoreApplication
import sys
import matplotlib.pyplot as plt
import os
class charWindow(QWidget):
    
    def __init__(self):
        super().__init__()
        
        #self.final_x = final_x
        #self.final_y = final_y
        
        self.initUI()
        
        
    def initUI(self):               
        self.window = QWidget()
        
        self.grid = QGridLayout()
        self.window.setLayout(self.grid)

        all_letters = range(int("0900", base=16), int("097F", base=16))
        row = 0
        column = 0
        self.checks = []
        for letter in all_letters:
            font = QFont()
            font.setPointSize(20)
            self.checks.append(QCheckBox(chr(letter)))
            self.checks[-1].setFont(font)
            self.grid.addWidget(self.checks[-1], row, column)
            column += 1
            if column==10:
                column = 0
                row += 1
        self.done_button = QPushButton("DONE")
        self.grid.addWidget(self.done_button, row, column, row+1, column+1)
        self.done_button.clicked.connect(self.done_save)
        
        self.window.setWindowTitle("Labeling")
        #self.window.setGeometry(300, 300, 250, 150)
        self.window.move(300,300)
        self.window.show()
        print("Labelling Begins")
        
    def done_save(self):
        filename, extension = sys.argv[1].split('.')
        image = plt.imread(filename+"."+extension)
        os.remove(filename+"."+extension)
        for check in self.checks:
            if check.checkState():
                filename = filename + "_" + str(ord(check.text()))
        print("SAVING", filename+"."+extension)
        plt.imsave(arr = image, fname = filename+"."+extension)
        print("DONE")
        QCoreApplication.instance().quit()
        print("DONE2")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = charWindow()
    sys.exit(app.exec_())


#-- 108krohan -- qt_char_select by shubhvachher https://www.linkedin.com/in/shubh-vachher-440b16a2/
