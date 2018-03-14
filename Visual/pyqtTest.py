import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox, QLabel
#from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow

class Window(QtWidgets.QMainWindow):

	def __init__(self):
		super(Window, self) .__init__()
		self.setGeometry(50,50,500,300)
		self.setWindowTitle("KRASSES DING")
		#self.setWindowIcon(QtGui.QIcon(''))
		self.home()

	def home(self):
		btn = QtWidgets.QPushButton("Quit", self)
		btn.clicked.connect(self.close_app)
		btn.resize(btn.sizeHint())
		btn.move(100,100)

		btn2 = QtWidgets.QPushButton("Read", self)
		btn2.resize(btn.sizeHint())
		btn2.move(200,200)
		btn2.clicked.connect(self.readText)

		label1 = QLabel("Price", self)
		label1.move(50,100)
		#label1.setStyleSheet("QLabel {background-color: red;}")
		label1.resize(label1.sizeHint())

		self.textbox = QtWidgets.QLineEdit(self)
		self.textbox.move(20,20)
		self.textbox.resize(280,40)

		self.show()

	@QtCore.pyqtSlot()
	def readText(self):
		text = self.textbox.text()
		print(text)
		QMessageBox.question(self, 'Title', "You typed: " + text, QMessageBox.Ok)
		print("heheh")

	def close_app(self):
		print("hey")
		sys.exit()

app = QtWidgets.QApplication(sys.argv)

GUI = Window()

sys.exit(app.exec_())
