import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox, QLabel
import datetime
import pygsheets
import matplotlib.pyplot as plt

class Window(QtWidgets.QMainWindow):

    items = ['Baendiger', 'White Pearl', 'Blut', 'Tears', 'Fuehrungsregi', 'Dunkle Perlen',
    'Hades', 'Zeus', 'Poseidons', 'Gegenstand Segnen']
    selectedItem = items[0]

    def __init__(self):
        super(Window, self).__init__()

        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Windows"))

        self.date = self.getCurrentDate()
        googleAPI = pygsheets.authorize()
        self.sheet = googleAPI.open('Preise').sheet1

        self.setGeometry(250,250,600,400)
        self.setWindowTitle("Metin2 Preise")
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.init_HUD()
        self.show()

    def init_HUD(self):
        self.defineFonts()
        self.createDropdown()
        self.createButtons()
        self.createTextLabels()
        self.createInputFields()

    def defineFonts(self):
        self.basicFont = QtGui.QFont("Times", 12)
        self.titleFont = QtGui.QFont("Times", 16)

    def createTextLabels(self):
        l_title = QLabel("Takania2-Price-Checker", self)
        l_title.move(30,30)
        l_title.setFont(self.titleFont)
        l_title.resize(l_title.sizeHint())

        l_price = QLabel("Price:", self)
        l_price.move(30,140)
        l_price.setFont(self.basicFont)
        l_price.resize(l_price.sizeHint())
        #label1.setStyleSheet("QLabel {background-color: red;}")

        l_quantity = QLabel("Quantity:", self)
        l_quantity.move(30,170)
        l_quantity.setFont(self.basicFont)
        l_quantity.resize(l_quantity.sizeHint())
        #label1.setStyleSheet("QLabel {background-color: red;}")

    def createButtons(self):
        #Quit Button
        btn_quit = self.create_pushButton("Quit")
        btn_quit.clicked.connect(self.close_app)
        btn_quit.move(525,370)

        #Send Data Button
        btn_send = self.create_pushButton("Send")
        btn_send.move(140,210)
        btn_send.clicked.connect(self.processInput)

        #Plot Button
        btn_plot = self.create_pushButton("Plot")
        btn_plot.move(210, 85)
        btn_plot.clicked.connect(self.plot_variable)

    def create_pushButton(self, name):
        button = QtWidgets.QPushButton(name, self)
        button.setFont(self.basicFont)
        button.resize(button.sizeHint())
        return button

    def createDropdown(self):
        #Item select
        comboBox = QtWidgets.QComboBox(self)
        for item in self.items:
            comboBox.addItem(item)

        comboBox.move(30,90)
        comboBox.activated[str].connect(self.onSwitchedItem)
        comboBox.setFont(self.basicFont)
        comboBox.resize(comboBox.sizeHint())

    def createInputFields(self):
        self.tb_price = QtWidgets.QLineEdit(self)
        self.tb_price.move(140,140)
        self.tb_price.setFont(self.basicFont)
        self.tb_price.resize(80,25)

        self.tb_quantity = QtWidgets.QLineEdit(self)
        self.tb_quantity.move(140,170)
        self.tb_quantity.setFont(self.basicFont)
        self.tb_quantity.resize(80,25)

    def calcPricePerItem(self):
        #get Input and covert
        price = self.tb_price.text()
        quantity = self.tb_quantity.text()
        price = self.convertThousand(price)
        quantity = self.convertThousand(quantity)

        #check if not Int
        if(not self.checkIfInt(price) or not self.checkIfInt(quantity)):
            return False

        price = int(price)
        quantity = int(quantity)

        #convert to int since behind the comma is not very important
        self.pricePerItem = int(price / quantity)
        return True

        #self.convertNumberToString(pricePerItem)


    def processInput(self):
        #during call calcAlready occured
        if(not self.calcPricePerItem()):
            return

        data = self.formatData()
        if(self.confirmInputFromUser(data)):
            self.sendData(data)
            self.clearInput()


    def confirmInputFromUser(self, data):
        choice = QMessageBox.question(self, "Send confirmation",
                   "Item: {} \n Price: {} \n Date: {} ?".format(data[0],data[1],data[2]),
                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if(choice == QMessageBox.Yes):
            return True
        return False

    def clearInput(self):
        self.tb_price.setText("")
        self.tb_quantity.setText("")

    def onSwitchedItem(self, item):
        self.selectedItem = item

    def convertNumberToString(self, number):
        form = str(int(number))
        form = form[::-1]
        form = form.replace("000","k")
        form = form[::-1]
        return form

    def checkIfInt(self, number):
        try:
            int(number)
            return True
        except ValueError:
            QMessageBox.warning(self, "Warning", "Sorry, but " + number + " is not a valid input")
            return False

    def convertThousand(self, string):
        return string.replace("k","000")

    def close_app(self):
        print("Quit App")
        sys.exit()

    def getCurrentDate(self):
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        return date

    def formatData(self):
        dataArr = []
        dataArr.append(self.selectedItem)
        dataArr.append(self.pricePerItem)
        dataArr.append(self.date)
        return dataArr


    def sendData(self, data):
        self.sheet.insert_rows(row=2, number=1, values=data)
        print("Send Data: ", data)

    def plot_variable(self):
        #get selected var
        row_ids = self.get_rows_for_selected_item()
        #get data from spreadsheet
        prices, dates = self.get_values_for_ids(row_ids)

        #get average:
        average = int(sum(prices)/len(prices))
        round_int = len(str(average)) -1
        print(round_int)
        print(average)
        print("rounded: {:,}".format(round(average, -round_int)))

        norm_prices, y_label = self.normalize_prices(prices)
        #self.plot_simple_graph(prices)
        print(norm_prices)
        #TODO: here is a problem!
        #self.plot_boxplot(norm_prices)

        #plot data

    def get_rows_for_selected_item(self):
        all_names = self.sheet.get_col(1, "cell", False)
        row_ids = []
        for cell in all_names:
            if cell.value == self.selectedItem:
                #append id to array, add 1 since it starts at 0
                row_ids.append(cell.row + 1)
        return row_ids

    def get_values_for_ids(self, row_ids):
        prices = []
        dates = []
        for row in row_ids:
            row_info = self.sheet.get_row(row, "matrix", False)
            price = int(row_info[1])
            date = row_info[2]
            prices.append(price)
            dates.append(date)
        return prices, dates

    def plot_simple_graph(self, prices):
        plt.plot(prices)
        plt.show()

    def normalize_prices(self, prices):
        y_label = ""

        average = int(sum(prices)/len(prices))
        length_int = len(str(average))
        number_thousands = length_int // 3

        for _ in range(number_thousands):
            y_label += "k"

        # print(average)
        # print("y_label: ", y_label)
        # print("rounded thingy: ", round(average, -number_thousands*3))
        # print("final number = ", round(average/1000**number_thousands))

        norm_prices = [round(x/1000**number_thousands) for x in prices]
        return norm_prices, y_label

    def plot_boxplot(self, prices):
        fig, ax = plt.subplots()
        labels = [self.selectedItem]
        box = plt.boxplot(prices, labels=labels)
        #doesn't improve plot in anyway
        #plt.ylim(1000000000,10000000000)
        plt.show()

app = QtWidgets.QApplication(sys.argv)

GUI = Window()

sys.exit(app.exec_())
