import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMessageBox, QLabel


class Window(QtWidgets.QMainWindow):
    # TODO: initialize somewhere else
    algorithms = ["Epsilon-Greedy", "Optimistic initial value"]
    environments = ["Bandits"]

    def __init__(self):
        super(Window, self).__init__()

        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

        # self.date = self.getCurrentDate()

        self.setGeometry(250, 250, 400, 400)
        self.setWindowTitle("Reinforcement Learning")
        # self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.init_hud()
        self.show()

    def init_hud(self):
        self.define_fonts()
        self.create_main_screen()

    def define_fonts(self):
        self.basic_font = QtGui.QFont("Times", 12)
        self.title_font = QtGui.QFont("Times", 16)

    def create_main_screen(self):
        self.create_dropdowns()
        self.create_buttons()
        self.create_labels()
        # self.createInputFields()

    def create_dropdowns(self):
        # Environment Dropdown
        env_dropdown = QtWidgets.QComboBox(self)
        for env in self.environments:
            env_dropdown.addItem(env)

        env_dropdown.move(30, 70)
        env_dropdown.setFont(self.basic_font)
        env_dropdown.resize(env_dropdown.sizeHint())
        self.env_dropdown = env_dropdown

        # Algorithm Dropdown
        algo_dropdown = QtWidgets.QComboBox(self)
        for algo in self.algorithms:
            algo_dropdown.addItem(algo)

        algo_dropdown.move(30, 150)
        algo_dropdown.setFont(self.basic_font)
        algo_dropdown.resize(algo_dropdown.sizeHint())
        self.algo_dropdown = algo_dropdown


    def create_labels(self):
        l_title = QLabel("Reinforcement Learning", self)
        l_title.move(85, 10)
        l_title.setFont(self.title_font)
        l_title.resize(l_title.sizeHint())

        l_env_dropdown = QLabel("Select an environment:", self)
        l_env_dropdown.move(30, 50)
        l_env_dropdown.setFont(self.basic_font)
        l_env_dropdown.resize(l_env_dropdown.sizeHint())
        #label1.setStyleSheet("QLabel {background-color: red;}")

        l_algo_dropdown = QLabel("Select an algorithm:", self)
        l_algo_dropdown.move(30, 130)
        l_algo_dropdown.setFont(self.basic_font)
        l_algo_dropdown.resize(l_algo_dropdown.sizeHint())
        #label1.setStyleSheet("QLabel {background-color: red;}")

    def create_buttons(self):
        btn_settings_env = self.basic_push_button("Settings")
        btn_settings_env.move(250, 70)
        btn_settings_env.clicked.connect(self.open_settings_env)

        btn_settings_algo = self.basic_push_button("Settings")
        btn_settings_algo.move(250, 150)
        btn_settings_algo.clicked.connect(self.open_settings_algo)

        btn_add_algo = self.basic_push_button("+")
        btn_add_algo.resize(40, 30)
        btn_add_algo.move(30, 180)
        btn_add_algo.clicked.connect(self.add_algorithm)

        btn_run = self.basic_push_button("Run")
        btn_run.move(160, 300)
        btn_run.clicked.connect(self.start_simulation)

        btn_quit = self.basic_push_button("Quit")
        btn_quit.clicked.connect(self.close_app)
        btn_quit.move(318, 370)

    def basic_push_button(self, name):
        button = QtWidgets.QPushButton(name, self)
        button.setFont(self.basic_font)
        button.resize(button.sizeHint())
        return button

    def createInputFields(self):
        self.tb_price = QtWidgets.QLineEdit(self)
        self.tb_price.move(140,140)
        self.tb_price.setFont(self.basic_font)
        self.tb_price.resize(80,25)

        self.tb_quantity = QtWidgets.QLineEdit(self)
        self.tb_quantity.move(140,170)
        self.tb_quantity.setFont(self.basic_font)
        self.tb_quantity.resize(80,25)

    def add_algorithm(self):
        # TODO
        print("Add")

    def open_settings_env(self):
        # TODO
        print("Settings Environments")

    def open_settings_algo(self):
        # TODO
        print("Settings Algorithms")

    def start_simulation(self):
        # TODO
        print("Start Simulation")

    def onSwitchedItem(self, item):
        self.selectedItem = item

    def clearInput(self):
        self.tb_price.setText("")
        self.tb_quantity.setText("")

    def close_app(self):
        print("Quit App")
        sys.exit()


app = QtWidgets.QApplication(sys.argv)
GUI = Window()
sys.exit(app.exec_())


