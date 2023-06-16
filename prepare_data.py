"""This script runs the interface."""

from src import MainWindow
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()