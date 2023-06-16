"""This module contains the interface for setting up the Active Learning data.

This is done via a PyQt5 GUI where the user selects the data directory and the output file.
"""

import json
import os

from pathlib import Path
from typing import List, Dict

import numpy as np

from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QWidget,
    QMessageBox
)

from src.data_processor import DataProcessor

class MainWindow(QMainWindow):
    """The main window of the GUI."""


    def __init__(self):
        """Constructor method."""

        super().__init__()

        self.data_dirs = []


        self.setWindowTitle("Active Learning Data Setup")
        self.setMinimumSize(500, 500)

        self.main_widget = QWidget()
        # Add a grid layout to the main widget
        # This will be used to add the widgets to the main window
        # The grid layout will be divided into 3 rows and 2 columns
        # 
        # The first row will be used for the data directory selection
        # The second row will be used for the output file selection
        # The third row will be used for the buttons
                
        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        

        # Add a text field to select the data directory
        self.data_dir_label = QLabel("Data Directories:")
        self.data_dir_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.main_layout.addWidget(self.data_dir_label, 0, 0)

        # Add a button that allows the user to navigate the file system and select the data directory
        self.data_dir_button = QPushButton("Select Data Directory")
        self.data_dir_button.clicked.connect(self.select_data_dir)
        self.main_layout.addWidget(self.data_dir_button, 0, 1)

        # Asign the layout to the main window
        self.setCentralWidget(self.main_widget)

    
    def select_data_dir(self):
        """Selects the data directory."""

        # Open a file dialog to select the data directory
        data_dir = QFileDialog.getExistingDirectory(self, "Select Data Directory", os.getcwd())
        if data_dir:
            # If the user selected a directory, set the data directory
            self.data_dirs.append(data_dir)
            self.data_dirs = list(set(self.data_dirs))
            selected_folder_names = "\n".join(self.data_dirs) 
            self.data_dir_label.setText(f'Data Directory: \n {selected_folder_names}')


            # Create a popup window where the user can click yes if they want to select another file and no if they do not
            msg = QMessageBox()
            msg.setWindowTitle("Data Directory")
            msg.setText("Would you like to select another data directory?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg = msg.exec()

            if msg == QMessageBox.StandardButton.Yes:
                # If the user wants to select another data directory, call the select_data_dir method again
                self.select_data_dir()

            if len(self.data_dirs) > 0:
                self.output_file_label = QLabel("Output File:")
                self.output_file_label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
                self.main_layout.addWidget(self.output_file_label, 1, 0)

                # Add a button that allows the user to navigate the file system and select the output file
                self.output_file_button = QPushButton("Select Output File")
                self.output_file_button.clicked.connect(self.select_output_file)
                self.main_layout.addWidget(self.output_file_button, 1, 1) 

    def select_output_file(self):
        """Selects the output file."""

        # Open a file dialog to select the output file
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", os.getcwd())
        if output_dir:
            # If the user selected a directory, ask them to input the file name
            self.output_dir = output_dir

            # Add a prompt for the user to type some text
            self.output_file_prompt = QLabel("Enter the output file name:")
            # self.output_file_prompt.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.main_layout.addWidget(self.output_file_prompt, 2, 0)

            # Add a text field to enter the output file name
            self.output_file_name = QLineEdit("active_learning_data")


            # self.output_file_name.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            self.main_layout.addWidget(self.output_file_name, 2, 1)



            # Add a button to start the data processing
            self.process_button = QPushButton("Process Data")
            self.process_button.clicked.connect(self.process_data)
            self.main_layout.addWidget(self.process_button, 3, 0)

    def process_data(self):
        """Processes the data."""

        print("Creating the data json")


        # # Create a data processor object
        data_processor = DataProcessor(
            data_dir = self.data_dirs, 
            file_types = ["png", "jpg"],
            output_dir = self.output_dir,
            output_file = self.output_file_name.text()
        )    
        
        # # Process the data
        data_processor.process_data()

        # Show a message box to inform the user that the data has been processed
        msg = QMessageBox()
        msg.setWindowTitle("Data Processing")
        msg.setText("Data processing complete!")
        msg.exec()
