"""This module is responsible for creating the json file with the data to be used in the Active Learning process."""

import json

from pathlib import Path
from typing import List, Dict, Union

import numpy as np

class DataProcessor:
    """This class is responsible for creating the json file with the data to be used in the Active Learning process.
    
    Attributes:
        data_dir (str): The directory where the data is located.
        file_types (List[str]): The file types to be processed.
        output_dir (str): The directory where the output file will be saved.
        output_file (str): The name of the output file.
    """
    
    def __init__(self, 
            data_dir: Union[str, List[str]], 
            file_types: List[str] = ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"],
            output_dir: str = "./", 
            output_file: str = "AL_data.json"
        ):
        """Constructor method.
        
        Args:
            data_dir (str): The directory where the data is located.
            file_types (List[str]): The file types to be processed.
            output_dir (str): The directory where the output file will be saved.
            output_file (str): The name of the output file.
            
        """

        if isinstance(data_dir, str):
            data_dir = [data_dir]
        
        assert len(data_dir) > 0, "data_dir must not be empty."
        assert all(isinstance(path, str) for path in data_dir), "data_dir must be a list of strings."
        assert isinstance(file_types, list), "file_types must be a list."
        assert len(file_types) > 0, "file_types must not be empty."
        assert all(isinstance(file_type, str) for file_type in file_types), "file_types must be a list of strings."
        assert isinstance(output_dir, str), "output_dir must be a string."
        assert isinstance(output_file, str), "output_file must be a string."
        assert self.__data_dir_exists(data_dir), "At least one data directory does not exist."

        self.data_dir = data_dir
        self.file_types = file_types

        output_file = output_file if output_file.endswith(".json") else f"{output_file}.json"
        self.output_path = Path(output_dir) / output_file


    def get_examination_folder_paths(self) -> List[str]:
        """Gets the folders of the examinations.
        
        The returned paths are sorted.
        Returns:
            List[str]: The paths to folders of the examinations.
        """
        
        exam_paths = []
        for exam_data_dir in self.data_dir:
            current_exam_folders = [str(folder) for folder in Path(exam_data_dir).iterdir() if folder.is_dir()]
            exam_paths.extend(current_exam_folders)
        exam_paths.sort()
        return exam_paths


    def asign_fold_numbers(self, examination_paths: List[str]) ->  Dict[str, int]:
        """Assigns fold numbers to the examinations.
        
        Adds a random number to each examination path. This number is used to determine 
        if the examination is used for training or validation. The numbers are between 0 and 99.

        Args:
            examination_paths (List[str]): The paths to the folders of the examinations.
            
        Returns:
            Dict[str, int]: The paths to the folders of the examinations as keys with the #
            fold number as values.
        """
        
        fold_numbers = np.random.randint(0, 100, len(examination_paths))
        fold_dict = dict(zip(examination_paths, fold_numbers))
        return fold_dict 
        

    def get_examination_file_paths(self, examination_path: str) -> List[str]:
        """Gets the paths of the image files of the examination.
        
        Args:
            examination_path (str): The path to the folder of the examination.

        Returns:
            List[str]: The paths to the image files of the examination. The paths
            are sorted.
        """

        file_paths = []
        for file_type in self.file_types:
            file_paths.extend([str(file) for file in Path(examination_path).glob(f"*.{file_type}")])
        file_paths.sort()

        return file_paths
    

    def collect_examination_data(self, examination_path: str, fold_number: int) -> Dict[str, str]:
        """Collects the data of the examination.
        
        Args:
            examination_path (str): The path to the folder of the examination.
            fold_number (int): The number of the fold.

        Returns:
            Dict[str, str]: The data of the examination.
        """

        examination_data = {}
        examination_data["examination_path"] = examination_path
        examination_data["fold_number"] = fold_number
        examination_data["file_paths"] = self.get_examination_file_paths(examination_path)
        return examination_data
    

    def create_json_file_data(self) -> List[Dict]:
        """Creates the json file with the data of the examinations.
        
        Args:
            examination_data (List[Dict[str, str]]): The data of the examinations.
        """

        examination_paths = self.get_examination_folder_paths() 
        examination_folds = self.asign_fold_numbers(examination_paths)

        all_examination_data = []

        for exam_dir_path, fold in examination_folds.items():

            examination_name = exam_dir_path.split("/")[-1]

            file_paths = self.get_examination_file_paths(exam_dir_path)
            
            exam_file_data = [{
                    "path": file_path,
                    "crop": None,
                    "fold": int(fold),
                    "video_filename": examination_name,
                } 
                for file_path in file_paths
            ]

            all_examination_data.extend(exam_file_data)

        for n, exam_data in enumerate(all_examination_data):
            exam_data["id"] = n

        return all_examination_data


    def save_json_file(self, json_data: List[Dict]) -> None:
        """Saves the json file with the data of the examinations.
        
        Args:
            json_data (List[Dict]): The data of the examinations.
        """

        with open(self.output_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)


    def process_data(self):
        """Processes the data of the examinations and saves it as a json file.
        """

        json_data = self.create_json_file_data()
        self.save_json_file(json_data)

    def __data_dir_exists(self, data_dir: str) -> bool:
        """Checks if the data directory exists.
        
        Returns:
            bool: True if the data directory exists, False otherwise.
        """
        
        return all(list(map(lambda x: Path(x).exists(), data_dir)))