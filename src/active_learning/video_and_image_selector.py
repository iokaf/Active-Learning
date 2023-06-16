"""This module implements the functions to select frames from videos.
"""
import json
import os
from typing import Dict, List 
import warnings

import numpy as np
import pandas as pd


class DataInclusion:
    """This class implements the utilities for selecting data to be used in the
    active learning process.
    """

    def __init__(self, config: Dict):
        """Initialize the class.

        Args:
            config: A dictionary containing the configuration.
        """

        self.config = config


    def select_data(self) -> List[Dict]:
        """This method selects the data to be used in the current active learning step.
        
        If use-all-data:
            Read the all images json using the get_all_image_data method.
        Else:
            Select the video filenames to be included.
            Read the images for the particular videos.

        After loading, exclude previously used data.

        Returns:
            A list of dictionaries containing the data to be used in the current active learning step.
            Dictionary represents and image and has the following keys:
                - image_id: The id for the image. Easiest choice is (video_filename, frame_number).
                - frame_path: The path to the image
                - crop: Cropping dimensions for the image or None if no crop is needed
                - fold: The fold for the image to determine if it is used for training or validation.
                - video_filename: The video where the image came from.

        """

        # Load all available data for the step
        use_all_videos = self.config["data-inclusion"]["use-all-videos"]
        if use_all_videos:
            data = self.get_all_image_data()
        else:
            video_filenames = self.select_from_available_videos()
            data = []
            for video_filename in video_filenames:
                video_data = self.get_video_image_data(video_filename)
                data.extend(video_data)
        
        # Read the image ids used in previous active learning steps
        previously_selected_images = self.read_image_ids_used()
        
        previously_selected_images = list(map(tuple, previously_selected_images))

        # Exclude previously used images

        final_data = []
        for data_point in data:
            if data_point["image_id"] in previously_selected_images:
                continue
            final_data.append(data_point)
        
        return final_data


    def get_included_videos_file_path(self) -> str:
        """Returns the path to the included videos file."""

        outputs_dir = self.config["general"]["output-directory"]
        included_videos_path = os.path.join(
            outputs_dir, 'included_videos.txt'
        )
        return included_videos_path

    def create_included_videos_file(self):
        """Creates the included videos file."""

        included_videos_path = self.get_included_videos_file_path()
        with open(included_videos_path, 'w') as f:
            f.write('')

    def reset_included_videos_file(self):
        """Resets the included videos file."""

        included_videos_path = self.get_included_videos_file_path()
        with open(included_videos_path, 'w') as f:
            f.write('')
    
    def read_previously_selected_videos(self):
        """Reads the list of videos that have been previously selected for labeling."""

        included_videos_path = self.get_included_videos_file_path()
        if not os.path.exists(included_videos_path):
            self.create_included_videos_file()

        with open(included_videos_path) as f:
            videos = f.readlines()

        # Strip videos, remove new lines, remove empty lines
        videos = [video.strip() for video in videos if video.strip()]
        videos = [video.replace("\n", "") for video in videos]
        
        # Filter out empty strings
        videos = [video for video in videos if video]

        return videos


    def update_previously_selected_videos(self, videos: List[str]):
        """Updates the list of videos that have been previously selected for labeling.
        
        Args:
            videos: A list of video filenames.    
        """

        included_videos_path = self.get_included_videos_file_path()
        with open(included_videos_path, 'a') as f:
            for video in videos:
                f.write(video + '\n')

    def get_all_video_filenames(self):
        """Gets the filenames for all videos included in the AL."""

        video_names_file = self.config["data-inclusion"]["path-to-video-names-file"]
        with open(video_names_file, "r") as file:
            video_filenames = file.readlines()

        video_filenames = [video_filename.strip() for video_filename in video_filenames]
        video_filenames = [video_filename.replace("\n", "") for video_filename in video_filenames]
        video_filenames = [video_filename for video_filename in video_filenames if video_filename]

        return video_filenames

    def get_available_video_filenames(self):
        """Gets the filenames for all videos that have not been previously selected for labeling."""

        video_filenames = self.get_all_video_filenames()
        selected_videos = self.read_previously_selected_videos()

        available_video_filenames = [video for video in video_filenames if video not in selected_videos]

        return available_video_filenames

    def select_from_available_videos(self):
        """Select video filenames from the available for this step"""
        num_select = self.config["data-inclusion"]["number-of-videos-each-iteration"]
        available_videos = self.get_available_video_filenames()

        if len(available_videos) < num_select:
            # If not enough videos are available, reset the selected videos file
            self.reset_included_videos_file()
            selected_videos = self.select_from_available_videos()
            return selected_videos

        selected_videos = np.random.choice(
            available_videos, num_select, replace=False
        )
        
        self.update_previously_selected_videos(selected_videos)
        
        return selected_videos

    def get_video_image_data(self, video_filename: str) -> List[Dict]:
        """Gets the image data for a video.
        
        Args:
            video_filename (str): The filename of the video.

        Returns:
            (List[Dict]): A list of dictionaries containing the image data for the video.

        Notes:
        Instead of reading the data from the json file, we could read the data
        from the database. 
        """
        video_frames_dir = self.config["data-inclusion"]["video-frames-json-directory"]

        video_file = os.path.join(video_frames_dir, f"{video_filename}.json")

        
        with open(video_file) as f:
            video_data = json.load(f)

        image_data = []

        for data_point in video_data:
            video_name = data_point["video_filename"]
            image_name = data_point["path"].split("/")[-1].split(".")[0]

            image_id = (video_name, image_name)
            image_data.append(
                {
                    "image_id": image_id,
                    "frame_path": data_point["path"],
                    "crop": data_point["crop"],
                    "fold": data_point["fold"],
                    "video_filename": data_point["video_filename"],
                }
            )

        return image_data        

    def get_all_image_data(self) -> List[Dict]:
        """Gets the image data for all videos included in the AL."""

        all_data_file = self.config["data-inclusion"]["all-image-data-json-path"]
        
        data = pd.read_json(all_data_file)
        image_data = []

        for _, row in data.iterrows():
            video_name = row["video_filename"]
            image_name = row["path"].split("/")[-1].split(".")[0]

            image_id = (video_name, image_name)
            image_data.append(
                {
                    "image_id": image_id,
                    "frame_path": row["path"],
                    "crop": row["crop"],
                    "fold": row["fold"],
                    "video_filename": row["video_filename"],
                }
            )

        return image_data

    def get_used_ids_file_path(self) -> bool:
        """Checks if the image ids per label file exists.

        Returns:
            True if the file exists, False otherwise.
        """

        outputs_dir = self.config["general"]["output-directory"]
        return os.path.join(outputs_dir, "image_ids_used.json")

    def create_if_not_exists_image_ids_per_label_file(
            self, labels: List[str]):
        """Creates the image ids per label file.

        Create file that contains the image ids that have been used,
        if it does not exist.

        Args:
            labels (List[str]): A list of labels.
        """
        used_ids_file_path = self.get_used_ids_file_path()

        if not os.path.exists(used_ids_file_path):
            image_ids_per_label = {label: [] for label in labels}

            with open(used_ids_file_path, "w") as file:
                json.dump(image_ids_per_label, file)


    def read_image_ids_per_label(self) -> Dict[str, List]:
        """Reads the image ids per label from the file.

        Returns:
            Dictionary with the image ids per label. The keys are the labels
            and the values are the image ids.
        """
        labels = self.config["labels"]["names"]
        self.create_if_not_exists_image_ids_per_label_file(labels)

        used_ids_file_path = self.get_used_ids_file_path()
        with open(used_ids_file_path, "r") as file:
            image_ids_per_label = json.load(file)

        return image_ids_per_label


    def read_image_ids_used(self) -> List:
        """Reads the image ids used from the file.

        Returns:
            List: A list with all the image ids that have been used so far.
        """
        image_ids_per_label = self.read_image_ids_per_label()

        image_ids_used = [img_id for img_ids in image_ids_per_label.values() for img_id in img_ids]
        return image_ids_used


    def write_image_ids_per_label(self, image_ids_per_label: Dict[str, List[int]]):
        """Writes the image ids per label to the file.

        Args:
            image_ids_per_label (Dict[str, list(int)]): The image ids selected
                per label.
        """
        used_ids_file_path = self.get_used_ids_file_path()
        with open(used_ids_file_path, "w") as file:
            json.dump(image_ids_per_label, file)


    def update_image_ids_used(
        self,
        image_ids_old: Dict[str, List[int]],
        image_ids_new: Dict[str, List[int]],
    ) -> None:
        """Updates the image ids used dictionary.

        This function updates the image ids used dictionary with the image ids
        selected for the current AL step. It does not return a dictionary, but
        it updates the first dictionary passed as argument.

        Args:
            image_ids_old (Dict[str, list(int)]): The image ids selected
                per label over the previous AL steps.
            image_ids_new (Dict[str, list(int)]): The image ids selected
                per label for the current AL step.

        Raises:
            ValueError: If the image ids selected for the current AL step are
                already selected for the previous AL steps.
        """
        labels = self.config["labels"]["names"]
        for label in labels:

            if label not in image_ids_old:
                warnings.warn(f"Label {label} was not used in previous selection steps")
                image_ids_old[label] = []

            image_ids_included = image_ids_old[label]
            new_prediction = image_ids_new[label]

            image_ids, _ = zip(*new_prediction)  # Ignore the probabilities
            image_ids_included.extend(list(image_ids))

            image_ids_old[label] = image_ids_included

            