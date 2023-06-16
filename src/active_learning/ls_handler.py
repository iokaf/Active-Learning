"""This module contains the functions to handle the communication 
with Label Studio.

This communication is done via the Label Studio API.
The functions include: 
    - Creating a new project in Label Studio
    - Creating tasks for Label Studio
    - Posting tasks to Label Studio
    - Getting the results from Label Studio
    - Deleting tasks from Label Studio
    - Transforming the results from Label Studio to a format that can be used
        for active learning.

Parameters to get from the configuration file:
IMAGES_TO_SERVE_DIR: The path to the folder where the images to be annotated are stored.
LOCAL_WEBSERVER_URL: The url of the local webserver where the images are served.
LABELSET: The list of labels that are to be used for annotation.
PROJECT: The name of the project that is to be used for annotation.
LS_LABEL_PROJECT: A dictionary with PROJECT as key and the LS project id for the project as value.
LS_URL: The url of the Label Studio API.
LS_HEADERS: The headers for the requests to the Label Studio API.
LS_RESULTS_DIR: The path to the folder where the results from Label Studio are stored.
LABELSET_ONEHOT: A dictionary with the labels as keys and the one-hot encoding 
    of the labels as values.
MUTUALLY_EXCLUSIVE: A boolean indicating whether the labels are mutually exclusive.
"""

from datetime import datetime
from glob import glob
import json
import os
import requests as rq
from typing import Dict, List, Tuple
import warnings

import cv2
import numpy as np


class LabelStudioHandler:
    """This class is responsible for handling the communication with Label Studio.
    
    It's responsibilities include:
        - Exporting results from a Label Studio project.
        - Deleting the tasks from a Label Studio project.
        - Posting tasks to a Label Studio project.
    """
    def __init__(self, config: Dict):
        """Initialize the LabelStudioHandler.

        Args:
            config (Dict): The configuration dictionary.    
        """

        ip_address = config["general"]["address"]        
        self.ip_address = self.validate_ip_address(ip_address)

        port = config["label-studio"]["port"]
        self.ls_port = str(port)

        project_id = config["label-studio"]["project-id"]
        self.project_id = str(project_id)

        output_dir = config["general"]["output-directory"]
        ls_results_directory_name = config["label-studio"]["results-directory-name"]

        self.output_dir = os.path.join(output_dir, ls_results_directory_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.token = config["label-studio"]["token"]

    def get_ls_project_url(self) -> str:
        """Get the url for the Label Studio project.

        Returns:
            (str) The url for the Label Studio project.
        """
        base_url = f"{self.ip_address}:{self.ls_port}"

        project_url = os.path.join(
            base_url, "projects", str(self.project_id)
        )

        return project_url

    def create_ls_api_url(self, task: str) -> str:
        """Create the url communicating with the Label-Studio API.

        Args:
        task (str): The task for which the annotations are to be retrieved.

        Returns:
            (str) The url for the label studio export.
        """
        base_url = f"{self.ip_address}:{self.ls_port}"

        export_url = os.path.join(
            base_url, "api", "projects", str(self.project_id), task
        )
        
        return export_url

    def make_headers(self) -> Dict:
        """Create the headers for the requests to the Label Studio API.
        
        Returns:
            A dictionary with the headers for the requests to the Label Studio API.
        """

        headers = {
            "Authorization": f"Token {self.token}",
        }

        return headers

    def get_current_annotations(self) -> List[Dict]:
        """Get the current annotations for a project.

        Returns:
            (List[Dict]): A list of dictionaries, each of which contains the data for one annotation.
        """

        export_url = self.create_ls_api_url("export")

        headers = self.make_headers()
        response = rq.get(export_url, headers=headers)
        response_json = response.json()

        # Save the annotations as a json with the current datetime as filename
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

        filename = f"{dt_string}.json"

        output_file = os.path.join(self.output_dir, filename)

        # Create the directory if it does not exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as file:
            json.dump(response_json, file)

        return response_json
    
    def delete_all_tasks(self) -> rq.Response:
        """Delete all the tasks from a label studio project.

        Returns:
            (requests.models.response) The response from the delete request.
        """
        delete_url = self.create_ls_api_url("tasks")
        headers = self.make_headers()
        response = rq.delete(delete_url, headers=headers)

        return response

    def validate_ip_address(self, ip_address: str) -> str:
        """Validate the ip address.

        Args:
            ip_address (str): The ip address to be validated.

        Returns:
            The validated ip address.
        """

        if not ip_address.startswith("http"):
            ip_address = "http://" + ip_address

        if ip_address.endswith("/"):
            ip_address = ip_address[:-1]
        
        return ip_address

    def post_ls_tasks(self, annotation_tasks: List[Dict]) -> rq.Response:
        """Post the selected frames to label studio.

        Args:
            annotation_tasks (List[Dict]): The list of annotation tasks to be posted.

        Returns:
            A list of frame indices whose images where not found
        """


        ls_task_url = self.create_ls_api_url("import")
        headers = self.make_headers()
        request = rq.post(ls_task_url, json=annotation_tasks, headers=headers)

        return request


class AnnotationsHandler:
    """This class is responsible for handling the annotations.
    
    It's responsibilities include:
        - Collecting all label studio exports
        - Converting Label-Studio exports to a usable format.
        - Creating Label-Studio tasks for the selected images.

    The convert_ls_export_to_annotations method is to be overwritten 
    based on the problem and the data format to be used. 
    Here, we consider image classification problem.
    """
    def __init__(self, config: Dict, ls_handler: LabelStudioHandler):
        """Initialize the AnnotationsHandler.
        
        Args:
            config (Dict): The configuration dictionary.
            ls_handler (LabelStudioHandler): The LabelStudioHandler object to be
                used for communicating with Label Studio.

        """
        labels = config["labels"]["names"]
        # Convert labels to upper case so that checking is not case sensitive
        self.labels = [label.upper() for label in labels]

        self.mutually_exclusive = config["labels"]["mutually-exclusive"]

        self.labels_onehot = self.create_onehot_labels()

        self.ls_output_dir = self.create_ls_output_directory(config)

        self.image_server_directory = config["image-server"]["directory"]

        self.ls_handler = ls_handler

    def create_onehot_labels(self) -> Dict:
        """Create the onehot labels for the labels.

        Returns:
            (Dict): A dictionary with the onehot labels for each label.
        """

        labels_onehot = {}

        for i, label in enumerate(self.labels):
            onehot = [0] * len(self.labels)
            onehot[i] = 1

            labels_onehot[label] = onehot

        return labels_onehot

    def create_webserver_url(self, config: Dict) -> str:
        """Create the url for the webserver.

        Args:
            config (Dict): The configuration dictionary.

        Returns:
            (str): The url for the webserver.
        """

        ip_address = config["general"]["address"]
        port = config["image-server"]["port"]

        if not ip_address.startswith("http"):
            ip_address = "http://" + ip_address

        if ip_address.endswith("/"):
            ip_address = ip_address[:-1]

        webserver_url = f"{ip_address}:{port}"

        return webserver_url

    def convert_ls_export_to_annotations(self, ls_export: List[Dict]) -> List[Dict]:
        """Convert Label-Studio export to usable format.

        Takes the export from Label-Studio and converts it to the data format that
        will be used in the dataset for training the model. 
        
        This method is to be overwritten by the user based on the model and training
        specifications that they have.
        
        Args:
            ls_export (List[Dict]): The json exported from label studio.

        Returns:
            (List[Dict]): A list of dictionaries, each of which contains the data for one annotation in a format
            that is compatible with the rest of the code.

        Note:

        Annotation skip conditions:
            - Annotations that are empty are skipped. This means that even in the negative class
            has to have a label. We can not just skip the annotation.
            
            - Annotations that have the "was_cancelled" flag set to True are skipped.

            - Annotations whose label does not belong to the labelset are skipped.

            - If we have mutually exclusive labels, we skip the annotation if not
              exactly one label is selected.
        """

        annotations = []
        for ls_annot in ls_export:
            # Skip annotations that are empty. This means that even in the negative class
            # has to have a label. We can not just skip the annotation.
            if len(ls_annot.get("annotations")) == 0:
                continue

            if ls_annot.get("annotations")[0].get("was_cancelled"):
                continue
            
            
            new_image_path = ls_annot["meta"]["new_image_path"]
            annotation_result = ls_annot.get("annotations")[0].get("result")

            # If the annotation is empty, skip it. We don't allow empty annotations.
            if len(annotation_result) == 0:
                continue
            annotation_result = annotation_result[0].get("value").get("choices")

            # Go through the choices and add them to the annotation
            # This will createa vector with ones where the labels where assigned and zeros
            # where they were not.
            annotation = np.zeros(len(self.labels))


            for choice in annotation_result:
                choice = choice.upper()
                # This allows using annotations that contained more labels than the current         
                if choice not in self.labels:
                    continue

                choice_ohe = self.labels_onehot[choice]
                choice_ohe = np.array(choice_ohe)
                annotation = annotation + choice_ohe

            # In the mutually exclusive case, do not add the annotation if it is empty
            if self.mutually_exclusive and np.sum(np.abs(annotation)) != 1:
                continue

            annotation_result = annotation.tolist()

            image_id = ls_annot["meta"]["image_id"]
            labelset = ls_annot["meta"]["labelset"]
            video_filename = ls_annot["meta"]["video_filename"]
            fold = ls_annot["meta"]["fold"]

            annotations.append(
                {
                    "new_image_path": new_image_path,
                    "image_id": image_id,
                    "labelset": labelset,
                    "annotation": annotation_result,
                    "video_filename": video_filename,
                    "fold": fold,
                }
            )
        return annotations

    def create_ls_output_directory(self, config: Dict) -> str:
        """Get the label studio output directory.

        Args:
            config (Dict): The configuration dictionary.

        Returns:
            The label studio output directory.
        """
        output_dir = config["general"]["output-directory"]
        ls_results_directory_name = config["label-studio"]["results-directory-name"]

        output_dir = os.path.join(output_dir, ls_results_directory_name)
        return output_dir

    def collect_all_ls_annotations(self) -> List[Dict]:
        """Read all the annotations from the label studio export directory.

        Returns:
            A list of dictionaries, each of which contains the data for one annotation.
        """
        # Get all the files in the label studio results directory

        label_studio_results_files = glob(f"{self.ls_output_dir}/*.json")

        all_annotations = []
        all_annotation_ids = []

        for file in label_studio_results_files:
            with open(file, "r") as json_file:
                ls_annots = json.load(json_file)


            converted_annotations = self.convert_ls_export_to_annotations(ls_annots)
            
            
            for conv_annot in converted_annotations:
                image_id = conv_annot["image_id"]
                # This is a failsafe to avoid duplicate annotations
                if image_id in all_annotation_ids:
                    continue

                all_annotation_ids.append(image_id)
                all_annotations.append(conv_annot)

        return all_annotations
    
    def get_labels_each_image(
        self,
        selected_images: Dict[str, List[Tuple[int, float]]]
    ) -> Dict[int, List[Tuple[str, float]]]:
        """Convert frame selection dictionary to a dictionary with frame numbers as keys
            and the list of tuples with (label, prob) for which the frame has been selected as values.

            This might seem redundant as one image can only be selected for one label, but
            this is done to be able to use the same function but one image can be selected
            for multiple labels.

        Args:
            selected_images (Dict[str, List[Tuple[int, float]]]): The dictionary
                with labels as keys and the list of frame numbers selected for 
                this label as values.

        Returns:
            A dictionary with frame numbers as keys and the list of labels for
            which the frame has been selected as values.
        """

        # The values of the dictionary are lists of tuples where the first element is the frame id
        # and the second element is the model probability.
        # This is converted to a dictionary with frame numbers as keys and the
        # list of (labels, probability) for which the frame has been selected as values.

        image_ids_and_preds = [
            id_and_pred for id_list in selected_images.values() for id_and_pred in id_list
        ]

        labels_each_image = {
            image_id: [
                (label, prediction)
                for label in selected_images
                if (image_id, prediction) in selected_images[label]
            ]
            for image_id, prediction in image_ids_and_preds
        }

        return labels_each_image

    def prepare_data_ls_tasks(
        self,
        selected_images: Dict[int, List], all_image_data: List[Dict],
        config: Dict) -> Tuple[List[Dict], List[int]]:
        """Prepare selected files to be posted for annotation.

        This method does two things:
        First it copies the selected frame to the path for the webserver
        Second it creates a list with the data that are to be converted in
        to annotation tasks.

        Args:
            selected_images (Dict[int, List]): The dictionary with image ids
                as keys and the list of tuples with (label, prob) for which the
                frame has been selected as values.

            all_image_data (List[Dict]): The list of dictionaries with the data for all frames.

        Returns:
            A tuple with a list of dictionaries with the data for the annotation tasks
            and a list of the image ids that could not be found.

        """

        image_ids = selected_images.keys()

        missing_frames = []
        annotation_tasks = []

        webserver_url = self.create_webserver_url(config)

        for image_id in image_ids:
            # Find the image data for the current image id
            # FixMe: This could be faster if we used a dictionary instead of a list for all_image_data
            image_dict = list(
                filter(lambda x: x.get("image_id") == image_id, all_image_data)
            )

            # If the image is not found in the all_image_data list
            if len(image_dict) == 0:
                missing_frames.append(image_id)
                continue

            image_dict = image_dict[0]

            image_path = image_dict["frame_path"]
            image_path = image_path.replace("BackUp2", "BackUp3")
            crop = image_dict["crop"]
            video_filename = image_dict["video_filename"]
            fold = image_dict["fold"]

            new_image_name = f"{'___'.join(image_id)}.jpg"
            
            new_image_path = os.path.join(
                self.image_server_directory, new_image_name
            )

            # Change the path to the image with the webserver url
            image_url = new_image_path.replace(
                self.image_server_directory, webserver_url
            )

            # Load load the image
            img = cv2.imread(image_path)

            # Check if the image needs to be cropped
            if crop is not None and not np.isnan(crop):
                img = img[crop[0] : crop[1], crop[2] : crop[3]]

            # Save the image to the IMAGES_TO_SERVE_DIR
            cv2.imwrite(new_image_path, img)

            task = {
                "data": {
                    "image": image_url,
                },
                "meta": {
                    "image_id": image_id,
                    "video_filename": video_filename,
                    "fold": fold,
                    "new_image_path": new_image_path,
                    "original_image_path": str(image_path),
                    "labels": [pred[0] for pred in selected_images[image_id]],
                    "labelset": self.labels
                },
            }

            annotation_tasks.append(task)

        return annotation_tasks, missing_frames

    def post_selected_images_to_ls(
        self, 
        selected_images: Dict[int, List], all_image_data: List[Dict],
        config: Dict) -> rq.Response:
        """Post the selected frames to label studio.

        Args:
            selected_images (List[Dict]): The dictionary with image ids as keys 
            and the list of tuples with (label, prob) for which the frame has 
            been selected as values.
            
            all_image_data (List[Dict]): The list of dictionaries with the data
            config (Dict): The config dictionary.
        Returns:
            The post request response
        """
        tasks = self.get_labels_each_image(selected_images)

        prepared_tasks, missing = self.prepare_data_ls_tasks(
            tasks, all_image_data, config
        )

        if len(missing) > 0:
            warnings.warn(f"Warning: {missing} frames were not found.")

        requests = self.ls_handler.post_ls_tasks(prepared_tasks)

        return requests


