"""This module contains the class and functionality responsible 
for the acquisition of frames for annotation.

Methods implemented here are:
    - Random
    - Uncertain
    - Modified Uncertain
    - Diverse (The one proposed in the paper)

Parameters to get from the configuration file:
CONFIG_DIR: Directory where the configuration files are stored.
LABELSET: List of labels for the dataset.
MUTUALLY_EXCLUSIVE: If True, the labels are mutually exclusive.
SELECTION_METHOD: The method used to select the frames.
NUM_FRAMES_PER_LABEL: Number of frames to select per label.
SAVE_SELECTION_RESULTS: If True, the results of the frame selection
    will be saved.
SAVE_SELECTION_RESULTS_DIR: Directory where the results of the frame
    selection will be saved.
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple

import datetime

import numpy as np




class FrameSelector:
    """This class implements the frame selection methods."""

    def __init__(
        self,
        config: Dict,
        predictions: Dict[int, List[float]],
        image_ids_each_label: Dict[str, List[int]],
    ) -> None:
        """Initializes the FrameSelector instance.
        The arguments are saved as attributes of the class.

        The difference between mutual exclusive and non-mutual exclusive
        in the selection process is that in the first case, the frames
        selected for each label are not allowed to be selected for the
        other labels. In the second case, the frames selected for each
        label can be selected for the other labels.


        Args:
            config (Dict): The configuration dictionary.
            predictions (Dict[int, list(float)]): The predictions for each
                frame. The keys are the frame ids and the values are the
                predictions for each label.
            image_ids_each_label (Dict[str, list(int)]): The image ids
                selected per label over the previous AL steps.
        """
        # A dictionary with labels and key and a list of image ids that have already been selected
        # for that label as value

        self.config = config
        self.image_ids_each_label = image_ids_each_label

        # A list of all image ids that have been selected so far
        self.all_image_ids = []
        for _, ids in self.image_ids_each_label.items():
            self.all_image_ids.extend(ids)

        # Have a list of functions that can be called
        self.method_functions = {
            "random": self.select_random_label_frames,
            "diverse": self.select_diverse_label_frames,
        }

        # Based on the config file, select the method
        selection_method = config["task-selection"]["method"]
        self.acquisition_function = self.method_functions[selection_method]

        # self.frame_ids are the id of the frames to be selected
        self.frame_ids, self.predictions = zip(*predictions.items())
        self.predictions = np.array(self.predictions)

    def select_frames(self) -> Dict[str, List[int]]:
        """Select diverse frames for each label in the selected labelset.

        Returns:
            Dict[str, list[int]]: A dictionary where keys are the labels
            and values are the frame ids for the frames selected for the label.
        """

        selected_frames = {}

        labels = self.config["labels"]["names"]
        for label in labels:
            # ToDo: For the selection where we make one project per label,
            # this should only be the label selected indices
            indices_to_skip = self.all_image_ids
            # else:
            #    indices_to_skip = self.image_ids_each_label[label]

            selection_result = self.acquisition_function(label, indices_to_skip)

            # Get the image ids for the selected frames for the label
            selected_frames.update(selection_result)

            # We update the all_image_ids to include the selected images
            self.all_image_ids.extend([pred[0] for pred in selection_result[label]])

        save_selection_results = self.config["task-selection"]["save-results"]
        if save_selection_results:
            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_dir = self.config["general"]["output-directory"]
            selection_dir = os.path.join(output_dir, "selection-results")
            os.makedirs(selection_dir, exist_ok=True)

            selection_filename = f"{current_datetime}.txt"
            selection_filepath = os.path.join(selection_dir, selection_filename)

            # Save the selected frames dictionary
            with open(selection_filepath, "w") as file:
                file.write(str(selected_frames))

        return selected_frames

    def get_label_predictions(self, label: str) -> np.array:
        """Returns the predictions for a given label.

        Args:
            label (str): The label.

        Returns:
            np.array: The predictions for the label.

        Raises:
            ValueError: If the label is not valid.
        """

        labels = self.config["labels"]["names"]
        if label not in labels:
            raise ValueError("Label not found in the LABELSET.")

        label_idx = labels.index(label)

        return self.predictions[:, label_idx]

    def rearrange_label_predictions(
        self, predictions: np.array
    ) -> List[Tuple[int, List[float]]]:
        """Rearranges the predictions for a given label to be sorted by the
        image_id in descending order.

        Args:
            predictions (np.array): The predictions for a given label.

        Returns:
            List[Tuple[int, List[float]]]: A list of tuples where the first
            element is the frame id and the second element is the prediction
            for the label. The list is sorted by the prediction in descending
            order of the frame id.
        """
        pairs = list(zip(self.frame_ids, predictions))
        pairs.sort(key=lambda x: x[0], reverse=True)

        return pairs

    def prepare_predictions_for_selection(
        self, label: str, ids_to_skip: List[int]
    ) -> Tuple[List[int], np.array, List[Tuple[int, List[float]]]]:
        """Prepares the predictions for a given label to be used in the
        selection process.

        Args:
            label (str): The label.
            ids_to_skip (List[int]): The image ids that should be skipped

        Returns:
            Tuple[List[int], np.array]: A tuple where the first element is
            the frame ids and the second element is the predictions for the
            label.
        """
        predictions = self.get_label_predictions(label)
        rearranged_predictions = self.rearrange_label_predictions(predictions)

        # ! Filter out image ids that already have been annotated
        rearranged_predictions = [
            pred for pred in rearranged_predictions if pred[0] not in ids_to_skip
        ]

        # Separate the frame ids and the predictions but keep the order
        frame_indices, predictions = zip(*rearranged_predictions)

        return frame_indices, np.array(predictions), rearranged_predictions

    def select_random_label_frames(
        self, label: str, ids_to_skip: List[int]
    ) -> Dict[str, List[int]]:
        """Randomly select images to be annotated for each label.

        Args:
            label (str): The label to select images for.
            ids_to_skip (List[int]): The image ids that should not be included in
                the selection process.

        Returns:
            A dictionary where keys are the labels and values are the frame ids
            of the frames selected for the label.
        """
        # Separate the frame ids and the predictions but keep the order
        (
            frame_indices,
            _,
            rearranged_predictions,
        ) = self.prepare_predictions_for_selection(label, ids_to_skip)

        num_tasks_per_label = self.config["task-selection"]["number-of-tasks-per-label"]
        # Select frame indices randomly
        selected_frames = np.random.choice(
            frame_indices, num_tasks_per_label, replace=False
        )

        # Get the indices for the selected frames
        selected_indices = [frame_indices.index(frame) for frame in selected_frames]

        # Get the selected predictions
        selected_predictions = [
            rearranged_predictions[index] for index in selected_indices
        ]

        return {label: selected_predictions}

    def select_diverse_label_frames(self, label: str, ids_to_skip: List[int]):
        """Select images to be annotated for each label.

        This is the selection method used in the paper.

        Args:
            label: The label to be selected.
            ids_to_skip (List[int]): The image ids to be excluded from the selection.

        Returns:
            A dictionary where keys are the labels names (str) and values are
            tuples of two elements, the first is the image id (first column of all images)
            and the second is the prediction probability.
        """
        (
            ids_to_select_from,
            predictions,
            rearranged_predictions,
        ) = self.prepare_predictions_for_selection(label, ids_to_skip)

        bins, _ = np.histogram(predictions, bins=10, range=(0, 1))

        num_nonempty_bins = sum(bins > 0)

        image_selection_probabilities = []
        for p_bin in bins:
            if p_bin > 0:
                image_selection_probabilities.extend(
                    p_bin * [1 / (num_nonempty_bins * p_bin)]
                )

        image_selection_probabilities = image_selection_probabilities[::-1]

        # If the number of images to select from is less than the required number,
        # we select all the images. Otherwise, we apply the selection probabilities.
        num_tasks_per_label = self.config["task-selection"]["number-of-tasks-per-label"]

        

        if len(ids_to_select_from) <= num_tasks_per_label:
            selected_image_ids = ids_to_select_from
        else:
            selected_image_ids_indices = np.random.choice(
                range(len(ids_to_select_from)),  # The ids to select from
                num_tasks_per_label,  # The number of ids to select
                replace=False,
                p=image_selection_probabilities,  # The probabilities for each image
            )

            selected_image_ids = [ids_to_select_from[idx] for idx in selected_image_ids_indices]

        # Get the predictions for the selected images
        selected_predictions = [
            rearranged_prediction
            for rearranged_prediction in rearranged_predictions
            if rearranged_prediction[0] in selected_image_ids
        ]

        return {label: selected_predictions}
