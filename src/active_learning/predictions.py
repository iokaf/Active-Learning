"""This module is responsible for creating the dataset and dataloader 
for the pool of unlabeled images. Additioannly, it implements the 
functiosn for obtaining model predictions for these functions.

"""
import datetime
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union, Any
from tqdm.auto import tqdm

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


from src import get_transforms

class Predictor:
    """This class is responsible for collecting the predictions for
    the unlabeled images in the pool.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the class.

        Args:
            config: Dictionary containing the configuration.
        """
        self.config = config


    def create_prediction_image_dataset(
        self,
        image_data: List[Dict], testing: bool = False
    ) -> Dataset:
        """Create the image dataset for predictions.

        Creates a dataset that contains the image ids, paths, and crops
        for the images that will be used for predictions.

        Args:
            image_data: List of dictionaries containing the image ids,
                paths, and crops.
            testing: Boolean indicating whether the dataset is for testing
                or not. If True, the dataset will not contain the image
        """

        image_ids_paths_crops = [
            (image["image_id"], image["frame_path"], image["crop"]) for image in image_data
        ]

        image_ids, image_paths, image_crops = zip(*image_ids_paths_crops)

        transforms = get_transforms(self.config)

        return ImageDataset(image_ids, image_paths, image_crops, transforms)


    def create_image_dataloader(
        self, image_dataset: Dataset ) -> DataLoader:
        """Creates the image dataloader for the active learning.

        Args:
            image_dataset: Dataset containing the images to be used for
                predictions.

        Returns:
            dataloader: Dataloader for the image dataset.
        """
        batch_size = self.config["predictions"]["batch-size"]
        num_workers = self.config["predictions"]["workers"]
        dataloader = DataLoader(
            image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return dataloader


    def save_predictions(
            self, all_image_predictions: Dict[int, np.ndarray]) -> None:
        """Saves the predictions to a csv file.

        Args:
            all_image_predictions: Dictionary containing the predictions
                for all the images in the dataset. The keys are the image
                ids and the values are the predictions.
        """
        # Convert predictions to dataframe
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        outputs_dir = self.config["general"]["output-directory"]
        predictions_dir = os.path.join(outputs_dir, "predictions")

        os.makedirs(predictions_dir, exist_ok=True)
        predictions_filename = f"predictions_{current_datetime}.json"
        
        predictions_path = os.path.join(predictions_dir, predictions_filename)
        
        updated_predictions = {
            "__".join(key): value for key, value in all_image_predictions.items()
        }

        with open(predictions_path, "w") as f:
            json.dump(updated_predictions, f)

    def get_all_image_predictions(
        self,
        image_data: List[Dict],
        model: Union[torch.nn.Module, None] = None,
    ) -> Dict[int, np.ndarray]:
        """Gets the predictions for all the images in the dataset.

        Args:
            image_data: List of dictionaries containing the image ids,
                paths, and crops.
            model: Model to be used for predictions. If None, random
                predictions will be used.

        Returns:
            all_image_predictions: Dictionary containing the predictions
                for all the images in the dataset. The keys are the image
                ids and the values are the predictions.
        """
        dataset = self.create_prediction_image_dataset(image_data)
        dataloader = self.create_image_dataloader(dataset)

        all_image_predictions = {}
        model.eval()
        
        activ_func = self.config["predictions"]["activation"]

        if activ_func == "softmax":
            activation = torch.nn.Softmax(dim=1)
        elif activ_func == "sigmoid":
            activation = torch.nn.Sigmoid()
        else:
            raise ValueError(
                f"Activation function {activation} is not supported. "
                "Please use either softmax or sigmoid."
            )
    
        device = self.config["training"]["device"]
        round_digits = self.config["predictions"]["round-digits"]
        save_predictions = self.config["predictions"]["save-predictions"]
        model = model.to(device)
        with torch.inference_mode():
            for batch in tqdm(
                dataloader, total=len(dataloader), leave=False, desc="Predicting"
            ):
                images = batch["image"].to(device)
                image_ids = batch["image_id"]
                
                image_ids = list(zip(*image_ids))

                predictions = model(images).detach()
                predictions = activation(predictions)
                predictions = predictions.cpu().numpy().round(round_digits)
                predictions = predictions.tolist()
                
                id_preds = {
                    image_id: prediction
                    for image_id, prediction in zip(image_ids, predictions)
                }

                all_image_predictions.update(id_preds)

                del images, image_ids, predictions, id_preds
                torch.cuda.empty_cache()

        if save_predictions:
            self.save_predictions(all_image_predictions)

        return all_image_predictions


class ImageDataset(Dataset):
    """Image dataset for the active learning.

    When getting item, returns image tensor and image id.
    """

    def __init__(
            self, 
            image_ids: List[Tuple[int]], 
            image_paths: List[str], 
            image_crops: List[Tuple[int]],
            transforms: Any):
        """Initializes the image dataset.

        Args:
            image_ids (List[int]): List of image ids.
            image_paths (List(str)): List of image paths.
            image_crops (List(Tuple(int))): List of image crops.
            transforms (Optional): Transforms to be applied to the images. Default value is
                taken from the config file.
            testing (bool): Boolean indicating whether the dataset is for testing. If True, the
                dataset will not contain the image. Default value is False.
        """

        self.image_ids = image_ids
        self.image_paths = image_paths
        self.image_crops = image_crops
        self.transforms = transforms

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            length (int): Length of the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Returns the image tensor and image id for the given index.

        Args:
            idx (int): Index of the image.

        Returns:
            image (torch.Tensor): Image tensor.
            image_id (int): Image id.
        """
        image_id = self.image_ids[idx]

        # For testing purposes, we don't need to load the image
        image_path = self.image_paths[idx]

        image_path = image_path.replace("BackUp2", "BackUp3")
        crop = self.image_crops[idx]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not np.isnan(crop):
            image = image[crop[0] : crop[1], crop[2] : crop[3]]

        image = self.transforms(image=image)["image"]

        return {
            "image": image,
            "image_id": image_id,
        }
