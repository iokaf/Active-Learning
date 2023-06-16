"""This module is responsible for training the model and saving model checkpoints.
"""
import os
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Union, Any

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from src import get_augmentations, get_transforms

class Trainer:
    """This class is responsible for training the model.
    """

    def __init__(
            self, 
            config: Dict, 
            Model: torch.nn.Module):
        """Initializes the Trainer class."""

        self.config = config
        self.Model = Model

        self.transforms = get_transforms(config)
        self.augmentations = get_augmentations(config)

    def get_chechoints_directory(self) -> str:
        """Returns the directory where the checkpoints are saved.

        Returns:
            str: Directory where the checkpoints are saved.
        """
        outputs_dir = self.config["general"]["output-directory"]
        cpt_dir_name = self.config["training"]["checkpoints-directory-name"]

        checkpoints_dir = os.path.join(outputs_dir, cpt_dir_name)

        os.makedirs(checkpoints_dir, exist_ok=True)

        return checkpoints_dir



        
        
    def train_valid_split(
            self,
            annotations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split the annotations into train and validation sets.

        In this project, each image source (examination) is assigned a random 
        integer, called a fold. The folds take values from 0 to 99. Assigning a 
        data point as training or validation is done by taking the remainder of
        the fold number when divided by 10. If the remainder is in the list of
        validation folds, then the data point is assigned to the validation set.
        Otherwise, it is assigned to the training set.

        Args:
            annotations (List[Dict]): List of annotations.

        Returns:
            Tuple[List[Dict], List[Dict]]: Train and annotations
        """

        valid_folds = self.config["data"]["valid-folds"]
        verbose = self.config["training"]["verbose"]

        train_annotations = []
        valid_annotations = []

        if verbose:
            annotations = tqdm(annotations, 
                desc="Splitting into train and validation sets"
            )

        for annotation in annotations:
            if annotation["fold"] % 10 in valid_folds:
                valid_annotations.append(annotation)
            else:
                train_annotations.append(annotation)

        return train_annotations, valid_annotations


    def prepare_train_data(
        self,
        annotations: List[Dict],
    ) -> Tuple[DataLoader, DataLoader, Dataset]:
        """Prepare the train and validation data loaders.

        The Training dataset is returned in case its length is needed for a learning rate scheduler.

        Args:
            annotations (List[Dict]): List of annotations.

        Returns:
            Tuple[DataLoader, DataLoader, Dataset]: Train and validation data loaders and
                the train dataset. First element of the tuple is the train data loader,
                second element is the validation data loader and the third element is
                the train dataset.
        """
        train_annotations, valid_annotations = self.train_valid_split(annotations)

        train_ds = TrainImageDataset(
            train_annotations, self.transforms, self.augmentations
        )
        valid_ds = TrainImageDataset(valid_annotations, self.transforms)

        print(f"Train dataset length: {len(train_ds)}")
        print(f"Valid dataset length: {len(valid_ds)}")

        train_bs = self.config["training"]["train-batch-size"]
        valid_bs = self.config["training"]["valid-batch-size"]

        train_workers = self.config["training"]["train-workers"]
        valid_workers = self.config["training"]["valid-workers"]

        train_dl = DataLoader(
            train_ds, batch_size=train_bs, shuffle=True, 
            num_workers=train_workers, pin_memory=True
        )
        valid_dl = DataLoader(
            valid_ds, batch_size=valid_bs, shuffle=False, 
            num_workers=valid_workers, pin_memory=True
        )

        return {
            "train_data_loader": train_dl,
            "valid_data_loader": valid_dl,
            "train_dataset": train_ds,
        }


    def train_model(
        self,
        annotations: List[Dict], al_step_num: int,
        model: Union[torch.nn.Module, None] = None,
        ) -> torch.nn.Module:
        """ Train the model.

        Args:
            annotations (List[Dict]): List of annotations.
            al_step_num (int): Active learning step number.

        Returns:
            Model: Trained model.
        """

        # Get necessary parameters from the config file

        device = self.config["training"]["device"]
        max_epochs = self.config["training"]["max-epochs"]

        min_checkpoint_epoch = self.config["training"]["min-checkpoint-epoch"]



        if model is None:
            model = self.Model(self.config)
            model.to(device)

        loaders = self.prepare_train_data(annotations)

        train_dl = loaders["train_data_loader"]
        valid_dl = loaders["valid_data_loader"]

        
        smallest_val_loss = None

        train_bar = tqdm(
            range(max_epochs), desc="Training", leave=False
        )

        train_results = {"losses": [], "targets": [], "logits": []
        }

        valid_results = {"losses": [], "targets": [], "logits": []
        }


        # Create a training loop for the model
        for epoch in train_bar:

            # Set the model to train mode
            model.train()

            epoch_steps_bar = tqdm(
                train_dl, total=len(train_dl), desc=f"Epoch {epoch}", leave=False
            )
            train_loss = 0

            for data in epoch_steps_bar:
                # Move the data to the correct device
                image = data["image"].to(device)
                target = data["label"].to(device)

                # Zero the gradients
                model.optimizer.zero_grad()
                # Forward pass
                output = model(image)

                # Calculate the loss
                loss = model.criterion(output, target)

                # Backward pass
                loss.backward()

                # Update the weights
                model.optimizer.step()

                train_results["losses"].append(loss.item())
                train_results["targets"].extend(target.detach().cpu().numpy().tolist())
                train_results["logits"].extend(output.detach().cpu().numpy().tolist())

                # Update the training loss
                train_loss += loss.item()

                del loss, output, data, target
                if device == "cuda":
                    torch.cuda.empty_cache()

            # Calculate the training loss
            train_loss /= len(train_dl)

            # Set the model to eval mode
            model.eval()
            # Initialize the validation loss
            val_loss = 0

            with torch.inference_mode():
                # Loop over the validation data

                valid_bar = tqdm(
                    valid_dl, total=len(valid_dl), 
                    desc=f"Epoch {epoch} - Validation", leave=False
                )

                for batch in valid_bar:
                    # Move the data to the correct device
                    image = batch["image"].to(device)
                    target = batch["label"].to(device)

                    # Forward pass
                    output = model(image)
                    # Calculate the loss
                    loss = model.criterion(output, target)

                    # Update the validation loss
                    val_loss += loss.item()

                    valid_results["losses"].append(loss.item())
                    valid_results["targets"].extend(target.detach().cpu().numpy().tolist())
                    valid_results["logits"].extend(output.detach().cpu().numpy().tolist())

                # Calculate the average validation loss
                val_loss /= len(valid_dl)

                # Write the validation loss on the progress bar
                description = f"Epoch {epoch} - train loss {train_loss:05f} - val loss {val_loss:.5f}"
                train_bar.set_description(description)

                if epoch < min_checkpoint_epoch:
                    continue

                if smallest_val_loss is None or val_loss <= smallest_val_loss:
                    print(
                        f"New best model found at epoch {epoch} with val loss {val_loss:.5f}"
                    )
                    print("")
                    smallest_val_loss = val_loss
                    best_model = model.state_dict()

                    checkpoint_dir = self.get_chechoints_directory()
                    checkpoint_name = f"AL_iteration_{al_step_num}_{model.name}_epoch_{epoch}_val_loss_{val_loss:.5f}.pt"
                    best_model_path = os.path.join(checkpoint_dir, checkpoint_name)

        # Save the state dict
        torch.save(best_model, best_model_path)

        # Load the best model
        model.load_state_dict(best_model)

        # Set the model to eval mode and return it
        model.to(device)
        model.eval()
        return model


class TrainImageDataset(Dataset):
    """Train and validation dataset"""

    def __init__(
        self,
        annotations: List[Dict],
        transform: Any,
        augmentations: Union[Any, None] = None,
    ):
        """Initialize the dataset.

        Args:
            annotations (List[Dict]): List of annotations.
            transform (Any): Transformations to apply to the images.
            augmentations (Union[Any, None], optional): Augmentations
            to apply to the images. Defaults to None.
        """
        self.annotations = annotations
        self.transform = transform
        self.augmentations = augmentations

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return an item from the dataset.

        Args:
            idx (int): Index of the item to return.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and label.
        """

        annotation = self.annotations[idx]
        image_path = annotation["new_image_path"]
        image_path = image_path.replace("BackUp2", "BackUp3")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            image = self.augmentations(image=image)["image"] 
        if self.transform:
            image = self.transform(image=image)["image"]

        label = torch.Tensor(annotation["annotation"]).float()

        result =  {
            "image": image,
            "label": label,
        }

        return result
