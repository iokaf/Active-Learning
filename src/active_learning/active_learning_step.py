"""This module contains the implementation of the active learning iteration."""

import os
from typing import Dict, Any

from src import (
    LabelStudioHandler, AnnotationsHandler, Trainer, DataInclusion,
    Predictor, FrameSelector
)


class ActiveLearning:
    """A class for performing active learning iterations.
    """
    def __init__(
            self, config: Dict, Model: Any):
        """Initialize the class.
        
        Args:
            config: A dictionary containing the configuration.

            model: The model class. This is passed as an argument
                as the user can select their own model.
        """

        self.config = config
        self.Model = Model

        self.ls_handler = LabelStudioHandler(config=config)
        self.annotations_handler = AnnotationsHandler(
            config=config,
            ls_handler=self.ls_handler
        )

        self.trainer = Trainer(
            config=config,
            Model=self.Model
        )

        self.data_inclusion = DataInclusion(config=config)
        self.predictor = Predictor(config=config)

    def get_active_learning_step(self):
        """Get the current active learning step."""
        output_dir = self.config["general"]["output-directory"]
        al_step_path = os.path.join(output_dir, "al_step.txt")

        if not os.path.exists(al_step_path):
            os.makedirs(output_dir, exist_ok=True)

            with open(al_step_path, "w") as f:
                f.write("0")
            return 0
        else:
            with open(al_step_path, "r") as f:
                al_step = int(f.read())
            return al_step


    def update_al_step(self, current_step: int) -> None:
        """Increases the stored active learning step by one.
        
        Args:
        current_step (int): The current active learning step.
        """
        output_dir = self.config["general"]["output-directory"]
        al_step_path = os.path.join(output_dir, "al_step.txt")

        with open(al_step_path, "w") as f:
            f.write(str(current_step + 1))
        return


    def iterate(self):
        """Perform a complete active learning iteration.
        
        Notes:
        ------
        The steps for the iteration are the following:
        1. Read the current active learning step number from the output file.

        2. Get the label studio project id from the config file.
        3. Collect all annotations from the label studio project.


        """
        al_step = self.get_active_learning_step()

        # Get current annotations
        project_id = self.config["label-studio"]["project-id"]

        # ----------------------------------------------------------
        # self.ls_handler.get_current_annotations()

        # # Now delete the tasks
        self.ls_handler.delete_all_tasks()
        # ----------------------------------------------------------

        # # Collect all annotation jsons
        annotations = self.annotations_handler.collect_all_ls_annotations()
        
        # # FixMe: This is for testing purposes only
        annotations = []

        print(f"Total Number of Annotations: {len(annotations)}")

        if len(annotations) == 0:
            # This is for the first step, we should load the base model here
            model = self.Model(self.config)

            device = self.config["training"]["device"]
            model = model.to(device)
        else:
            # Train model
            model = self.trainer.train_model(annotations=annotations, al_step_num=al_step)


        # Get all data, filter image ids that have already been used and select only the unused ones
        # ToDo: This filtering can not be done in the sparse annotations case. For sparse annotations 
        # we can only exclude images that have been used for all labels.
        
        
        all_image_data = self.data_inclusion.get_all_image_data()
        
        used_image_ids_per_label = self.data_inclusion.read_image_ids_per_label() 

        all_image_ids_used = self.data_inclusion.read_image_ids_used()
        
        all_image_ids_used = list(map(tuple, all_image_ids_used))
        images_this_step = self.data_inclusion.select_data()

        print(f"All data: {len(all_image_data)}, used already: {len(all_image_ids_used)}, availablethis step: {len(images_this_step)}")

        images_this_step = images_this_step[:500]

        # # Use new model to predict
        predictions = self.predictor.get_all_image_predictions(images_this_step, model=model) # model=model)

        # # # Make selection of images
        frame_selector = FrameSelector(
            config=self.config,
            predictions=predictions,
            image_ids_each_label=used_image_ids_per_label,
        )

        selected_frames = frame_selector.select_frames()

        print(selected_frames)

        self.data_inclusion.update_image_ids_used(
            used_image_ids_per_label, selected_frames
        )

        self.data_inclusion.write_image_ids_per_label(used_image_ids_per_label)

        # # Create label studio tasks
        request = self.annotations_handler.post_selected_images_to_ls(
            selected_frames, images_this_step, self.config
        )

        # # Update active learning step
        self.update_al_step(al_step)

        return request