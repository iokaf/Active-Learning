import toml


with open("config.toml", "r") as f:
    my_config = toml.load(f)

from src.active_learning.ls_handler import LabelStudioHandler, AnnotationsHandler
from src.active_learning.transforms import get_transforms, get_augmentations
from src.active_learning.trainer import Trainer
from src.active_learning.video_and_image_selector import DataInclusion
from src.active_learning.predictions import Predictor
from src.active_learning.frame_selector import FrameSelector
from src.active_learning.active_learning_step import ActiveLearning