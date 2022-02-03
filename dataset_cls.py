from cv2 import transform
from torch.utils.dataset import Dataset
from rich.console import Console
import albumentations as A
import torch
import pandas as pd


class vggface_ds(Dataset):
    def __init__(self, csv:str, transform = None):
        self.csv = csv
        self.trf = transform
        
    