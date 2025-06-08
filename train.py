import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline import ChordRecognitionModel
from utils.dataset import ChordDataset
