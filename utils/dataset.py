import os
import torch
from torch.utils.data import Dataset

from utils.label_parser import load_chord_annotations, get_label_vector

class ChordDataset(Dataset):
    def __init__(self, features_dir, labels_dir, file_list, sr=22050, hop_length=512):
        """
        Args:
            features_dir (str): Path to .npy CQT features.
            labels_dir (str): Path to chord label (.lab) files.
            file_list (list): List of files without extensions (e.g. ['song1', 'song2']).
        """
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.file_list = file_list
        self.sr = sr
        self.hop_length = hop_length