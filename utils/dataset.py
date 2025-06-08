import os
import torch
from torch.utils.data import Dataset
import numpy as np

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

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_id = self.file_list[idx]
        cqt_path = os.path.join(self.features_dir, file_id + ".npy")
        lab_path = os.path.join(self.labels_dir, file_id + ".lab")

        cqt = np.load(cqt_path)
        chords = load_chord_annotations(lab_path)
        labels = get_label_vector(chords, total_frames=cqt.shape[0],
                                  sr=self.sr, hop_length=self.hop_length)
        
        cqt_tensor = torch.tensor(cqt, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        return cqt_tensor, label_tensor