import os
import argparse
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_cqt(filepath, sr=22050, hop_length=512, bins_per_octave=36, n_octaves=7):
    """Extract CQT Features from an audio file"""
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        cqt = librosa.cqt(y, sr=sr, hop_length=hop_length,
                            bins_per_octave=bins_per_octave,
                            n_bins=bins_per_octave * n_octaves)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt))
        return cqt_db.T
    except Exception as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")
        return None