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
    
def process_folder(input_dir, output_dir, overwrite=False):
    """Process all audio files in a folder and save their CQT features as .npy"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = ['.wav', '.mp3', '.flac']

    for audio_file in tqdm(list(input_dir.glob("*"))):
        if audio_file.suffix.lower() not in audio_extensions:
            continue

        output_path = output_dir / (audio_file.stem + ".npy")
        if output_path.exists() and not overwrite:
            continue

        features = extract_cqt(str(audio_file))
        if features is not None:
            np.save(output_path, features)