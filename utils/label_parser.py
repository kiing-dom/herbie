import numpy as np
import os

def load_chord_annotations(filepath):
    """Load chord labels from a .lab or .txt file."""
    chords = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, label = parts
                chords.append((float(start), float(end), label))
    return chords

def get_label_vector(chords, total_frames, sr=22050, hop_length=512):
    """Convert chord annotations into a framewise label vector."""
    frame_times = np.arange(total_frames) * hop_length / sr
    labels = []

    for t in frame_times:
        label = "N"
        for start, end, chord in chords:
            if start <= t < end:
                label = chord
                break
        labels.append(label)
        
    return np.array(labels)