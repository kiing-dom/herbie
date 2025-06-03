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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CCQT features from audio files.")
    parser.add_argument("--input", required=True, help="Input file or directory of audio files")
    parser.add_argument("--output", default="features/", help="Where to save .npy features")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        process_folder(args.input, args.output, overwrite=args.overwrite)
    elif os.path.isfile(args.input):
        Path(args.output).mkdir(parents=True, exist_ok=True)
        features = extract_cqt(args.input)
        if features is not None:
            np.save(Path(args.output) / (Path(args.input).stem + ".npy"), features)
    else:
        print(f"[ERROR] Input path not found: {args.input}")