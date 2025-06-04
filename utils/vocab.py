TRIAD_VOCAB = [
    f"{root}:{quality}"
    for root in ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G#', 'A', 'A#', 'B']
    for quality in ['maj', 'min']
]

# "N": No chord 
# "X": unknown/unmapped
SPECIAL_TOKENS = ["N", "X"]
ALL_LABELS = SPECIAL_TOKENS + TRIAD_VOCAB

LABEL_TO_INDEX = {label: i for i, label in enumerate(ALL_LABELS)}
INDEX_TO_LABEL = {i: label for label, i, in LABEL_TO_INDEX.items()}

def label_to_index(label: str) -> int:
    """Map chord label string to index."""
    return LABEL_TO_INDEX.get(label, LABEL_TO_INDEX["X"])