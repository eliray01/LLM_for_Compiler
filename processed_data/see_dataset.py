from datasets import load_from_disk
from pathlib import Path

dataset = load_from_disk(str(Path(__file__).resolve().parent))
print(dataset)
if len(dataset) > 0:
    print(dataset[0].keys())
