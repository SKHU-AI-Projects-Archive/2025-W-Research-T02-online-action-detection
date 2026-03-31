import sys
sys.path.insert(0, '.')
from utils.env import Configuration
from datasets import PKUMMD
from collections import Counter

config = Configuration('configs/pkummd_baseline.ini')
dataset = PKUMMD(split='train', config=config)

x, y = dataset[0]
print("x shape:", x.shape)
print("y shape:", y.shape)
print("y sample:", y)
print("unique in first sample:", y.unique())

# 윈도우 1000개 샘플링
all_labels = []
for i in range(min(1000, len(dataset))):
    x, y = dataset[i]
    all_labels.extend(y.tolist())

c = Counter(all_labels)
print("\nbg(0):", c[0])
print("action(non-0):", sum(v for k,v in c.items() if k != 0))
print("most common:", c.most_common(10))