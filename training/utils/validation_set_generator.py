import pandas as pd
from pathlib import Path
import random

path = Path("tumor-lungs/training/dataset.csv").resolve()
validation_set_length = 60

file = pd.read_csv(path)
n = random.sample(range(0, len(file)), validation_set_length)
n_2 = [i for i in range(0, len(file))]

for i in n_2:
    if i in n:
        file.iloc[i, 1] = "no"
    else: 
        file.iloc[i, 1] = "yes"

file.to_csv(path, sep=',', index=False)
