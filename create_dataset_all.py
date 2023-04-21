import shutil
import os

from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

samples = glob("data/consep/*/*.npy")
samples.extend(glob("data/liz1/*/*.npy"))
samples.extend(glob("data/pan1/*/*.npy"))

X_train, X_test, y_train, y_test = train_test_split(samples,
                                                    samples,
                                                    test_size=0.1,
                                                    random_state=42)

for sample in tqdm(X_train, total=len(X_train)):
    shutil.copyfile(sample, f"data/all/train/{os.path.basename(sample)}")

for sample in tqdm(X_test, total=len(X_test)):
    shutil.copyfile(sample, f"data/all/test/{os.path.basename(sample)}")
