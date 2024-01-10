import ast
import shutil
from os import path

import pandas as pd
from PIL import Image

df = pd.read_csv(
    "data/imgs.csv",
    usecols=["image_path", "path_exists", "labels"],
    dtype={"image_path": "str", "path_exists": "str", "labels": "O"},
)

df = df.loc[df["path_exists"] == "True"]

for index in df.index:
    source_path = df["image_path"][index].replace("\\", "/")
    labels = ast.literal_eval(df["labels"][index])
    for label in labels:
        dest_path = f"imgs/dataset/{label}/"
        assert path.exists(source_path)
        assert path.exists(dest_path)
        try:
            im = Image.open(source_path)
            shutil.copy(source_path, dest_path)
        except IOError:
            print(f"Invalid file: {source_path}")
