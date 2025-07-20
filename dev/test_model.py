from model import Model
from dataset import EmotionDataset
from pathlib import Path
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    model = Model(num_classes=8)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    # print(model)


    # load the dataset
    load_dotenv()
    path_datasets = Path(os.getenv("PATH_DATASETS"))
    dataset = EmotionDataset(root_dir=path_datasets, resample_rate=16_000)
    
    X, y, dataset_source = dataset[0]

    pred = model(X)