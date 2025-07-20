"""
This downloads the datasets to be used for training and evaluation
"""
from dotenv import load_dotenv
import os
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import kagglehub
import shutil
# from pathlib import Path

def download_file_if_not_exists(url: str, save_path: str):
    if save_path.exists():
        print(f"File {save_path} already exists, skipping download...")
        return
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    wrote = 0

    with open(save_path, "wb") as file, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {save_path.name}"
    ) as t:
        for data in response.iter_content(block_size):
            wrote = wrote + len(data)
            file.write(data)
            t.update(len(data))
    if total_size != 0 and wrote != total_size:
        print("WARNING: Downloaded size does not match expected size!")

def extract_zip(zip_path: str, extract_path: str):
    if (extract_path / zip_path.name.split(".")[0]).exists():
        print(f'{zip_path.name.split(".")[0]} already extracted, skipping...')
        return
    print(f'Extracting {zip_path.name} to {extract_path}...')

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    with open(extract_path / zip_path.name.split(".")[0], 'w') as f:
        f.write(f'Extracted {zip_path.name} to {extract_path}')

    print(f'Extracted {zip_path.name} to {extract_path}')

def move_dataset_from_kaggle(dataset_path: str, save_path: str):
    dataset_path = Path(dataset_path)
    if dataset_path.resolve() != save_path.resolve():
        for item in dataset_path.iterdir():
            dest = save_path / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(item), str(dest))
            else:
                shutil.move(str(item), str(dest))
        # Optionally remove the original empty folder
        try:
            dataset_path.rmdir()
        except Exception:
            pass

def download_ravdess(save_path: str):

    print("Downloading RAVDESS dataset...")

    url1 = "https://zenodo.org/records/1188976/files/Audio_Song_Actors_01-24.zip?download=1"
    url2 = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

    # create folder if it doesn't exist
    # if not save_path.exists():
    save_path = save_path / "RAVDESS"
    save_path.mkdir(parents=True, exist_ok=True)

    download_file_if_not_exists(url1, save_path / "Audio_Song_Actors_01-24.zip")
    download_file_if_not_exists(url2, save_path / "Audio_Speech_Actors_01-24.zip")

    # unzip the files
    print("Unzipping RAVDESS dataset...")
    actors_path = save_path / "actors"
    actors_path.mkdir(parents=True, exist_ok=True)
    extract_zip(save_path / "Audio_Song_Actors_01-24.zip", actors_path)

    speech_path = save_path / "speech"
    speech_path.mkdir(parents=True, exist_ok=True)
    extract_zip(save_path / "Audio_Speech_Actors_01-24.zip", speech_path)
    print("Done")

def download_savee(save_path: str):
    # check if the path is empty:
    if (save_path / "SAVE-E").exists():
        print(f"SAVE-E dataset already exists in {save_path}, skipping download...")
        return
    
    print("Downloading SAVE-E dataset...")
    save_path = save_path / "SAVE-E"
    save_path.mkdir(parents=True, exist_ok=True)


    # Download to default location, then move to save_path
    dataset_path = kagglehub.dataset_download("ejlok1/surrey-audiovisual-expressed-emotion-savee")
    
    # Move the downloaded dataset to the desired save_path if it's not already there
    move_dataset_from_kaggle(dataset_path, save_path)

    

def download_tess(save_path: str):
    if (save_path / "TESS").exists():
        print(f"TESS dataset already exists in {save_path}, skipping download...")
        return
    
    print("Downloading TESS dataset...")
    save_path = save_path / "TESS"
    save_path.mkdir(parents=True, exist_ok=True)

    # download the dataset from the google drive
    dataset_path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")

    # Move the downloaded dataset to the desired save_path if it's not already there
    move_dataset_from_kaggle(dataset_path, save_path)

def download_jl_corpus(save_path: str):
    if (save_path / "JL-Corpus").exists():
        print(f"JL-Corpus dataset already exists in {save_path}, skipping download...")
        return
    
    print("Downloading JL-Corpus dataset...")
    save_path = save_path / "JL-Corpus"
    save_path.mkdir(parents=True, exist_ok=True)

    # download the dataset from the google drive
    dataset_path = kagglehub.dataset_download("tli725/jl-corpus")

    # Move the downloaded dataset to the desired save_path if it's not already there
    move_dataset_from_kaggle(dataset_path, save_path)

def download_crema_d(save_path: str):
    if (save_path / "CREMA-D").exists():
        print(f"CREMA-D dataset already exists in {save_path}, skipping download...")
        return
    
    print("Downloading CREMA-D dataset...")
    save_path = save_path / "CREMA-D"
    save_path.mkdir(parents=True, exist_ok=True)

    # download the dataset from kaggle
    dataset_path = kagglehub.dataset_download("ejlok1/cremad")

    # Move the downloaded dataset to the desired save_path if it's not already there
    move_dataset_from_kaggle(dataset_path, save_path)

def main():
    load_dotenv()

    save_path = Path(os.getenv("PATH_DATASETS"))

    download_ravdess(save_path)
    download_savee(save_path)
    download_tess(save_path)
    download_jl_corpus(save_path)
    download_crema_d(save_path)

if __name__ == "__main__":
    main()
