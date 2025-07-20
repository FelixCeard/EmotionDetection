"""
Check that all the datasets are installed correctly
"""

from pathlib import Path
from dotenv import load_dotenv
import os

def check_crema_d(save_path: Path) -> bool:
    """Check CREMA-D dataset"""
    crema_path = save_path / "CREMA-D"
    if not crema_path.exists():
        print(f"❌ CREMA-D dataset not found in {save_path}")
        return False
    
    audio_path = crema_path / "AudioWAV"
    if not audio_path.exists():
        print(f"❌ CREMA-D dataset is not installed correctly... Expected path: {audio_path} but it doesn't exist")
        return False
    
    print(f"✅ CREMA-D dataset found at {crema_path}")
    return True

def check_esd(save_path: Path) -> bool:
    """Check ESD (Emotional Speech Dataset)"""
    esd_path = save_path / "ESD"
    if not esd_path.exists():
        print(f"❌ ESD dataset not found in {save_path}")
        return False
    
    # Check for speaker directories (0001-0020)
    expected_speakers = [f"{i:04d}" for i in range(1, 21)]
    missing_speakers = []
    
    for speaker in expected_speakers:
        speaker_path = esd_path / speaker
        if not speaker_path.exists():
            missing_speakers.append(speaker)
    
    if missing_speakers:
        print(f"❌ ESD dataset missing speaker directories: {missing_speakers}")
        return False
    
    print(f"✅ ESD dataset found at {esd_path} with all 20 speakers")
    return True

def check_jl_corpus(save_path: Path) -> bool:
    """Check JL-Corpus dataset"""
    jl_path = save_path / "JL-Corpus"
    if not jl_path.exists():
        print(f"❌ JL-Corpus dataset not found in {save_path}")
        return False
    
    # Check for expected files and directories
    expected_items = [
        "README.md",
        "Corpus Setup.docx", 
        "Recording Context.docx",
        "Picture Stimuli",
        "Raw JL corpus (unchecked and unannotated)"
    ]
    
    missing_items = []
    for item in expected_items:
        item_path = jl_path / item
        if not item_path.exists():
            missing_items.append(item)
    
    if missing_items:
        print(f"❌ JL-Corpus dataset missing items: {missing_items}")
        return False
    
    print(f"✅ JL-Corpus dataset found at {jl_path}")
    return True

def check_ravdess(save_path: Path) -> bool:
    """Check RAVDESS dataset"""
    ravdess_path = save_path / "RAVDESS"
    if not ravdess_path.exists():
        print(f"❌ RAVDESS dataset not found in {save_path}")
        return False
    
    # Check for expected directories and files
    expected_items = [
        "actors",
        "speech", 
        "Audio_Song_Actors_01-24.zip",
        "Audio_Speech_Actors_01-24.zip"
    ]
    
    missing_items = []
    for item in expected_items:
        item_path = ravdess_path / item
        if not item_path.exists():
            missing_items.append(item)
    
    if missing_items:
        print(f"❌ RAVDESS dataset missing items: {missing_items}")
        return False
    
    print(f"✅ RAVDESS dataset found at {ravdess_path}")
    return True

def check_save_e(save_path: Path) -> bool:
    """Check SAVE-E dataset"""
    save_e_path = save_path / "SAVE-E"
    if not save_e_path.exists():
        print(f"❌ SAVE-E dataset not found in {save_path}")
        return False
    
    all_path = save_e_path / "ALL"
    if not all_path.exists():
        print(f"❌ SAVE-E dataset missing ALL directory at {all_path}")
        return False
    
    print(f"✅ SAVE-E dataset found at {save_e_path}")
    return True

def check_tess(save_path: Path) -> bool:
    """Check TESS dataset"""
    tess_path = save_path / "TESS"
    if not tess_path.exists():
        print(f"❌ TESS dataset not found in {save_path}")
        return False
    
    tess_data_path = tess_path / "TESS Toronto emotional speech set data"
    if not tess_data_path.exists():
        print(f"❌ TESS dataset missing data directory at {tess_data_path}")
        return False
    
    print(f"✅ TESS dataset found at {tess_path}")
    return True

def main():
    load_dotenv()
    save_path = Path(os.getenv("PATH_DATASETS"))
    
    if not save_path:
        print("❌ PATH_DATASETS environment variable not set")
        return
    
    print(f"Checking datasets in: {save_path}")
    print("-" * 50)
    
    # Check all datasets
    results = {
        "CREMA-D": check_crema_d(save_path),
        "ESD": check_esd(save_path),
        "JL-Corpus": check_jl_corpus(save_path),
        "RAVDESS": check_ravdess(save_path),
        "SAVE-E": check_save_e(save_path),
        "TESS": check_tess(save_path)
    }
    
    print("-" * 50)
    print("Summary:")
    all_good = all(results.values())
    
    for dataset, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {dataset}")
    
if __name__ == "__main__":
    main()
    


