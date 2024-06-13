import shutil
from pathlib import Path

from tqdm import tqdm

PATH = Path("")


def main():
    dir = PATH
    for split in ("train", "val", "test"):
        (dir / split).mkdir(exist_ok=True)
    for file in tqdm(dir.glob("*.tif")):
        file_num = int(file.stem.split("_")[0])
        if file_num < 1873:
            shutil.copy(file, dir / "train")
        elif file_num < 1873 + 535:
            shutil.copy(file, dir / "val")
        else:
            shutil.copy(file, dir / "test")


if __name__ == "__main__":
    main()
