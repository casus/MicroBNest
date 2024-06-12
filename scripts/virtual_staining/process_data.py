import os

import cv2
import numpy as np
from tqdm import tqdm

PATH = ""


def read_tiff_files(directory):
    tiff_files = [
        file
        for file in os.listdir(directory)
        if file.endswith(".tif") or file.endswith(".tiff")
    ]

    train_images = []
    val_images = []
    test_images = []

    for file in tqdm(tiff_files):
        file_path = os.path.join(directory, file)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        file_num = int(file.split(".")[0])
        if file_num < 1873:
            train_images.append(image)
        elif file_num < 1873 + 535:
            val_images.append(image)
        else:
            test_images.append(image)

    return np.array(train_images), np.array(val_images), np.array(test_images)


train, val, test = read_tiff_files(PATH)

print(train.shape)
print(val.shape)
print(test.shape)

# cyt -> nuc
np.save(f"cyt_to_nuc/train/x.npy", train[:, :, :, 2])
np.save(f"cyt_to_nuc/train/y.npy", train[:, :, :, 0])
np.save(f"cyt_to_nuc/val/x.npy", val[:, :, :, 2])
np.save(f"cyt_to_nuc/val/y.npy", val[:, :, :, 0])
np.save(f"cyt_to_nuc/test/x.npy", test[:, :, :, 2])
np.save(f"cyt_to_nuc/test/y.npy", test[:, :, :, 0])


np.save(f"nuc_to_cyt/train/x.npy", train[:, :, :, 0])
np.save(f"nuc_to_cyt/train/y.npy", train[:, :, :, 2])
np.save(f"nuc_to_cyt/val/x.npy", val[:, :, :, 0])
np.save(f"nuc_to_cyt/val/y.npy", val[:, :, :, 2])
np.save(f"nuc_to_cyt/test/x.npy", test[:, :, :, 0])
np.save(f"nuc_to_cyt/test/y.npy", test[:, :, :, 2])
