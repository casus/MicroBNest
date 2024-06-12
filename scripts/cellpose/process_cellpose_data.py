import os
import shutil

from tqdm import tqdm

source_dir_imgs = "/bigdata/casus/MLID/nips_benchmark/rb_images_normalized/half_res"
source_dir_masks = "/bigdata/casus/MLID/nips_benchmark/hela_cyto_masks/half_res"
target_dir = "/bigdata/casus/MLID/nips_benchmark/hela_cyto_cellpose"

os.makedirs(target_dir, exist_ok=True)


def copy_and_rename_files(source_dir, target_dir, suffix):
    for filename in tqdm(os.listdir(source_dir)):
        source_path = os.path.join(source_dir, filename)
        if os.path.isfile(source_path):
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}{suffix}{ext}"
            target_path = os.path.join(target_dir, new_filename)
            shutil.copy2(source_path, target_path)


copy_and_rename_files(source_dir_imgs, target_dir, "_img")
copy_and_rename_files(source_dir_masks, target_dir, "_masks")
