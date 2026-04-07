import os
import shutil

SRC = r"D:\CT_Datasets\LDCT"
DEST = r"D:\CT_Datasets_clean_redcnn\LDCT"

for split in ["train", "val", "test"]:

    src_split = os.path.join(SRC, split)
    dest_split = os.path.join(DEST, split)

    os.makedirs(dest_split, exist_ok=True)

    for root, dirs, files in os.walk(src_split):
        for file in files:

            if file.endswith("_ldct.png") or file.endswith("_ndct.png"):

                src_path = os.path.join(root, file)

                # make unique name (avoid overwrite)
                new_name = root.replace("\\", "_").replace(":", "") + "_" + file
                dest_path = os.path.join(dest_split, new_name)

                shutil.copy(src_path, dest_path)

print("DONE")
