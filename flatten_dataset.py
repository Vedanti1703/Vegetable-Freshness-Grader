import os
import shutil

base_path = "dataset/Test"

for category in ["fresh", "rotten"]:
    category_path = os.path.join(base_path, category)

    for folder in os.listdir(category_path):
        folder_path = os.path.join(category_path, folder)

        if os.path.isdir(folder_path):
            print(f"Processing {folder}...")

            for file in os.listdir(folder_path):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    src = os.path.join(folder_path, file)
                    dst = os.path.join(category_path, file)

                    # handle duplicate names
                    if os.path.exists(dst):
                        base, ext = os.path.splitext(file)
                        dst = os.path.join(category_path, base + "_copy" + ext)

                    shutil.move(src, dst)

            # remove empty folder
            os.rmdir(folder_path)

print("✅ DONE! Dataset flattened.")