#NO USE
import os
import shutil

base_path = "dataset/Train"

fresh_path = os.path.join(base_path, "fresh")
rotten_path = os.path.join(base_path, "rotten")

os.makedirs(fresh_path, exist_ok=True)
os.makedirs(rotten_path, exist_ok=True)

print("Folders inside Train:", os.listdir(base_path))

moved_count = 0

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)

    if not os.path.isdir(folder_path):
        continue

    if folder.lower() in ["fresh", "rotten"]:
        continue

    print(f"\nProcessing: {folder}")

    for file in os.listdir(folder_path):
        if file.endswith((".png", ".jpg", ".jpeg")):
            src = os.path.join(folder_path, file)

            if folder.lower().startswith("fresh"):
                dst = os.path.join(fresh_path, file)
            elif folder.lower().startswith("rotten"):
                dst = os.path.join(rotten_path, file)
            else:
                continue

            print(f"Moving: {file} → {dst}")
            shutil.move(src, dst)
            moved_count += 1

    # remove empty folder
    try:
        os.rmdir(folder_path)
    except:
        pass

print(f"\n✅ DONE! Total files moved: {moved_count}")