import os
import glob

# Folder containing the pkl files
folder = r"/host_data/van/LDA/data/finedance/feat"

# Find all matching .pkl files
for filepath in glob.glob(os.path.join(folder, "*.pkl")):
    dirname, filename = os.path.split(filepath)
    if filename.startswith("aist_"):
        new_name = filename.replace("aist_", "", 1)  # remove only the first occurrence
        new_path = os.path.join(dirname, new_name)
        os.rename(filepath, new_path)
        print(f"Renamed: {filename} -> {new_name}")
