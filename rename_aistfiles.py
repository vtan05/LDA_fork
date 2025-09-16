import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.startswith("aist_"):
            new_name = filename.replace("aist_", "", 1)  # remove only the first "aist_"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    folder_path = "/host_data/van/LDA/results/aist_8s"  # change this to your target folder
    rename_files(folder_path)
