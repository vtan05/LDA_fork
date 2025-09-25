import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.startswith("ybot__"):
            new_name = filename.replace("ybot__", "", 1)  # remove only the first "ybot__"
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    folder_path = "/host_data/van/LDA/data/edge_aistpp_v1/ybot_bvh"  # change this to your target folder
    rename_files(folder_path)
