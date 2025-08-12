import os

# === Configuration ===
target_dir = "/host_data/van/LDA/data/finedance/feat"  # change to your folder path
pattern = "expmap_24fps"

# === Script ===
removed_count = 0
for filename in os.listdir(target_dir):
    if filename.endswith(".pkl") and pattern in filename:
        file_path = os.path.join(target_dir, filename)
        try:
            os.remove(file_path)
            removed_count += 1
            print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Failed to remove {file_path}: {e}")

print(f"\nTotal .pkl files removed: {removed_count}")
