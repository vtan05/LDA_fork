import os
import glob
import re

# === Configuration ===
data_dir = "/host_data/van/LDA/data/finedance/feat/"  # Change this to your actual data directory
output_dir = "/host_data/van/LDA/data/finedance/feat/"                # Output directory for the split lists

# Generate all sample IDs from 001 to 211
all_list = [str(i).zfill(3) for i in range(1, 212)]

# Predefined test and ignore lists
test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193",
             "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]

# Filter out ignored and test samples
valid_list = [x for x in all_list if x not in ignor_list]
train_list = [x for x in valid_list if x not in test_list]

# Regex to extract the 3-digit ID before `.expmap_30fps.pkl`
pattern = re.compile(r"_(\d{3})\.expmap_30fps\.pkl$")

# Helper to strip full extension
def strip_suffix(filename):
    return filename.replace(".expmap_30fps.pkl", "")

# Index files by ID
all_files = glob.glob(os.path.join(data_dir, "*.expmap_30fps.pkl"))
id_to_file = {}

for f in all_files:
    base = os.path.basename(f)
    match = pattern.search(base)
    if match:
        idx = match.group(1)
        id_to_file[idx] = base
    else:
        print(f"[Warning] Skipped file (no match): {base}")

# Build split file lists (without extension)
train_names = [strip_suffix(id_to_file[x]) for x in train_list if x in id_to_file]
test_names = [strip_suffix(id_to_file[x]) for x in test_list if x in id_to_file]

# Save to txt
with open(os.path.join(output_dir, "train_list.txt"), "w") as f:
    for name in sorted(train_names):
        f.write(f"{name}\n")

with open(os.path.join(output_dir, "test_list.txt"), "w") as f:
    for name in sorted(test_names):
        f.write(f"{name}\n")

print(f"✅ Train samples written: {len(train_names)}")
print(f"✅ Test samples written:  {len(test_names)}")