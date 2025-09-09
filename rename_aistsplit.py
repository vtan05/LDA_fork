input_file = "/host_data/van/LDA/data/edge_aistpp/feat/crossmodal_train.txt"
output_file = "/host_data/van/LDA/data/edge_aistpp/feat/crossmodal_train.txt"  # change to input_file if you want to overwrite

with open(input_file, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    name = line.strip()
    if not name.startswith("aist_"):
        name = "aist_" + name
    new_lines.append(name)

with open(output_file, "w") as f:
    f.write("\n".join(new_lines))

print(f"Updated filenames written to {output_file}")
