import re
from pathlib import Path

# Input and output paths (edit if needed)
input_file = Path("/host_data/van/LDA/data/edge_aistpp/feat/crossmodal_test.txt")
styles_all_file = Path("/host_data/van/LDA/data/edge_aistpp/feat/styles_all.txt")
styles_unique_file = Path("/host_data/van/LDA/data/edge_aistpp/feat/styles_unique.txt")

# Regex: match style like gWA, gLH, gBR, etc.
STYLE_RE = re.compile(r'(?:(?<=^)|(?<=_))(g[A-Za-z]{2})(?=_)')

def main():
    lines = input_file.read_text(encoding="utf-8").splitlines()

    styles_all = []
    for raw in lines:
        name = raw.strip()
        if not name:
            continue
        match = STYLE_RE.search(name)
        if match:
            styles_all.append(match.group(1))

    # Save all styles (with duplicates, in order)
    styles_all_file.write_text("\n".join(styles_all) + "\n", encoding="utf-8")

    # Save unique styles (sorted)
    unique_sorted = sorted(set(styles_all))
    styles_unique_file.write_text("\n".join(unique_sorted) + "\n", encoding="utf-8")

    print(f"✅ Extracted {len(styles_all)} styles to {styles_all_file}")
    print(f"✅ Found {len(unique_sorted)} unique styles → {styles_unique_file}")

if __name__ == "__main__":
    main()
