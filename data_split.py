import os
import random
from collections import defaultdict

def get_style_prefix(filename):
    return filename.split('_')[1]  # e.g., 'gPO'

if __name__ == '__main__':
    # Paths
    directory = r'/host_data/van/LDA/data/motorica/feat'
    out_directory = r'/host_data/van/LDA/data/motorica/feat'

    # Load files
    pkl_files = [f for f in os.listdir(directory) if f.endswith('.expmap_30fps.pkl')]
    original_files = [f for f in pkl_files if not f.endswith('_mirrored.expmap_30fps.pkl')]
    mirrored_files = [f for f in pkl_files if f.endswith('_mirrored.expmap_30fps.pkl')]

    # Remove extensions
    original_no_ext = [f.replace('.expmap_30fps.pkl', '') for f in original_files]
    mirrored_no_ext = [f.replace('.expmap_30fps.pkl', '') for f in mirrored_files]

    # Fixed target styles
    selected_styles = ['gCH', 'gJZ', 'gKR', 'gLH', 'gLO', 'gPO', 'gTP']

    # Forced test files (manually specified)
    forced_test_raw = [
        'kthjazz_gCH_sFM_cAll_d02_mCH_ch01_whitemanpaulandhisorchestraloisiana_006',
        'kthjazz_gJZ_sFM_cAll_d02_mJZ_ch01_bennygoodmansugarfootstomp_003',
        'kthjazz_gTP_sFM_sngl_d02_015',
        'kthstreet_gKR_sFM_cAll_d01_mKR_ch01_chargedcableupyour_001',
        'kthstreet_gLH_sFM_cAll_d01_mLH_ch01_thisisit_001',
        'kthstreet_gLH_sFM_cAll_d02_mLH_ch01_lala_001',
        'kthstreet_gLO_sFM_cAll_d02_mLO_ch01_arethafranklinrocksteady_002',
        'kthstreet_gPO_sFM_cAll_d01_mPO_ch01_bombom_002',
        'kthstreet_gPO_sFM_cAll_d02_mPO_ch01_bombom_001',
    ]

    # Group all original files by style
    style_to_files = defaultdict(list)
    for f in original_no_ext:
        style = get_style_prefix(f)
        style_to_files[style].append(f)

    # Group forced files by style
    forced_test_by_style = defaultdict(list)
    for f in forced_test_raw:
        style = get_style_prefix(f)
        forced_test_by_style[style].append(f)

    # Build final test set with exactly 2 files per style
    final_test_set = set()
    returned_to_train = []

    for style in selected_styles:
        pool = style_to_files.get(style, [])
        forced = forced_test_by_style.get(style, [])
        selected = []

        # Use up to 2 forced test files
        selected += forced[:2]
        returned_to_train += forced[2:]
        needed = max(0, 2 - len(selected))

        # Fill with additional files if needed
        remaining = [f for f in pool if f not in selected]
        if needed > 0 and len(remaining) >= needed:
            selected += random.sample(remaining, needed)

        final_test_set.update(selected)

    # Remaining go to train
    train_set = set(original_no_ext) - final_test_set
    train_set.update(returned_to_train)

    # Add mirrored files to train if not related to test
    mirrored_train = [f for f in mirrored_no_ext if f.replace('_mirrored', '') not in final_test_set]
    train_full = sorted(list(train_set) + mirrored_train)

    # Remove any mirrored version of test files
    train_full = [
        f for f in train_full
        if f not in final_test_set and f.replace('_mirrored', '') not in final_test_set
    ]

    # Save to file
    with open(os.path.join(out_directory, 'train_files.txt'), 'w') as f:
        for name in sorted(train_full):
            f.write(f"{name}\n")

    with open(os.path.join(out_directory, 'test_files.txt'), 'w') as f:
        for name in sorted(final_test_set):
            f.write(f"{name}\n")

    # Logging
    print(f"Total .pkl files: {len(pkl_files)}")
    print(f"Original (no ext): {len(original_no_ext)}")
    print(f"Train files (with mirrored): {len(train_full)}")
    print(f"Test files: {len(final_test_set)} (2 per style × {len(selected_styles)} styles)")
    for style in selected_styles:
        count = sum(1 for f in final_test_set if get_style_prefix(f) == style)
        print(f"  {style}: {count}")
