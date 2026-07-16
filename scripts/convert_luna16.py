"""Convert LUNA16 .mhd/.zraw volumes to 2D PNGs and copy into COCO train folder.

Usage example:
python scripts/convert_luna16.py \
    --input-dir "seg-lungs-LUNA16-20260715T002444Z-1-001/seg-lungs-LUNA16" \
    --output-dir "lung_ct_version_n_512.v2i.coco/train" \
    --max-slices 5 --limit-volumes 50
"""
import argparse
import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk


def normalize_to_uint8(img):
    # robust min/max
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-8)
    img = (img * 255.0).astype(np.uint8)
    return img


def process_volume(mhd_path, out_dir, max_slices=5):
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)  # shape: [z, y, x]
    z = arr.shape[0]
    if z == 0:
        return 0
    # choose slice indices evenly
    if z <= max_slices:
        indices = list(range(z))
    else:
        indices = np.linspace(0, z - 1, max_slices, dtype=int).tolist()

    base = Path(mhd_path).stem
    saved = 0
    for i in indices:
        slice_img = arr[i]
        slice_img = normalize_to_uint8(slice_img)
        pil = Image.fromarray(slice_img)
        # convert to RGB
        pil = pil.convert("RGB")
        out_name = f"{base}_s{i}.png"
        out_path = out_dir / out_name
        # avoid overwrite
        if out_path.exists():
            out_path = out_dir / (f"{base}_s{i}_{uuid.uuid4().hex[:8]}.png")
        pil.save(out_path)
        saved += 1
    return saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-slices", type=int, default=5)
    p.add_argument("--limit-volumes", type=int, default=None, help="Max number of volumes to process")
    args = p.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input dir {input_dir} not found")
    output_dir.mkdir(parents=True, exist_ok=True)

    mhd_files = sorted(input_dir.glob("*.mhd"))
    if args.limit_volumes:
        mhd_files = mhd_files[: args.limit_volumes]

    total_saved = 0
    for idx, mhd in enumerate(mhd_files, 1):
        try:
            saved = process_volume(mhd, output_dir, max_slices=args.max_slices)
            total_saved += saved
        except Exception as e:
            print(f"Skipping {mhd.name}: {e}")
    print(f"Processed {len(mhd_files)} volumes, saved {total_saved} PNGs to {output_dir}")


if __name__ == "__main__":
    main()
