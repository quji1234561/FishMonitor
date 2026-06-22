from pathlib import Path
import shutil
import zipfile

# 改成你的真实路径
ORIGINAL = Path(r"output/fish_detection/PF-001_yolo_fish_step5")
ENHANCED = Path(r"output/fish_detection/PF-001_yolo_fish_step5_all_enhanced")
OUT = Path(r"output/fish_detection/PF-001_yolo_fish_step5_mixed")

def copy_files(src_dir, dst_dir, prefix=""):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.iterdir():
        if src.is_file():
            dst = dst_dir / f"{prefix}{src.name}"
            shutil.copy2(src, dst)

if OUT.exists():
    shutil.rmtree(OUT)

# train: 原图 + 增强图
copy_files(ORIGINAL / "images" / "train", OUT / "images" / "train", prefix="orig_")
copy_files(ORIGINAL / "labels" / "train", OUT / "labels" / "train", prefix="orig_")

copy_files(ENHANCED / "images" / "train", OUT / "images" / "train", prefix="enh_")
copy_files(ENHANCED / "labels" / "train", OUT / "labels" / "train", prefix="enh_")

# val/test: 只用原图
for split in ["val", "test"]:
    copy_files(ORIGINAL / "images" / split, OUT / "images" / split)
    copy_files(ORIGINAL / "labels" / split, OUT / "labels" / split)

# 写 data.yaml
(OUT / "data.yaml").write_text(
f"""path: {OUT}
train: images/train
val: images/val
test: images/test

names:
  0: fish
""",
encoding="utf-8"
)

# 打 zip
zip_path = OUT.with_suffix(".zip")
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in OUT.rglob("*"):
        if file.is_file():
            zf.write(file, arcname=str(file.relative_to(OUT.parent)))

print("Done:", OUT)
print("Zip:", zip_path)

for split in ["train", "val", "test"]:
    print(
        split,
        "images:", len(list((OUT / "images" / split).glob("*"))),
        "labels:", len(list((OUT / "labels" / split).glob("*"))),
    )