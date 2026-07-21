from pathlib import Path
import shutil


DATASETS = [
    r"D:\SportsStrategyCoachAI\SportsStrategyCoachAI\datasets\ds1",
    r"D:\SportsStrategyCoachAI\SportsStrategyCoachAI\datasets\ds2",
    r"D:\SportsStrategyCoachAI\SportsStrategyCoachAI\datasets\ds3",
    r"D:\SportsStrategyCoachAI\SportsStrategyCoachAI\datasets\ds4",
]

OUTPUT = Path(
    r"D:\SportsStrategyCoachAI\SportsStrategyCoachAI\datasets\merged_dataset"
)

# ساخت پوشه‌ها
for split in ["train", "valid", "test"]:
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

counter = 0

for dataset in DATASETS:
    dataset = Path(dataset)

    print(f"Processing: {dataset.name}")

    for split in ["train", "valid", "test"]:
        img_dir = dataset / split / "images"
        lbl_dir = dataset / split / "labels"

        if not img_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            ]:
                continue

            stem = f"{dataset.name}_{counter}"
            counter += 1

            new_img = OUTPUT / split / "images" / f"{stem}{img_file.suffix}"
            shutil.copy2(img_file, new_img)

            label_file = lbl_dir / f"{img_file.stem}.txt"

            if label_file.exists():
                new_lbl = OUTPUT / split / "labels" / f"{stem}.txt"
                shutil.copy2(label_file, new_lbl)

print("Dataset merge completed!")

yaml_text = """
train: train/images
val: valid/images
test: test/images

nc: 4

names:
  0: ball
  1: goalkeeper
  2: player
  3: referee
"""

with open(OUTPUT / "data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_text)

print("data.yaml created!")