import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

try:
    from PIL import Image
except ImportError:
    print("Missing dependency. Run: pip install -r requirements.txt --break-system-packages")


def find_split_dirs(src: Path):
    """Find dataset splits and normalize valid/val naming."""
    mapping = {}
    split_names = [
        ("train", "train"),
        ("valid", "val"),
        ("val", "val"),
        ("test", "test"),
    ]

    for name, canonical in split_names:
        split_dir = src / name
        if split_dir.exists() and (split_dir / "images").exists():
            mapping[canonical] = split_dir

    return mapping


def restructure(src: Path, dst: Path, splits: dict):
    dst.mkdir(parents=True, exist_ok=True)

    for split_name, src_dir in splits.items():
        for folder_name in ["images", "labels"]:
            input_dir = src_dir / folder_name
            output_dir = dst / folder_name / split_name
            output_dir.mkdir(parents=True, exist_ok=True)

            if not input_dir.exists():
                print(f"  warning: {input_dir} missing; skipping")
                continue

            copied = 0
            for file_path in input_dir.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, output_dir / file_path.name)
                    copied += 1

            print(f"  copied {copied:5d} files to {output_dir}")


def load_class_names(src: Path):
    yaml_path = src / "data.yaml"
    if not yaml_path.exists():
        print("  warning: no data.yaml found; using ['ball', 'player']")
        return ["ball", "player"]

    with open(yaml_path) as file:
        data = yaml.safe_load(file)

    return data.get("names", ["ball", "player"])


def write_data_yaml(dst: Path, names: list, out_path: Path):
    content = {
        "path": str(dst.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if (dst / "images" / "test").exists() else "images/val",
        "nc": len(names),
        "names": names,
    }

    with open(out_path, "w") as file:
        yaml.safe_dump(content, file, sort_keys=False)

    print(f"  wrote {out_path}")


def validate_and_report(dst: Path, names: list):
    print("\nValidation report")

    class_counts = Counter()
    image_sizes = Counter()
    corrupt_images = []
    missing_labels = []
    empty_labels = []

    for split_name in ["train", "val", "test"]:
        image_dir = dst / "images" / split_name
        label_dir = dst / "labels" / split_name
        if not image_dir.exists():
            continue

        images = sorted(
            image_path
            for image_path in image_dir.iterdir()
            if image_path.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        print(f"\n{split_name}: {len(images)} images")

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image_sizes[image.size] += 1
            except Exception as error:
                corrupt_images.append((image_path, str(error)))
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(image_path)
                continue

            lines = [line for line in label_path.read_text().splitlines() if line.strip()]
            if not lines:
                empty_labels.append(label_path)
                continue

            for line in lines:
                class_id = int(line.split()[0])
                if 0 <= class_id < len(names):
                    class_counts[names[class_id]] += 1
                else:
                    print(f"  warning: bad class id {class_id} in {label_path}")

    print("\nClass balance")
    total = sum(class_counts.values())
    for name in names:
        count = class_counts.get(name, 0)
        percentage = 100 * count / total if total else 0
        print(f"  {name}: {count:6d}  ({percentage:5.1f}%)")

    if class_counts.get("ball", 0) and class_counts.get("player", 0):
        ratio = class_counts["player"] / max(class_counts["ball"], 1)
        print(f"  player:ball ratio ~ {ratio:.1f}:1", end="")
        if ratio > 15:
            print("  (heavy imbalance; ball recall may be weak on the first run)")
        else:
            print()

    print("\nImage size distribution")
    for size, count in image_sizes.most_common(5):
        print(f"  {size}: {count} images")

    print("\nIntegrity issues")
    print(f"  Corrupt images: {len(corrupt_images)}")
    for path, error in corrupt_images[:5]:
        print(f"    {path.name}: {error}")

    print(f"  Images missing label files: {len(missing_labels)}")
    for path in missing_labels[:5]:
        print(f"    {path.name}")

    print(f"  Empty label files: {len(empty_labels)}")

    if corrupt_images or missing_labels:
        print("\n  Fix these before training. Ultralytics may skip bad files,")
        print("  which makes the usable dataset smaller than expected.")
    else:
        print("\n  Dataset structure looks clean.")

    return class_counts


def run_setup(args) -> Path:
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.exists():
        print(f"ERROR: source path does not exist: {src}")
        sys.exit(1)

    print(f"Source: {src}")
    print(f"Destination: {dst}\n")

    splits = find_split_dirs(src)
    if not splits:
        print("ERROR: could not find train/valid/test folders with an images folder inside src.")
        sys.exit(1)

    print(f"Found splits: {list(splits.keys())}")
    names = load_class_names(src)
    print(f"Classes: {names}\n")

    print("Restructuring dataset...")
    restructure(src, dst, splits)

    out_yaml = dst / "data.yaml"
    write_data_yaml(dst, names, out_yaml)
    validate_and_report(dst, names)

    print(f"\nSetup complete. data.yaml is ready at: {out_yaml}")
    return out_yaml


def run_train(args, data_yaml: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Missing dependency. Run: pip install -r requirements.txt --break-system-packages")
        sys.exit(1)

    print(f"\nLoading base model: {args.model}")
    model = YOLO(args.model)

    print("Starting training...")
    print(f"  data={data_yaml}")
    print(f"  imgsz={args.imgsz}  batch={args.batch}  epochs={args.epochs}  device={args.device}\n")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project="../runs",
        name=args.name,
        mosaic=1.0,
        close_mosaic=10,
        scale=0.5,
        cos_lr=True,
        plots=True,
    )

    best_weights = f"../runs/{args.name}/weights/best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")

    print("\nRunning validation...")
    val_model = YOLO(best_weights)
    metrics = val_model.val(data=str(data_yaml), imgsz=args.imgsz)

    print("\nPer-class results")
    try:
        for index, name in metrics.names.items():
            precision = metrics.box.p[index]
            recall = metrics.box.r[index]
            ap50 = metrics.box.ap50[index]
            print(f"  {name:10s}  precision={precision:.3f}  recall={recall:.3f}  mAP50={ap50:.3f}")
    except Exception:
        print("  per-class breakdown unavailable; check runs/.../results.csv")

    print(f"\nOverall mAP50: {metrics.box.map50:.3f}")
    print(f"Overall mAP50-95: {metrics.box.map:.3f}")

    print("\nNext steps:")
    print(f"  yolo detect predict model={best_weights} source=<video.mp4> imgsz={args.imgsz} conf=0.25 save=True")
    print("  Feed detections into ByteTrack for player and ball tracking.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Path to the extracted Roboflow YOLOv8 export")
    parser.add_argument("--dst", required=True, help="Destination inside SportsStrategyCoachAI/datasets/")
    parser.add_argument("--skip-setup", action="store_true", help="Skip dataset setup")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default=0)
    parser.add_argument("--name", default="player_ball_v1")
    args = parser.parse_args()

    dst = Path(args.dst).expanduser().resolve()

    if not args.skip_setup:
        if not args.src:
            print("ERROR: --src is required unless --skip-setup is set")
            sys.exit(1)

        data_yaml = run_setup(args)
    else:
        data_yaml = dst / "data.yaml"
        if not data_yaml.exists():
            print(f"ERROR: --skip-setup was set but {data_yaml} does not exist. Run setup first.")
            sys.exit(1)

        print(f"Skipping setup. Using existing: {data_yaml}")

    if not args.skip_train:
        run_train(args, data_yaml)
    else:
        print("Skipping training.")


if __name__ == "__main__":
    main()
