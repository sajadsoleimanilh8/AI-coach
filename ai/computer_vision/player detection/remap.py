mapping = {
    0: 2,  # FCB -> player
    1: 2,  # RMA -> player
    2: 0,  # ball -> ball
    3: 1,  # goalkeeper -> goalkeeper
    4: 3   # referee -> referee
}

dataset = Path(r"D:\SportsStrategyCoachAI\SportsStrategyCoachAI\datasets\ds3")

for split in ["train", "valid", "test"]:
    label_dir = dataset / split / "labels"

    for txt_file in label_dir.glob("*.txt"):
        output = []

        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                old_cls = int(parts[0])
                parts[0] = str(mapping[old_cls])

                output.append(" ".join(parts))

        with open(txt_file, "w") as f:
            f.write("\n".join(output))

print("Done!")