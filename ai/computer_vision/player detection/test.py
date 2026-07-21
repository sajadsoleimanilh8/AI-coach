from ultralytics import YOLO
import json

model = YOLO("best.pt")

results = model.predict(
    source="19.mp4",
    stream=True,
    save=False
)

output = []

for frame_id, r in enumerate(results):

    frame_info = {
        "frame": frame_id,
        "detections": []
    }

    for box in r.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()

        frame_info["detections"].append({
            "class": model.names[cls_id],
            "confidence": round(conf, 3),
            "bbox": coords
        })

    output.append(frame_info)

with open("detections.json", "w") as f:
    json.dump(output, f, indent=2)

print("detections.json created")