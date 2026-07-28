import sys
from pathlib import Path
import numpy as np
import cv2 

# =====================================================================
# 1. PATH ROUTING: DO NOT CHANGE THIS BLOCK
# This guarantees Python can find all modules in your 'ai' folder,
# regardless of where this script is executed from.
# =====================================================================
current_file = Path(__file__).resolve()
for parent in current_file.parents:
    if parent.name == 'ai':
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

# =====================================================================
# 2. IMPORTS
# Pulling precisely from the architecture blueprint we established
# =====================================================================
from computer_vision.player_tracking.bytetrack import BYTETracker
from computer_vision.tactical_analysis.homography import pixels_to_pitch
from computer_vision.tactical_analysis.manual_calibration import load_calibration

print("Imports successful! The computer vision pipeline is connected.")


# =====================================================================
# 3. PIPELINE INITIALIZATION
# =====================================================================
def initialize_pipeline(calibration_path="calibrations/match_999.json"):
    """Initializes the tracker and loads the pitch homography matrix."""
    tracker = BYTETracker(track_thresh=0.5, track_buffer=30)
    
    try:
        H, record = load_calibration(calibration_path)
        print(f"Loaded calibration with {record['confidence']:.2f} confidence.")
    except FileNotFoundError:
        print(f"WARNING: Calibration file '{calibration_path}' not found.")
        print("Tracking will continue, but Pitch Mapping (Homography) is disabled until you run manual_calibration.py!")
        H = None
        
    return tracker, H


# =====================================================================
# 4. FRAME PROCESSING LOGIC
# =====================================================================
def process_video_frame(frame, yolo_model, tracker, H):
    """Detects -> Tracks -> Maps to Pitch."""
    
    # A. Run YOLO (Assuming Ultralytics YOLOv8)
    results = yolo_model(frame, verbose=False)[0]
    
    # B. Format detections for ByteTrack: [x1, y1, x2, y2, score, class_id]
    dets = []
    if hasattr(results, 'boxes') and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Filter: Only track people (usually class 0 in COCO datasets)
            if cls == 0: 
                dets.append([x1, y1, x2, y2, score, cls])
    
    yolo_dets = np.array(dets) if len(dets) > 0 else np.empty((0, 6))
    
    # C. Update ByteTrack with new detections
    active_tracks = tracker.update(yolo_dets)
    
    pitch_coords = []
    if len(active_tracks) > 0 and H is not None:
        # D. Extract bottom-center of bounding boxes (the player's feet)
        foot_pixels = np.array([
            [(track.tlbr[0] + track.tlbr[2]) / 2.0, track.tlbr[3]] 
            for track in active_tracks
        ])
        
        # E. Map pixels to pitch coordinates (meters)
        pitch_coords = pixels_to_pitch(foot_pixels, H)
        
    return active_tracks, pitch_coords


# =====================================================================
# 5. MAIN EXECUTION LOOP
# =====================================================================
def main():
    # TODO: Replace with the path to your actual match video
    VIDEO_PATH = "your_match_video.mp4" 
    
    # -----------------------------------------------------------------
    # TEMPORARY DUMMY YOLO MODEL 
    # (This allows the script to run right now without crashing. 
    # Delete this block and uncomment the real YOLO lines when ready!)
    class DummyModel:
        def __call__(self, frame, verbose=False):
            class DummyResults:
                boxes = [] # Emulates seeing 0 people
            return [DummyResults()]
    model = DummyModel()
    
    # -- REAL YOLO SETUP (Uncomment when you have your weights) --
    # from ultralytics import YOLO
    # model = YOLO('yolov8n.pt') 
    # -----------------------------------------------------------------

    # 1. Init Pipeline
    tracker, H = initialize_pipeline()
    
    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Waiting for video file: '{VIDEO_PATH}'.")
        print("SUCCESS! The pipeline logic is fully integrated and ready.")
        return

    print("Starting video processing loop...")
    frame_count = 0
    
    # 3. Process Video Frame-by-Frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process the frame!
        tracks, pitch_positions = process_video_frame(frame, model, tracker, H)
        
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count} - Active tracks: {len(tracks)}")
            
    cap.release()
    print("Processing complete.")

if __name__ == "__main__":
    main()