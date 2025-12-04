import os
import cv2
from ultralytics import YOLO
from src.tracking.tracker_utils import TrackerManager
from src.scoring.scoring_engine import ScoringEngine
from src.db.db import DB
from src.utils.draw import draw_boxes, draw_scoreboard

def process_video(
    video_path,
    weights_path='models/yolov8_custom_detector/best.pt',
    out_path=None,
    tracker_yaml="/Users/bobblehat/venv/lib/python3.11/site-packages/ultralytics/cfg/trackers/bytetrack.yaml",
    show=False
):
    # Determine output path
    if out_path is None:
        os.makedirs('outputs/video', exist_ok=True)
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = f'outputs/videos/{name}_annotated.mp4'

    # Load YOLOv8 model
    print(f"Loading YOLOv8 model from: {weights_path}")
    model = YOLO(weights_path)

    # Initialize DB, Scoring, Tracker
    db = DB()
    scoring = ScoringEngine('config/scoring_rules.json')
    tracker_mgr = TrackerManager()

    # Capture original video properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video properties: {input_w}x{input_h} @ {input_fps:.2f} FPS")

    # Initialize VideoWriter
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    writer = cv2.VideoWriter(out_path, fourcc, input_fps, (input_w, input_h))

    # Run YOLOv8 tracking in stream mode
    results = model.track(source=video_path, stream=True, tracker=tracker_yaml)

    for frame_idx, result in enumerate(results):
        frame = result.orig_img  # YOLOv8 frame

        # Ensure BGR 3-channel
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        detections = []
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy().tolist()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy.tolist()[0]
                conf = float(box.conf.cpu().numpy()[0]) if hasattr(box.conf, 'cpu') else float(box.conf.tolist()[0])
                cls = int(box.cls.cpu().numpy()[0]) if hasattr(box.cls, 'cpu') else int(box.cls.tolist()[0])
                name = model.names.get(cls, str(cls))
                track_id = None
                if hasattr(box, 'id') and box.id is not None:
                    track_id = int(box.id.cpu().numpy()[0]) if hasattr(box.id, 'cpu') else int(box.id)
                det = {'xyxy': xyxy, 'conf': conf, 'class_name': name}
                if track_id is not None:
                    det['track_id'] = track_id
                detections.append(det)

        # Update tracker
        tracker_mgr.update_from_detections(frame_idx, detections)

        # Evaluate scoring
        per_track = {}
        for d in detections:
            tid = d.get('track_id', None)
            if tid is None:
                continue
            cx = int((d['xyxy'][0] + d['xyxy'][2]) / 2)
            cy = int((d['xyxy'][1] + d['xyxy'][3]) / 2)
            per_track.setdefault(tid, []).append({'class_name': d['class_name'], 'center': (cx, cy)})

        for tid, dets in per_track.items():
            events = scoring.evaluate(tid, dets, frame_idx)
            for e in events:
                db.add_or_update_robot(tid)
                db.add_event(tid, e['rule'], e['points'], frame_idx)

        # Draw detections and scoreboard
        draw_boxes(frame, detections)
        draw_scoreboard(frame, db.get_scoreboard())

        # Write frame to video
        writer.write(frame)

        # Optional: display live
        if show:
            cv2.imshow('Annotated', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    writer.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--weights", type=str, default='models/yolov8_custom_detector/best.pt', help="YOLOv8 weights")
    parser.add_argument("--show", action='store_true', help="Display video live while processing")
    args = parser.parse_args()

    process_video(video_path=args.video, weights_path=args.weights, show=args.show)
