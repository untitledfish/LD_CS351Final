import os, cv2, time
from ultralytics import YOLO
from src.tracking.tracker_utils import TrackerManager
from src.scoring.scoring_engine import ScoringEngine
from src.db.db import DB
from src.utils.draw import draw_boxes, draw_scoreboard

def process_video(video_path, weights_path='models/yolov8_custom_detector/best.pt', out_path=None, tracker='bytetrack', imgsz=640, show=False):
    if out_path is None:
        os.makedirs('outputs/video', exist_ok=True)
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = f'outputs/video/{name}_annotated.mp4'

    model = YOLO(weights_path)
    db = DB()
    scoring = ScoringEngine('config/scoring_rules.json')
    tracker_mgr = TrackerManager()

    results = model.track(source=video_path, show=False, stream=True, tracker=tracker, imgsz=imgsz)

    writer = None
    frame_idx = 0
    for result in results:
        frame = result.orig_img
        detections = []
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                try:
                    xyxy = box.xyxy.cpu().numpy().tolist()[0]
                except Exception:
                    xyxy = box.xyxy.tolist()[0]
                try:
                    conf = float(box.conf.cpu().numpy()[0])
                except Exception:
                    conf = float(box.conf.tolist()[0])
                try:
                    cls = int(box.cls.cpu().numpy()[0])
                except Exception:
                    cls = int(box.cls.tolist()[0])
                name = model.names.get(cls, str(cls))
                track_id = None
                if hasattr(box, 'id') and box.id is not None:
                    try:
                        track_id = int(box.id.cpu().numpy()[0])
                    except Exception:
                        track_id = int(box.id)
                det = {'xyxy': xyxy, 'conf': conf, 'class_name': name}
                if track_id is not None:
                    det['track_id'] = track_id
                detections.append(det)

        tracker_mgr.update_from_detections(frame_idx, detections)

        per_track = {}
        for d in detections:
            tid = d.get('track_id', None)
            if tid is None:
                continue
            cx = int((d['xyxy'][0]+d['xyxy'][2])/2)
            cy = int((d['xyxy'][1]+d['xyxy'][3])/2)
            per_track.setdefault(tid, []).append({'class_name': d['class_name'], 'center': (cx,cy)})

        for tid, dets in per_track.items():
            evs = scoring.evaluate(tid, dets, frame_idx)
            for e in evs:
                db.add_or_update_robot(tid)
                db.add_event(tid, e['rule'], e['points'], frame_idx)

        draw_boxes(frame, detections)
        scoreboard = db.get_scoreboard()
        draw_scoreboard(frame, scoreboard)

        if writer is None:
            h,w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

        writer.write(frame)
        if show:
            cv2.imshow('annot', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_idx += 1

    if writer:
        writer.release()
    return out_path
