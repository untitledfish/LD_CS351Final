class RobotTrack:
    def __init__(self, track_id):
        self.track_id = track_id
        self.last_bbox = None
        self.history = []

    def update(self, frame_idx, bbox):
        x1,y1,x2,y2 = bbox
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        self.history.append((frame_idx,(cx,cy)))
        self.last_bbox = bbox

class TrackerManager:
    def __init__(self):
        self.tracks = {}

    def ensure_track(self, track_id):
        if track_id not in self.tracks:
            self.tracks[track_id] = RobotTrack(track_id)
        return self.tracks[track_id]

    def update_from_detections(self, frame_idx, detections):
        for d in detections:
            if 'track_id' not in d:
                continue
            t = self.ensure_track(d['track_id'])
            t.update(frame_idx, d['xyxy'])
