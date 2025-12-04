import json, time

class ScoringEngine:
    def __init__(self, rules_path="config/scoring_rules.json"):
        with open(rules_path, 'r') as f:
            cfg = json.load(f)
        self.class_map = cfg.get('classes', {})
        self.rules = cfg.get('rules', [])
        self.zones = cfg.get('zones', {})
        self.buffers = {}

    def _point_in_zone(self, center, zone_coords):
        x,y = center
        (x1,y1),(x2,y2) = zone_coords
        return x1 <= x <= x2 and y1 <= y <= y2

    def evaluate(self, track_id: int, detections: list, frame_index: int):
        events = []
        for rule in self.rules:
            name = rule['name']
            trigger = rule['class_trigger']
            frames_confirm = rule.get('frames_confirm', 1)
            cooldown = rule.get('cooldown_s', 0)
            zone_required = rule.get('zone_required', None)

            matched = False
            for det in detections:
                if det['class_name'] == trigger:
                    if zone_required:
                        zone = self.zones.get(zone_required)
                        if zone is None:
                            continue
                        if not self._point_in_zone(det['center'], zone):
                            continue
                    matched = True
                    break

            key = (track_id, name)
            if matched:
                b = self.buffers.get(key, {'count':0, 'last_frame':-999, 'last_award_ts':0})
                if frame_index - b['last_frame'] <= 2:
                    b['count'] += 1
                else:
                    b['count'] = 1
                b['last_frame'] = frame_index
                if b['count'] >= frames_confirm:
                    now = time.time()
                    if now - b.get('last_award_ts', 0) >= cooldown:
                        events.append({'rule': name, 'points': rule['score'], 'track_id': track_id})
                        b['last_award_ts'] = now
                        b['count'] = 0
                self.buffers[key] = b
            else:
                if key in self.buffers:
                    self.buffers[key]['count'] = 0
        return events
