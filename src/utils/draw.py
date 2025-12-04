import cv2

def draw_boxes(frame, dets):
    for d in dets:
        x1,y1,x2,y2 = map(int, d['xyxy'])
        label = f"{d['class_name']} {d.get('conf',0):.2f}"
        if 'track_id' in d:
            label = f"ID:{d['track_id']} {label}"
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

def draw_scoreboard(frame, scoreboard, x=10, y=20):
    for i,(tid, team, score) in enumerate(scoreboard[:8]):
        text = f"{i+1}. ID:{tid} {team or ''} S:{score}"
        cv2.putText(frame, text, (x, y + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(frame, text, (x, y + i*24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
