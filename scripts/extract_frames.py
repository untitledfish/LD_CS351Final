import cv2, os
os.makedirs('dataset/images', exist_ok=True)
cap = cv2.VideoCapture('assets/sample_videos/match1.mp4')
idx = 0
saved = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % 5 == 0:
        p = f'dataset/images/frame_{idx:06d}.jpg'
        cv2.imwrite(p, frame)
        saved += 1
    idx += 1
cap.release()
print('Saved', saved, 'frames')