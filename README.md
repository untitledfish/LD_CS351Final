FRCScorekeeper

This repository contains a working YOLOv8 inference pipeline (using Ultralytics), a scoring engine, and a simple Flask web dashboard that lets you upload a match video, 
runs inference to detect & track robots, and produces an annotated output video and a scoreboard.

Requirements
- Python 3.11
- pip install -r requirements.txt

Quickstart
1. Create and activate venv (Python 3.11):
2. python3.11 -m venv venv
3. source venv/bin/activate
4. python -m pip install --upgrade pip
5. pip install -r requirements.txt
6. Start the web dashboard:
7. export FLASK_APP=webapp.app
8. flask run --host=0.0.0.0 --port=5000
9. Then open http://localhost:5000
10. Upload a match video and press 'Process'. The server will run inference (synchronously) and produce an annotated video in outputs/video/ and store events in outputs/frc_events.db.