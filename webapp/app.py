from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os, uuid
from src.inference.run_tracking import process_video

app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get('FLASK_SECRET', 'devsecret')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('outputs/video', exist_ok=True)

@app.route('/')
def index():
    videos = []
    for fn in os.listdir('outputs/video'):
        if fn.lower().endswith('.mp4'):
            videos.append(fn)
    return render_template('index.html', videos=videos)

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    f = request.files['video']
    if f.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    filename = f"{uuid.uuid4().hex}_{f.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)
    try:
        out = process_video(path, weights_path=os.environ.get('YOLO_WEIGHTS', 'models/yolov8_custom_detector/best.pt'))
        flash(f'Processing complete: {out}')
    except Exception as e:
        flash(f'Error during processing: {e}')
    return redirect(url_for('index'))

@app.route('/outputs/video/<path:fname>')
def outputs_video(fname):
    return send_from_directory('outputs/video', fname)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
