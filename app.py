from pathlib import Path
import base64, io, os, threading, uuid, subprocess
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'; OUT.mkdir(exist_ok=True)
MODEL_PATH = ROOT / 'models' / 'best.pt'
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 120 * 1024 * 1024
_model = None
_lock = threading.RLock()

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError('Model weights missing. Run: python download_model.py')
        with _lock:
            if _model is None:
                _model = YOLO(str(MODEL_PATH))
    return _model

def detect(frame):
    with _lock:
        result = get_model().predict(source=frame, conf=0.25, imgsz=640, verbose=False, device='cpu')[0]
    labels=[]
    for b in result.boxes:
        idx=int(b.cls.item()); conf=float(b.conf.item())
        labels.append({'name':str(result.names[idx]), 'confidence':round(conf*100,1), 'box':[round(float(v)) for v in b.xyxy[0].tolist()]})
    return result.plot(), labels

def save_jpg(frame):
    name=uuid.uuid4().hex+'.jpg'
    cv2.imwrite(str(OUT/name),frame)
    return '/outputs/'+name

@app.route('/')
def home():
    return render_template('index.html', ready=MODEL_PATH.exists())

@app.route('/api/status')
def status():
    return jsonify(ready=MODEL_PATH.exists(), model='YOLOv8 traffic-sign detector', source='ankitjha07/Traffic-Sign-Detection-YOLOv8')

@app.route('/api/image', methods=['POST'])
def image():
    try:
        f=request.files.get('file')
        if not f: return jsonify(error='Select an image'),400
        arr=np.frombuffer(f.read(),dtype=np.uint8)
        frame=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        if frame is None:return jsonify(error='Invalid image'),400
        plotted,labels=detect(frame)
        return jsonify(image=save_jpg(plotted), detections=labels)
    except Exception as e:return jsonify(error=str(e)),500

@app.route('/api/frame', methods=['POST'])
def frame():
    try:
        raw=request.get_json(force=True).get('image','').split(',')[-1]
        arr=np.frombuffer(base64.b64decode(raw),dtype=np.uint8)
        img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        if img is None:return jsonify(error='Invalid camera frame'),400
        plotted,labels=detect(img)
        ok,jpg=cv2.imencode('.jpg',plotted,[cv2.IMWRITE_JPEG_QUALITY,75])
        return jsonify(image='data:image/jpeg;base64,'+base64.b64encode(jpg).decode(),detections=labels)
    except Exception as e:return jsonify(error=str(e)),500

@app.route('/api/video',methods=['POST'])
def video():
    try:
        f=request.files.get('file')
        if not f:return jsonify(error='Select a video'),400
        ext=Path(secure_filename(f.filename)).suffix.lower()
        if ext not in {'.mp4','.avi','.mov','.mkv','.webm'}:return jsonify(error='Unsupported video format'),400
        src=OUT/(uuid.uuid4().hex+ext); f.save(src)
        cap=cv2.VideoCapture(str(src))
        if not cap.isOpened():return jsonify(error='Cannot decode video'),400
        fps=cap.get(cv2.CAP_PROP_FPS) or 20
        fps=min(max(fps,1),60)
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not w or not h:cap.release();return jsonify(error='Invalid video dimensions'),400
        scale=min(1,960/max(w,h));w=int(w*scale)//2*2;h=int(h*scale)//2*2
        name=uuid.uuid4().hex+'.mp4'; target=OUT/name
        writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not writer.isOpened():cap.release();return jsonify(error='Video encoder unavailable'),500
        counts={};frames=0
        while frames<1800:
            ok,img=cap.read()
            if not ok:break
            img=cv2.resize(img,(w,h))
            plotted,labels=detect(img)
            writer.write(plotted);frames+=1
            for item in labels:counts[item['name']]=counts.get(item['name'],0)+1
        cap.release();writer.release();src.unlink(missing_ok=True)
        if not frames:
            return jsonify(error='No frames found'), 400
        converted = OUT / (uuid.uuid4().hex + "_h264.mp4")
        command = [
            "ffmpeg", "-y",
            "-i", str(target),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(converted)
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg conversion failed: " + result.stderr[-1000:]
            )
        target.unlink(missing_ok=True)
        name = converted.name
       
        return jsonify(video='/outputs/'+name, counts=counts, frames=frames, note='Counts are frame detections, not unique signs. Video limited to first 1800 frames.')
    except Exception as e:return jsonify(error=str(e)),500

@app.route('/outputs/<path:name>')
def output(name):return send_from_directory(OUT,name)

if __name__=='__main__':
    print('Open http://127.0.0.1:5000 in your browser')
    app.run(host='127.0.0.1',port=5000,debug=False,threaded=True)
