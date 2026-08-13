import os
import time
import argparse
import sys
import threading

import cv2
import numpy as np
import requests
from flask import Flask, Response, render_template_string, request, jsonify

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTOTXT = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
MODEL = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

PROTOTXT_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v1_caffe/MobileNetSSD_deploy.prototxt",
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
]

MODEL_URLS = [
    "https://github.com/opencv/opencv_3rdparty/raw/master/dnn_samples_mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
    "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
]
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_DST = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")

# Web streaming globals
output_frame = None
frame_lock = threading.Lock()


app = Flask(__name__)

# Try to enable CORS for the frontend calls. Prefer flask_cors if available,
# otherwise add a permissive header in after_request.
try:
    from flask_cors import CORS
    CORS(app)
    print('Enabled CORS via flask_cors')
except Exception:
    @app.after_request
    def add_cors_headers(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        return response

# CLIP preload globals (loaded in background to avoid blocking first requests)
CLIP_MODEL = None
CLIP_PROCESSOR = None
CLIP_READY = False
CLIP_LOCK = threading.Lock()


def preload_clip():
    global CLIP_MODEL, CLIP_PROCESSOR, CLIP_READY
    try:
        from transformers import CLIPProcessor, CLIPModel
        print("Preloading CLIP model in background (may take some time)...")
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        with CLIP_LOCK:
            CLIP_PROCESSOR = proc
            CLIP_MODEL = model
            CLIP_READY = True
        print("CLIP preload complete")
    except Exception as e:
        print("Background CLIP preload failed or transformers not installed:", e)


# Start background preload thread (won't block startup)
try:
    t = threading.Thread(target=preload_clip, daemon=True)
    t.start()
except Exception:
    pass


@app.route('/')
def index():
    html = open(os.path.join(os.path.dirname(__file__), 'web', 'index.html')).read()
    return render_template_string(html)


@app.route('/upload')
def upload_page():
    html = open(os.path.join(os.path.dirname(__file__), 'web', 'upload.html')).read()
    return render_template_string(html)


def generate_mjpeg():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            chunk = output_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')


def capture_current_frame():
    """Return the latest frame as a BGR numpy array, or None."""
    global output_frame, frame_lock
    with frame_lock:
        if output_frame is None:
            return None
        # decode JPEG bytes to BGR
        arr = np.frombuffer(output_frame, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img


def classify_image(img, labels=None):
    """Classify a BGR `img` (numpy) using CLIP zero-shot when available,
    otherwise fall back to ORB template matching against `models/scenery`.
    Returns a dict {label, score}.
    """
    if img is None:
        return {"label": "no_frame", "score": 0.0}

    if labels is None:
        labels = [
            "damaged rice field",
            "banana tree damage",
            "coconut tree damage",
            "flooded field",
            "drought damaged field",
            "pest damaged crops",
            "normal field",
            "normal trees",
        ]

    # If CLIP has been preloaded in background and is ready, use it (fast).
    global CLIP_READY, CLIP_MODEL, CLIP_PROCESSOR
    if CLIP_READY and CLIP_MODEL is not None and CLIP_PROCESSOR is not None:
        try:
            from PIL import Image
            import torch
            processor = CLIP_PROCESSOR
            model = CLIP_MODEL

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            inputs = processor(text=labels, images=pil, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0].cpu().numpy()
            best_idx = int(probs.argmax())
            return {"label": labels[best_idx], "score": float(probs[best_idx])}
        except Exception as e:
            print("CLIP inference failed; falling back to ORB:", e)
    else:
        # If CLIP isn't ready yet, return ORB-based result quickly to avoid long blocking.
        if not CLIP_READY:
            print("CLIP not ready; using ORB fallback for fast response")

    # ORB fallback: match against saved templates in models/scenery
    templates_dir = os.path.join(MODEL_DIR, 'scenery')
    if not os.path.exists(templates_dir):
        return {"label": "no_templates", "score": 0.0}

    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)

    best_label = 'unknown'
    best_score = 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for fname in os.listdir(templates_dir):
        path = os.path.join(templates_dir, fname)
        tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            continue
        kp2, des2 = orb.detectAndCompute(tmpl, None)
        if des1 is None or des2 is None:
            continue
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good = [m for m in matches if m.distance < 60]
        score = len(good) / max(1, len(matches))
        label = os.path.splitext(fname)[0]
        if score > best_score:
            best_score = score
            best_label = label

    return {"label": best_label, "score": float(best_score)}


@app.route('/classify_scene', methods=['POST'])
def classify_scene():
    try:
        img = capture_current_frame()
        result = classify_image(img)
        return jsonify(result)
    except Exception as e:
        print('Error in /classify_scene:', e)
        return jsonify({'error': 'server_error', 'message': str(e)}), 500


@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Accepts multipart/form-data with field 'image'
    if 'image' not in request.files:
        return jsonify({"error": "no_file"}), 400
    f = request.files['image']
    data = f.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "invalid_image"}), 400
    return jsonify(classify_image(img))


@app.route('/add_template_upload', methods=['POST'])
def add_template_upload():
    # Accepts multipart/form-data with 'image' file and optional 'label' form field
    if 'image' not in request.files:
        return jsonify({"status": "no_file"}), 400
    f = request.files['image']
    label = (request.form.get('label') or '').strip() or 'template'
    data = f.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"status": "invalid_image"}), 400

    templates_dir = os.path.join(MODEL_DIR, 'scenery')
    os.makedirs(templates_dir, exist_ok=True)
    ts = int(time.time())
    fname = f"{label}_{ts}.jpg"
    path = os.path.join(templates_dir, fname)
    cv2.imwrite(path, img)
    return jsonify({"status": "saved", "path": path})


@app.route('/add_template', methods=['POST'])
def add_template():
    # Read JSON from request via Flask utilities
    from flask import request
    payload = {}
    try:
        payload = request.get_json() or {}
    except Exception:
        payload = {}

    label = payload.get('label', '').strip() if isinstance(payload, dict) else ''
    if not label:
        return {"status": "no_label"}

    img = capture_current_frame()
    if img is None:
        return {"status": "no_frame"}

    templates_dir = os.path.join(MODEL_DIR, 'scenery')
    os.makedirs(templates_dir, exist_ok=True)
    ts = int(time.time())
    fname = f"{label}_{ts}.jpg"
    path = os.path.join(templates_dir, fname)
    cv2.imwrite(path, img)
    return {"status": "saved", "path": path}


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


def download_file(url, dst_path, chunk_size=8192):
    print(f"Downloading {url} -> {dst_path}")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def try_download(urls, dst_path):
    last_err = None
    for url in urls:
        try:
            download_file(url, dst_path)
            return True
        except requests.exceptions.HTTPError as e:
            print(f"Failed to download {url}: {e}")
            last_err = e
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            last_err = e
    if last_err:
        raise last_err
    return False


def ensure_models():
    if not os.path.exists(PROTOTXT):
        try_download(PROTOTXT_URLS, PROTOTXT)
    if not os.path.exists(MODEL):
        try_download(MODEL_URLS, MODEL)


def load_network(accurate=False):
    # If accurate=True, try to use a YOLOv8 model (ultralytics) first
    if accurate:
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            print("Using YOLOv8n accurate detector (ultralytics)")
            return ("yolo", model)
        except Exception as ye:
            print("YOLO accurate mode unavailable:", ye)
            print("To enable accurate mode, install: pip install ultralytics torch")

    try:
        ensure_models()
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        return ("dnn", net)
    except Exception as e:
        print("Warning: failed to load DNN model, attempting Haar cascade fallback:", e)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            return ("cascade", cascade)

        try:
            try_download([CASCADE_URL], CASCADE_DST)
            if os.path.exists(CASCADE_DST):
                try:
                    cascade = cv2.CascadeClassifier(CASCADE_DST)
                    return ("cascade", cascade)
                except Exception as ce:
                    print("CascadeClassifier not available or failed to load:", ce)
        except Exception as de:
            print("Failed to download cascade fallback:", de)

        try:
            bgsub = cv2.createBackgroundSubtractorMOG2()
            print("Using background-subtraction fallback detector")
            return ("bgsub", bgsub)
        except Exception as be:
            print("Background-subtractor not available:", be)

        raise RuntimeError("No fallback detector available; install models or ensure OpenCV data is present")


def main(source=0, conf_threshold=0.4, show=True, web=False):
    detector = load_network(getattr(main, 'accurate', False))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Failed to open video source:", source)
        return 1

    # If web mode is enabled, start Flask in a separate thread
    if web:
        server_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, threaded=True, debug=False), daemon=True)
        server_thread.start()
        print("Web UI available at http://127.0.0.1:5000/")

    prev_time = time.time()
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        h, w = frame.shape[:2]

        counts = {}
        kind, obj = detector

        if kind == "yolo":
            model = obj
            try:
                res = model(frame, imgsz=640, conf=conf_threshold, verbose=False)[0]
                try:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    clss = res.boxes.cls.cpu().numpy()
                except Exception:
                    boxes = res.boxes.xyxy.numpy()
                    confs = res.boxes.conf.numpy()
                    clss = res.boxes.cls.numpy()

                names = getattr(model, 'names', {})
                for i, box in enumerate(boxes):
                    startX, startY, endX, endY = box.astype(int)
                    confidence = float(confs[i])
                    cls_id = int(clss[i])
                    label = names.get(cls_id, str(cls_id))
                    counts[label] = counts.get(label, 0) + 1
                    text = f"{label}: {confidence:.2f}"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception as e:
                print("YOLO inference failed:", e)
                try:
                    detector = load_network(accurate=False)
                    continue
                except Exception as le:
                    print("Failed to reload fallback detectors:", le)

        elif kind == "dnn":
            net = obj
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < conf_threshold:
                    continue

                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
                counts[label] = counts.get(label, 0) + 1

                text = f"{label}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        elif kind == "cascade":
            cascade = obj
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, fw, fh) in faces:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (255, 0, 0), 2)
                counts['face'] = counts.get('face', 0) + 1
                cv2.putText(frame, f"face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        elif kind == "bgsub":
            bgsub = obj
            if not hasattr(bgsub, 'apply'):
                print("Warning: bgsub detector missing apply(); skipping bgsub this frame")
            else:
                fg = bgsub.apply(frame)
                _, th = cv2.threshold(fg, 244, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) < 500:
                        continue
                    x, y, wbox, hbox = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + wbox, y + hbox), (0, 128, 255), 2)
                    counts['object'] = counts.get('object', 0) + 1
                    cv2.putText(frame, f"object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)

        # Info overlay
        now = time.time()
        fps = 1.0 / (now - prev_time) if now - prev_time > 0 else 0.0
        prev_time = now
        info_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display counts at top-left
        y0 = 40
        for k, v in list(counts.items())[:6]:
            cv2.putText(frame, f"{k}: {v}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y0 += 18

        # Update web frame store
        global output_frame, frame_lock
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            with frame_lock:
                output_frame = jpeg.tobytes()

        if web:
            time.sleep(0.01)
            continue
        if show:
            cv2.imshow("Realtime Object Identifier - Feature Showcase", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            if counts:
                print("Detected:", counts)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Realtime Object Identifier - Feature Showcase")
    parser.add_argument('--source', default=0, help='Video source (0 for webcam or path to file)')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Run headless (no display)')
    parser.add_argument('--web', action='store_true', help='Start lightweight web UI (MJPEG)')
    parser.add_argument('--accurate', action='store_true', help='Use YOLOv8 accurate detection (requires ultralytics & torch)')
    args = parser.parse_args()
    src = int(args.source) if str(args.source).isdigit() else args.source
    main.accurate = args.accurate
    sys.exit(main(source=src, conf_threshold=args.conf, show=not args.no_display, web=args.web))
