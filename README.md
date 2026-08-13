# Feature Showcase: Realtime Image Processing Object Identifier

This is a minimal feature showcase for realtime object identification using OpenCV's DNN and MobileNet-SSD (Caffe).

Quick start

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Run the demo (uses your default webcam):

```bash
python app.py
```

Notes
- On first run the required model files will be downloaded into the `models/` folder.
- Press `q` in the display window to quit.

This project is for testing only — a simple local demo to showcase realtime image processing.

Deployment (recommended split)
---------------------------------
This repo contains two parts:
- `web/` — static frontend (HTML/JS) suitable for Vercel or any static host.
- Flask backend (`app.py`) — runs the ML server and provides API endpoints.

Recommended setup:

1) Frontend on Vercel (static)
 - Copy `web/config.example.js` to `web/config.js` and set `API_BASE_URL` to your backend URL (no trailing slash), e.g.
	 ```js
	 // web/config.js
	 const API_BASE_URL = 'https://your-backend.onrender.com';
	 ```
 - Push to GitHub and import the repo in Vercel (select the project and deploy). `vercel.json` is present to serve the `web/` folder.

2) Backend on Render (or Railway/Fly)
 - Create a new Web Service on Render.
 - Connect your GitHub repo and select the root as the deploy directory.
 - Build Command: `pip install -r requirements.txt`
 - Start Command: `gunicorn app:app`
 - Set any necessary environment variables (none required by default).

Notes:
- Vercel serverless functions have a hard bundle size limit; large ML packages (torch, ultralytics) will exceed this. That's why we split frontend (Vercel) and backend (Render).
- If you want a single-host deployment, use Render/Railway/Fly which support larger Python builds.

Local testing
--------------
Run the backend locally and open the frontend with `file://` or a simple static server. If both run locally on the same machine, leave `web/config.js` blank so the frontend uses relative URLs.

