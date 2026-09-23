# Native video API for Apple Silicon

Runs YOLO26n object detection and YOLO26n depth on the Apple GPU (PyTorch MPS).
One active camera, RTSP/HTTP stream, or video file can serve multiple local clients.
The API stores only the latest analyzed frame in memory; it does not record video.

## Run

Tested with native ARM Python 3.13 and PyTorch 2.14 on an M3 Mac.
From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . -r examples/native-video-api/requirements.txt
mkdir -p .local-runtime
cd .local-runtime
../.venv/bin/python -m uvicorn server:app \
  --app-dir ../examples/native-video-api --host 127.0.0.1 --port 8765 --workers 1
```

Models download on first use into the working directory. Use one server worker.
Keep the server bound to localhost: this API is intended for trusted applications
on this computer and has no authentication. No CORS access is enabled by default.
Optional `VIDEO_SOURCE=0` starts camera index 0 when the service starts.
Camera indices can change when devices are reconnected; check the camera list first.
macOS must grant camera access to the process launching Python.

## Use from another app

Interactive API documentation: <http://127.0.0.1:8765/docs>.

```bash
# Camera index as a string; replace with an RTSP/HTTP URL or absolute video path.
curl -X POST http://127.0.0.1:8765/stream \
  -H 'Content-Type: application/json' -d '{"source":"0","fps":5}'

curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/results
curl http://127.0.0.1:8765/depth.npy -o /tmp/depth.npy
curl http://127.0.0.1:8765/preview.jpg -o /tmp/preview.jpg
curl -X DELETE http://127.0.0.1:8765/stream
```

`POST /stream` returns 202 while models and capture initialize. A second start returns
409; stop the existing source before replacing it. `GET /results` returns 503 before
the first complete frame. Poll `/health` for `starting`, `running`, `ended`, or `error`.
An invalid source reports `error`; correct it and restart with DELETE followed by POST.
Stopping releases the camera and GPU memory, including a stalled capture process.

Results contain object names, confidence, pixel bounding boxes, and median depth
inside each box. `sequence` counts analyzed frames, not original video frame numbers.
`processed_at` is the result publication time, not the camera capture time.
Check `status` and `age_seconds`: a disconnected camera can leave an old result.
Live sources use the existing Ultralytics loader with buffering disabled, dropping
frames when needed. `fps` caps analysis frequency; it does not change camera FPS.
Files are processed sequentially at the analysis rate, rather than wall-clock playback.

The depth file is a float32 NumPy array with the original frame's height and width:

```python
import io

import numpy as np
import requests

response = requests.get("http://127.0.0.1:8765/depth.npy", timeout=10)
response.raise_for_status()
depth = np.load(io.BytesIO(response.content), allow_pickle=False)
```

Depth is **relative and uncalibrated, not reliable meters**. Larger values indicate
farther surfaces. Object-box medians can include the background and are approximate.
Separate requests can read different frames; binary responses include
`X-Frame-Sequence` so clients can identify the analyzed frame within a stream.
Poll `/health` alongside binary responses to detect a stopped or failed stream.

## Installed background service

On the configured Mac, the launch agent is
`~/Library/LaunchAgents/com.typossum.video-analysis.plist`.
It starts the localhost API at login and restarts it if it exits.
Its working directory and log files are under `.local-runtime` in this worktree.
Do not move or delete this worktree while the service is installed.
The configured agent uses `.local-runtime/Video Analysis.app` to request macOS camera
permission before starting Python. Its source is `.local-runtime/camera-launcher.swift`.
It selects the connected Innomaker USB camera by device identity at startup, then
passes its current index through `VIDEO_SOURCE`. If that device is absent, the API
starts idle. The launcher and device-specific settings are local to this Mac.

```bash
# Restart the installed API.
launchctl kickstart -k "gui/$(id -u)/com.typossum.video-analysis"

# Stop the API and disable it for this login session.
launchctl bootout "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.typossum.video-analysis.plist"

# Enable it again.
launchctl bootstrap "gui/$(id -u)" \
  "$HOME/Library/LaunchAgents/com.typossum.video-analysis.plist"
```

To uninstall, boot out the agent, then remove its plist. The Python environment and
model files can remain for manual use. No main Ultralytics package code is modified.
