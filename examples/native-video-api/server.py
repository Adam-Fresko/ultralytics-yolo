"""Local, single-source video detection and relative-depth API. Run with one Uvicorn worker."""

import asyncio
import io
import logging
import multiprocessing as mp
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field


def analyze(source, fps, shared):
    """Use the existing streaming predictor; publish only the latest complete frame."""
    import cv2
    import numpy as np
    import torch

    from ultralytics import YOLO

    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple GPU (MPS) is unavailable")
        detector, depth_model = YOLO("yolo26n.pt"), YOLO("yolo26n-depth.pt")
        predictions = detector.predict(source, stream=True, stream_buffer=False, device="mps", imgsz=640, verbose=False)
        for sequence, result in enumerate(predictions, 1):
            started = time.monotonic()
            depth = depth_model.predict(result.orig_img, device="mps", imgsz=640, verbose=False)[0]
            values = depth.depth.data.cpu().numpy().astype(np.float32)
            if not np.isfinite(values).all():
                raise RuntimeError("Depth model returned non-finite values")
            objects = result.summary()
            for obj in objects:
                box = obj["box"]
                x1, y1, x2, y2 = (int(box[key]) for key in ("x1", "y1", "x2", "y2"))
                region = values[y1:y2, x1:x2]
                obj["relative_depth"] = float(np.median(region)) if region.size else None
            buffer = io.BytesIO()
            np.save(buffer, values, allow_pickle=False)
            ok, preview = cv2.imencode(".jpg", result.plot())
            if not ok:
                raise RuntimeError("Could not encode preview")
            metadata = {
                "sequence": sequence,
                "processed_at": time.time(),
                "width": result.orig_shape[1],
                "height": result.orig_shape[0],
                "depth_units": "relative_uncalibrated",
                "objects": objects,
                "model_ms": {"detect": result.speed, "depth": depth.speed},
            }
            shared["snapshot"] = (metadata, buffer.getvalue(), preview.tobytes())
            shared["status"] = "running"
            time.sleep(max(0, 1 / fps - (time.monotonic() - started)))
        shared["status"] = "ended"
    except Exception as exc:
        logging.getLogger(__name__).exception("Video analysis failed")
        shared["error"] = str(exc)
        shared["status"] = "error"


@asynccontextmanager
async def lifespan(app):
    """Own the capture process so a blocked network read can always be stopped."""
    with mp.get_context("spawn").Manager() as manager:
        app.state.shared = manager.dict(status="idle", snapshot=None, error=None)
        app.state.worker = None
        app.state.stream_id = None
        if os.environ.get("VIDEO_SOURCE"):
            await start_stream(StreamRequest(source=os.environ["VIDEO_SOURCE"]))
        yield
        await stop_stream()


app = FastAPI(title="Local video detection and depth", lifespan=lifespan)
control = asyncio.Lock()


class StreamRequest(BaseModel):
    """Source understood by Ultralytics: RTSP/HTTP URL, video path, or camera index string."""

    source: str = Field(min_length=1)
    fps: float = Field(default=5, gt=0, le=30)


@app.get("/health")
async def health():
    """Report stream state; an idle API is ready to accept a source."""
    status = app.state.shared["status"]
    worker = app.state.worker
    if worker is not None and not worker.is_alive() and status in {"starting", "running"}:
        status = "error"
    return {
        "status": status,
        "stream_id": app.state.stream_id,
        "device": "mps",
        "error": app.state.shared["error"],
    }


@app.post("/stream", status_code=202)
async def start_stream(request: StreamRequest):
    """Start one source. Stop the current source before starting another."""
    async with control:
        if app.state.worker is not None:
            raise HTTPException(409, "Stop the current stream with DELETE /stream first")
        app.state.shared.update(status="starting", snapshot=None, error=None)
        app.state.stream_id = str(uuid4())
        app.state.worker = mp.get_context("spawn").Process(
            target=analyze, args=(request.source, request.fps, app.state.shared), daemon=True
        )
        app.state.worker.start()
        return await health()


@app.delete("/stream")
async def stop_stream():
    """Release the video source and GPU, including stalled sources."""
    async with control:
        worker = app.state.worker
        if worker is not None:
            worker.terminate()
            await asyncio.to_thread(worker.join, 5)
            if worker.is_alive():
                worker.kill()
                await asyncio.to_thread(worker.join)
            worker.close()
        app.state.worker = None
        app.state.stream_id = None
        app.state.shared.update(status="idle", snapshot=None, error=None)
        return {"status": "idle"}


def snapshot():
    """Read one atomic result, shared by every client without consuming it."""
    value = app.state.shared["snapshot"]
    if value is None:
        raise HTTPException(503, "No analyzed frame yet; check /health")
    return value


@app.get("/results")
async def results():
    """Return the latest detections; check status and age before treating them as live."""
    metadata, _, _ = snapshot()
    return {**metadata, **await health(), "age_seconds": time.time() - metadata["processed_at"]}


@app.get("/depth.npy")
async def depth_map():
    """Return an uncalibrated float32 H×W NumPy depth map aligned with the source image."""
    metadata, data, _ = snapshot()
    return Response(
        data, media_type="application/octet-stream", headers={"X-Frame-Sequence": str(metadata["sequence"])}
    )


@app.get("/preview.jpg")
async def preview():
    """Return the latest frame with object labels and bounding boxes."""
    metadata, _, data = snapshot()
    return Response(data, media_type="image/jpeg", headers={"X-Frame-Sequence": str(metadata["sequence"])})
