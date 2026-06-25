import os
import time
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from model import CLASSES, load_model, run_inference

app = FastAPI(title="Violence Detector API", version="1.0.0")

# ── Prometheus metrics ─────────────────────────────────────────────────────────
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "End-to-end inference latency per request",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0],
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence score of the winning class",
    ["predicted_class"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Cumulative predictions by class",
    ["predicted_class"],
)
MODEL_LOADED = Gauge("model_loaded", "1 when model weights are loaded and ready")

_model = None


@app.on_event("startup")
async def startup() -> None:
    global _model
    _model = load_model()
    MODEL_LOADED.set(1)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        t0              = time.perf_counter()
        label, probs    = run_inference(tmp_path, _model)
        latency         = time.perf_counter() - t0

        if label is None:
            raise HTTPException(
                status_code=422,
                detail="Video too short — need at least 8 frames.",
            )

        confidence = max(probs)
        INFERENCE_LATENCY.observe(latency)
        PREDICTION_CONFIDENCE.labels(predicted_class=label).observe(confidence)
        PREDICTIONS_TOTAL.labels(predicted_class=label).inc()

        return {
            "prediction":       label,
            "confidence":       round(confidence, 4),
            "probabilities":    {c: round(p, 4) for c, p in zip(CLASSES, probs)},
            "latency_seconds":  round(latency, 3),
        }
    finally:
        os.unlink(tmp_path)
