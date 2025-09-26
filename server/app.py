"""Nexia AI multi-agent NSFW scanning service.

This module exposes a Flask application that wraps the legacy
queue-based scanner with concurrent workers ("agents").  Up to four
agents can run in parallel and the limit can be changed at runtime via
an HTTP endpoint.  Jobs are accepted through the ``/scan`` endpoint and
processed in the background.  Results are saved to ``processed/`` in the
same format as the former standalone script so the Android application
remains compatible.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from flask import Flask, Response, jsonify, request
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class NSFWModel(nn.Module):
    """Very small CNN used for demonstration purposes."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 Klassen: SFW, NSFW

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Configuration & paths
# ---------------------------------------------------------------------------

QUEUE_DIR = "queue"
PROCESSED_DIR = "processed"
ERRORS_DIR = "errors"
MODELS_DIR = "models"
MAX_AGENTS = 4
DEFAULT_AGENT_COUNT = 2

for directory in (QUEUE_DIR, PROCESSED_DIR, ERRORS_DIR, MODELS_DIR):
    os.makedirs(directory, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = NSFWModel().to(device)
base_model.eval()

transform = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


@dataclass
class ModelDescriptor:
    slug: str
    path: str
    display: str
    version: Optional[int]


_loaded_model: Optional[ModelDescriptor] = None
_loaded_slug: Optional[str] = None
_loaded_lock = threading.Lock()


def parse_model_filename(fname: str) -> Tuple[str, str, Optional[int]]:
    base = os.path.splitext(fname)[0]
    lower = base.lower()

    tier = None
    if "fast" in lower:
        tier = "fast"
    elif "pro" in lower:
        tier = "pro"
    elif "lite" in lower:
        tier = "lite"

    version = None
    m_ver = re.search(r"(?:version|v|[_-])(\d+)$", lower)
    if m_ver:
        version = int(m_ver.group(1))

    if tier == "fast":
        prefix = "nexia fast"
    elif tier == "pro":
        prefix = "nexia ai pro"
    elif tier == "lite":
        prefix = "nexia ai lite"
    else:
        prefix = "nexia ai"

    display = f"{prefix} version {version}" if version is not None else prefix
    slug = base
    return slug, display, version


def discover_models() -> List[ModelDescriptor]:
    models: List[ModelDescriptor] = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pth"):
            full_path = os.path.join(MODELS_DIR, filename)
            slug, display, version = parse_model_filename(filename)
            models.append(ModelDescriptor(slug, full_path, display, version))

    if not models and os.path.exists("model.pth"):
        slug, display, version = parse_model_filename("model.pth")
        models.append(ModelDescriptor(slug, "model.pth", display, version))

    models.sort(key=lambda m: (-(m.version or -1), m.slug))
    return models


def load_model(slug: str = "auto") -> Tuple[str, str]:
    global _loaded_model, _loaded_slug
    with _loaded_lock:
        models = discover_models()
        if not models:
            raise RuntimeError("Keine Modelle gefunden!")

        if slug == "auto":
            target = models[0]
        else:
            target = next((m for m in models if m.slug == slug), None)
            if target is None:
                raise RuntimeError(f"Modell '{slug}' nicht gefunden!")

        if _loaded_slug != target.slug:
            state_dict = torch.load(target.path, map_location=device)
            base_model.load_state_dict(state_dict, strict=False)
            _loaded_model = target
            _loaded_slug = target.slug
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        assert _loaded_model is not None
        return _loaded_slug or target.slug, _loaded_model.display


# ---------------------------------------------------------------------------
# Job & queue handling
# ---------------------------------------------------------------------------


def read_model_request(image_path: str) -> str:
    sidecar_path = image_path + ".model"
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path, "r", encoding="utf-8") as file:
                return file.read().strip()
        except Exception:
            return "auto"
    return "auto"


@dataclass
class QueueJob:
    identifier: str
    path: str
    requested_model: str
    received_at: float


pending_jobs: "queue.Queue[QueueJob]" = queue.Queue()
results_lock = threading.Lock()
results_index: Dict[str, Dict[str, object]] = {}
processed_counter = 0
error_counter = 0

processing_semaphore = threading.Semaphore(DEFAULT_AGENT_COUNT)
current_agent_limit = DEFAULT_AGENT_COUNT
agent_limit_lock = threading.Lock()

active_workers = 0
active_workers_lock = threading.Lock()


def set_agent_limit(count: int) -> int:
    global current_agent_limit
    if count < 1 or count > MAX_AGENTS:
        raise ValueError(f"Agentenzahl muss zwischen 1 und {MAX_AGENTS} liegen.")

    with agent_limit_lock:
        if count == current_agent_limit:
            return current_agent_limit
        if count > current_agent_limit:
            delta = count - current_agent_limit
            for _ in range(delta):
                processing_semaphore.release()
        else:
            delta = current_agent_limit - count
            for _ in range(delta):
                processing_semaphore.acquire()
        current_agent_limit = count
        return current_agent_limit


worker_threads: List[threading.Thread] = []
worker_started = threading.Event()
shutdown_event = threading.Event()


def process_image(file_path: str, requested_model: Optional[str] = None) -> Tuple[str, float, Dict[str, object]]:
    requested = requested_model or read_model_request(file_path)
    used_slug, display_name = load_model(requested)

    image = Image.open(file_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = base_model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    elapsed = time.time() - start

    prediction = "NSFW" if predicted_idx.item() == 1 else "SAFE"
    model_info: Dict[str, object] = {
        "slug": used_slug,
        "display": display_name,
        "processing_time_ms": int(elapsed * 1000),
    }
    return prediction, float(confidence.item()), model_info



def _record_result(job: QueueJob, label: str, confidence: float, model_info: Dict[str, object]) -> None:
    global processed_counter
    result_file = os.path.join(PROCESSED_DIR, f"{job.identifier}.txt")
    json_file = os.path.join(PROCESSED_DIR, f"{job.identifier}.json")

    with open(result_file, "w", encoding="utf-8") as handle:
        handle.write(label.upper())

    payload = {
        "label": label.upper(),
        "confidence": confidence,
        "model": model_info,
        "time_ms": model_info.get("processing_time_ms", 0),
    }
    with open(json_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)

    with results_lock:
        results_index[job.identifier] = payload

    try:
        os.remove(job.path)
    except FileNotFoundError:
        pass
    sidecar = job.path + ".model"
    if os.path.exists(sidecar):
        os.remove(sidecar)

    processed_counter += 1



def _record_error(job: QueueJob, error: Exception) -> None:
    global error_counter
    error_counter += 1
    target = os.path.join(ERRORS_DIR, os.path.basename(job.path))
    try:
        os.rename(job.path, target)
    except FileNotFoundError:
        pass
    with open(os.path.join(ERRORS_DIR, f"{job.identifier}.error"), "w", encoding="utf-8") as handle:
        handle.write(str(error))



def worker_loop(worker_id: int) -> None:
    global active_workers
    worker_started.set()
    while not shutdown_event.is_set():
        try:
            job = pending_jobs.get(timeout=1.0)
        except queue.Empty:
            continue

        acquired = processing_semaphore.acquire(timeout=5)
        if not acquired:
            pending_jobs.put(job)
            pending_jobs.task_done()
            continue

        with active_workers_lock:
            active_workers += 1

        try:
            label, confidence, model_info = process_image(job.path, job.requested_model)
            _record_result(job, label, confidence, model_info)
        except Exception as exc:  # pylint: disable=broad-except
            _record_error(job, exc)
        finally:
            processing_semaphore.release()
            with active_workers_lock:
                active_workers -= 1
            pending_jobs.task_done()



def start_workers() -> None:
    if worker_threads:
        return
    for idx in range(MAX_AGENTS):
        thread = threading.Thread(target=worker_loop, args=(idx,), daemon=True)
        worker_threads.append(thread)
        thread.start()
    worker_started.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _queue_from_directory() -> Iterable[QueueJob]:
    files = [
        f
        for f in os.listdir(QUEUE_DIR)
        if os.path.isfile(os.path.join(QUEUE_DIR, f)) and not f.endswith(".model")
    ]
    files.sort(key=lambda name: os.path.getctime(os.path.join(QUEUE_DIR, name)))
    for name in files:
        path = os.path.join(QUEUE_DIR, name)
        identifier, _ = os.path.splitext(name)
        yield QueueJob(identifier=identifier, path=path, requested_model=read_model_request(path), received_at=os.path.getctime(path))



def bootstrap_queue_from_disk() -> None:
    for job in _queue_from_directory():
        pending_jobs.put(job)


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

app = Flask(__name__)
start_workers()
bootstrap_queue_from_disk()


@app.route("/status", methods=["GET"])
def status_endpoint() -> Response:
    queue_size = pending_jobs.qsize()
    with active_workers_lock:
        currently_processing = active_workers
    message = (
        f"Nexia AI Server aktiv – {currently_processing} von {current_agent_limit} Agenten prüfen. "
        f"Warteschlange: {queue_size} Dateien, verarbeitet: {processed_counter}, Fehler: {error_counter}."
    )
    return Response(message, mimetype="text/plain")


@app.route("/models", methods=["GET"])
def models_endpoint() -> Response:
    models = [
        {
            "slug": model.slug,
            "display": model.display,
            "version": model.version,
            "tier": (
                "fast"
                if "fast" in model.display.lower()
                else "pro"
                if "pro" in model.display.lower()
                else "lite"
                if "lite" in model.display.lower()
                else "standard"
            ),
        }
        for model in discover_models()
    ]
    return jsonify(models)


@app.route("/agents", methods=["GET", "POST"])
def agents_endpoint() -> Response:
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        requested = int(data.get("count", DEFAULT_AGENT_COUNT))
        limit = set_agent_limit(requested)
        return jsonify({"count": limit})
    return jsonify({"count": current_agent_limit, "max": MAX_AGENTS})


@app.route("/scan", methods=["POST"])
def scan_endpoint() -> Response:
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    image_payload = data.get("image")
    if not image_payload:
        return jsonify({"error": "Feld 'image' fehlt"}), 400

    requested_model = data.get("model", "auto")

    try:
        binary = base64.b64decode(image_payload)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Bild konnte nicht dekodiert werden: {exc}"}), 400

    identifier = data.get("id") or uuid.uuid4().hex
    filename = f"{identifier}.jpg"
    file_path = os.path.join(QUEUE_DIR, filename)

    with open(file_path, "wb") as handle:
        handle.write(binary)

    if requested_model and requested_model != "auto":
        with open(file_path + ".model", "w", encoding="utf-8") as handle:
            handle.write(requested_model)

    job = QueueJob(
        identifier=identifier,
        path=file_path,
        requested_model=requested_model,
        received_at=time.time(),
    )
    pending_jobs.put(job)

    return jsonify({"job_id": identifier, "queued": True})


@app.route("/results/<job_id>", methods=["GET"])
def results_endpoint(job_id: str) -> Response:
    with results_lock:
        payload = results_index.get(job_id)
    if payload is None:
        result_path = os.path.join(PROCESSED_DIR, f"{job_id}.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            return jsonify({"status": "pending"})
    return jsonify({"status": "done", "result": payload})


@app.route("/shutdown", methods=["POST"])
def shutdown_endpoint() -> Response:
    shutdown_event.set()
    return jsonify({"message": "Shutdown signal received"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7535)
