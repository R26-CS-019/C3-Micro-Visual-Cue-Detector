"""Local browser interface for the frozen visual research component (stdlib HTTP)."""
import argparse
import csv
import io
import json
import os
import re
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
JOBS = {}
LOCK = threading.Lock()
WORKER = ThreadPoolExecutor(max_workers=1)
ENGINE = None
COMPONENT = None
PROJECT = None
RUNS = None


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def update(job_id, **fields):
    with LOCK:
        JOBS[job_id].update(fields)


def initialise():
    global ENGINE, COMPONENT
    if ENGINE is not None:
        return
    sys.path.insert(0, str(PROJECT / "research_suite"))
    sys.path.insert(0, str(PROJECT / "Micro Visual Cue Component" / "src"))
    from engine import Engine
    import visual_component
    candidate = Engine(PROJECT, PROJECT / "research_suite" / "models.json")
    # Refuse a mismatch between the research registry and deployed component.
    for number in (1, 2):
        spec = candidate.spec(number)
        if (PROJECT / spec["checkpoint"]).resolve() != getattr(visual_component, f"MODEL{number}_PATH").resolve():
            raise ValueError(f"Model {number} checkpoint differs between component and registry")
        if spec["threshold"] != getattr(visual_component, f"MODEL{number}_THRESHOLD"):
            raise ValueError(f"Model {number} threshold differs between component and registry")
        candidate.model(number)
    COMPONENT = visual_component
    ENGINE = candidate


WARNING_TEXT = {
    "possible_access_block_or_verification": "A verification or access-block page may have been captured. Review the screenshot.",
    "near_blank_capture": "The screenshot is almost blank. Check that the intended page loaded.",
}


def capture(url, folder):
    """Capture with the exact browser profile used by the hosted API (hosting_diagnostics).

    Sharing one capture function keeps local and Azure screenshots comparable.
    """
    from hosting_diagnostics import capture_with_diagnostics, read_capture
    COMPONENT.require_public_url(url)
    html, screenshot, _ = capture_with_diagnostics(url, folder.parent, COMPONENT._guard_route, folder=folder)
    record = read_capture(screenshot)
    settings = record["settings"]
    warnings = [WARNING_TEXT.get(w, f"The server returned HTTP {w[11:]}." if w.startswith("http_error_") else w)
                for w in record["warnings"]]
    return html, {"requested_url": url, "final_url": record.get("final_url"), "title": record.get("title", ""),
                  "http_status": record.get("http_status"),
                  "viewport": [settings["viewport"]["width"], settings["viewport"]["height"]],
                  "headless": settings["headless"], "browser": f"{record.get('browser_channel_used')} {record.get('browser_version')}",
                  "capture_profile": settings["profile"], "user_agent": record.get("user_agent"),
                  "wait_after_dom_ms": settings["render_wait_ms"], "warnings": warnings,
                  "valid_for_assessment": record["valid_for_assessment"]}


def run_job(job_id):
    folder = RUNS / job_id
    folder.mkdir(parents=True)
    with LOCK:
        url = JOBS[job_id]["url"]
    try:
        update(job_id, status="running", stage="Loading and verifying the two checkpoints…")
        initialise()
        update(job_id, stage="Opening the URL and capturing the page…")
        html, captured = capture(url, folder)
        update(job_id, stage="Routing the page and running the selected model…")
        router = COMPONENT.detect_page_type(html, url)
        number = 1 if router["page_type"] == "LOGIN" else 2
        # Only one worker uses the component's preview directory at a time.
        COMPONENT.PREVIEW_DIR = folder
        image = folder / "screenshot.png"
        predict = getattr(COMPONENT, f"predict_model{number}")
        prediction = predict(ENGINE.model(number), image)
        prepare = getattr(COMPONENT, f"preprocess_model{number}")
        array = prepare(image)[0][0]
        import numpy as np
        if not np.array_equal(array, ENGINE.prepare(image, number)):
            raise ValueError("Interface preprocessing differs from the component")
        reference = ENGINE.score(array, number)
        difference = abs(reference - prediction["phishing_probability"])
        if difference > 1e-5:
            raise ValueError("Interface and component predictions disagree")
        update(job_id, stage="Generating phishing and legitimate Grad-CAM evidence…")
        explanation = ENGINE.explain(array, number, folder)
        result = {"id": job_id, "created_at": JOBS[job_id]["created_at"], "url": url,
                  "capture": captured, "router": router, "model_number": number,
                  "prediction": prediction["prediction"], "phishing_score": reference,
                  "threshold": prediction["threshold"], "checkpoint": ENGINE.metadata(number),
                  "explanation": explanation, "component_score_difference": difference,
                  "limitations": ["A model score is not a calibrated probability or proof of website authenticity.",
                                   "Grad-CAM shows model sensitivity; it does not prove the absence of brand memorisation.",
                                   "Routing uses rendered HTML and the submitted URL, as in the original component.",
                                   "Capture uses the same browser profile as the hosted API; websites may still serve different content to different networks or regions."]}
        write_json(folder / "result.json", result)
        with (folder / "prediction.csv").open("w", newline="", encoding="utf-8-sig") as f:
            fields = ["id", "created_at", "url", "model_number", "prediction", "phishing_score", "threshold"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerow(result)
        update(job_id, status="complete", stage="Analysis complete", result=result)
    except Exception as exc:
        traceback.print_exc()
        message = str(exc)
        write_json(folder / "error.json", {"url": url, "error": message})
        update(job_id, status="failed", stage="Analysis could not finish", error=message)


def clean_url(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError("Enter a webpage URL.")
    url = value.strip()
    if "://" not in url:
        url = "https://" + url
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or any(c.isspace() for c in url):
        raise ValueError("Use a valid http:// or https:// webpage URL.")
    if parsed.username or parsed.password:
        raise ValueError("Use a URL without embedded account credentials.")
    return url


class Handler(BaseHTTPRequestHandler):
    def valid_host(self):
        return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def log_message(self, fmt, *args):
        # Do not write submitted URLs or query strings to the console access log.
        pass

    def send(self, status, body, kind="application/json; charset=utf-8", attachment=None):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        if attachment:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # The browser navigated away or cancelled an image request.

    def do_GET(self):
        if not self.valid_host():
            return self.send(403, {"error": "Use the localhost interface address"})
        path = urlsplit(self.path).path
        if path in ("/", "/app.js", "/style.css"):
            filename, kind = {"/": ("index.html", "text/html; charset=utf-8"),
                              "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                              "/style.css": ("style.css", "text/css; charset=utf-8")}[path]
            return self.send(200, (HERE / "static" / filename).read_bytes(), kind)
        if path == "/api/jobs":
            with LOCK:
                jobs = [dict(v) for v in reversed(list(JOBS.values()))]
            return self.send(200, jobs)
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})", path)
        if match:
            with LOCK:
                job = dict(JOBS[match[1]]) if match[1] in JOBS else None
            return self.send(200, job) if job else self.send(404, {"error": "Run not found"})
        match = re.fullmatch(r"/files/([a-f0-9]{32})/([a-z_]+\.(?:png|json|csv|npy))", path)
        if match:
            filename = match[2]
            allowed = {"screenshot.png", "model_input.png", "phishing_overlay.png", "legitimate_overlay.png",
                       "result.json", "prediction.csv", "explanation.json", "model_input_float.npy",
                       "phishing_heatmap.npy", "legitimate_heatmap.npy"}
            target = RUNS / match[1] / filename
            if filename in allowed and target.is_file():
                kind = "image/png" if filename.endswith(".png") else "application/octet-stream"
                return self.send(200, target.read_bytes(), kind, None if kind == "image/png" else filename)
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        if not self.valid_host():
            return self.send(403, {"error": "Use the localhost interface address"})
        # Reject cross-origin requests to this localhost service.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.send(403, {"error": "Cross-origin requests are disabled"})
        if self.path != "/api/jobs":
            return self.send(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(length))
            url = clean_url(data.get("url"))
        except (ValueError, AttributeError) as exc:
            return self.send(400, {"error": str(exc)})
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "url": url, "status": "queued", "stage": "Waiting to start…",
               "created_at": datetime.now(timezone.utc).isoformat()}
        with LOCK:
            active = sum(j["status"] in ("queued", "running") for j in JOBS.values())
            if active >= 3:
                return self.send(429, {"error": "Three analyses are already waiting. Let them finish first."})
            JOBS[job_id] = job
        WORKER.submit(run_job, job_id)
        self.send(202, dict(job))


def main():
    global PROJECT, RUNS
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(r"C:\Users\oshan\Desktop\full  pp2 module"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    PROJECT = args.project.resolve(); RUNS = args.results.resolve()
    if not (PROJECT / "research_suite" / "models.json").is_file():
        parser.error("The project research_suite/models.json was not found")
    RUNS.mkdir(parents=True, exist_ok=True)
    for path in sorted(RUNS.glob("*/result.json"), key=lambda p: p.stat().st_mtime):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if re.fullmatch(r"[a-f0-9]{32}", path.parent.name):
                JOBS[path.parent.name] = {"id": path.parent.name, "url": result["url"], "status": "complete",
                                          "stage": "Analysis complete", "created_at": result["created_at"], "result": result}
        except (ValueError, KeyError):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\nVisual component interface: http://127.0.0.1:{args.port}\nResults: {RUNS}\nKeep this terminal open. Press Ctrl+C to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping the interface…", flush=True)
    finally:
        server.server_close()
        WORKER.shutdown(wait=True)


if __name__ == "__main__":
    main()
