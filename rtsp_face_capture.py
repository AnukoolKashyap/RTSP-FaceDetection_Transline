"""
rtsp_face_capture.py

Captures frames from an RTSP camera, detects faces with a YOLO face model,
and saves any frame containing >=1 face into an output folder, throttled
to a target rate (default 5 FPS).

Connection-loss resilience:
  - Grabber thread auto-reconnects on drop with no data loss
  - Every disconnect / reconnect is logged to session_gaps.jsonl
  - On Ctrl+C, a session_summary.json is written covering the full run

Output folder layout:
  captured_faces/
  ├── <timestamp>_faces<N>.jpg     saved frames
  ├── detections.jsonl             one line per saved frame (append-write, crash-safe)
  ├── session_gaps.jsonl           one line per connect/disconnect event
  ├── session_summary.json         written on clean shutdown
  └── crops/                       per-face crops (only if --save-crops)

Usage:
    python rtsp_face_capture.py --rtsp-url "rtsp://user:pass@192.168.1.50:554/stream1"
"""

import argparse
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# Gap logger — writes every connect / disconnect event to disk immediately
# ─────────────────────────────────────────────────────────────────────────────

class GapLogger:
    """
    Append-writes one JSON line per event to session_gaps.jsonl.
    Written immediately on each event so a crash mid-run still produces
    a complete gap record up to that point.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _write(self, record: dict):
        record["iso"] = datetime.now().isoformat()
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def log_connect(self):
        self._write({"event": "connected"})
        print(f"[grabber] connected at {datetime.now().strftime('%H:%M:%S')}")

    def log_disconnect(self):
        self._write({"event": "disconnected"})
        print(f"[grabber] connection lost at {datetime.now().strftime('%H:%M:%S')}")

    def log_reconnect(self, gap_seconds: float):
        self._write({"event": "reconnected", "gap_s": round(gap_seconds, 2)})
        print(f"[grabber] reconnected after {gap_seconds:.1f}s gap")


# ─────────────────────────────────────────────────────────────────────────────
# RTSP frame grabber with gap detection
# ─────────────────────────────────────────────────────────────────────────────

class RTSPFrameGrabber:
    """
    Background thread that reads the RTSP stream and always exposes the
    most recent frame. Decouples camera frame rate from inference speed.

    Gap detection:
      Three states — CONNECTING (initial / after drop), STREAMING (healthy),
      RECONNECTING (drop detected, timing the outage).

      Every state transition is reported to the GapLogger so there is always
      a permanent on-disk record of when the stream was unavailable and for
      how long.
    """

    def __init__(self, rtsp_url: str, gap_logger: GapLogger,
                 reconnect_delay: float = 3.0):
        self.rtsp_url      = rtsp_url
        self.gap_logger    = gap_logger
        self.reconnect_delay = reconnect_delay

        self.cap           = None
        self._lock         = threading.Lock()
        self._latest_frame = None

        # Connection state machine
        self._state        = "CONNECTING"  # CONNECTING | STREAMING | RECONNECTING
        self._drop_time    = None          # wall-clock time of last disconnect

        self.running       = False
        self._thread       = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self.cap.isOpened()

    def _on_drop(self):
        """Called the first time a read fails — transition to RECONNECTING."""
        if self._state == "STREAMING":
            self._state     = "RECONNECTING"
            self._drop_time = time.time()
            self.gap_logger.log_disconnect()

    def _on_reconnect(self):
        """Called when stream produces a good frame after a drop."""
        if self._state in ("CONNECTING", "RECONNECTING"):
            if self._state == "CONNECTING":
                # Very first connect — not a reconnect, just log as connected
                self.gap_logger.log_connect()
            else:
                gap = time.time() - (self._drop_time or time.time())
                self.gap_logger.log_reconnect(gap)
            self._state = "STREAMING"

    def _run(self):
        while self.running:
            # ── ensure we have an open capture ────────────────────────────
            if self.cap is None or not self.cap.isOpened():
                if not self._connect():
                    time.sleep(self.reconnect_delay)
                    continue

            # ── read one frame ─────────────────────────────────────────────
            ok, frame = self.cap.read()

            if not ok:
                self._on_drop()
                self.cap.release()
                self.cap = None
                time.sleep(self.reconnect_delay)
                continue

            # Good frame
            self._on_reconnect()

            with self._lock:
                self._latest_frame = frame

    # ── public ────────────────────────────────────────────────────────────────

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def get_latest_frame(self):
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    @property
    def is_streaming(self) -> bool:
        return self._state == "STREAMING"

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# Detection save (unchanged from before)
# ─────────────────────────────────────────────────────────────────────────────

def save_detection(frame, boxes, output_dir: Path,
                   draw_boxes: bool, save_crops: bool):
    """Persist one frame that contains >=1 detected face plus a log entry."""
    now  = datetime.now()
    ts   = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    n    = len(boxes)
    annotated  = frame.copy()
    box_records = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        conf = float(box.conf[0])
        box_records.append({"xyxy": [x1, y1, x2, y2], "confidence": round(conf, 4)})

        if draw_boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if save_crops:
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                crop_dir = output_dir / "crops"
                crop_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(crop_dir / f"{ts}_face{i}.jpg"), crop)

    filename = f"{ts}_faces{n}.jpg"
    cv2.imwrite(str(output_dir / filename), annotated)

    entry = {"timestamp": now.isoformat(), "filename": filename,
             "face_count": n, "boxes": box_records}
    with open(output_dir / "detections.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return filename, n


# ─────────────────────────────────────────────────────────────────────────────
# Session summary — written on clean shutdown
# ─────────────────────────────────────────────────────────────────────────────

def write_session_summary(output_dir: Path, start_time: datetime,
                          frames_saved: int, faces_total: int):
    """
    Reads session_gaps.jsonl and writes a human-readable session_summary.json.
    Safe to call even if gaps file is empty or missing.
    """
    end_time    = datetime.now()
    duration_s  = (end_time - start_time).total_seconds()

    gap_events  = []
    gaps_path   = output_dir / "session_gaps.jsonl"
    total_gap_s = 0.0
    gap_count   = 0

    if gaps_path.exists():
        with open(gaps_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ev = json.loads(line)
                    gap_events.append(ev)
                    if ev.get("event") == "reconnected":
                        total_gap_s += ev.get("gap_s", 0)
                        gap_count   += 1

    active_s = duration_s - total_gap_s

    summary = {
        "session_start":    start_time.isoformat(),
        "session_end":      end_time.isoformat(),
        "duration_s":       round(duration_s, 1),
        "active_stream_s":  round(active_s, 1),
        "total_gap_s":      round(total_gap_s, 1),
        "gap_count":        gap_count,
        "frames_saved":     frames_saved,
        "faces_detected":   faces_total,
        "gap_events":       gap_events,
    }

    path = output_dir / "session_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print a clean terminal summary
    print(f"\n{'='*55}")
    print(f"  SESSION SUMMARY")
    print(f"{'='*55}")
    print(f"  Started          : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Ended            : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total duration   : {duration_s/60:.1f} min")
    print(f"  Active stream    : {active_s/60:.1f} min")
    print(f"  Connection gaps  : {gap_count}  (total {total_gap_s:.1f}s offline)")
    print(f"  Frames saved     : {frames_saved}")
    print(f"  Faces detected   : {faces_total}")
    print(f"  Summary written  : {path}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RTSP -> YOLO face detection -> folder capture (connection-resilient)"
    )
    parser.add_argument("--rtsp-url",   required=True)
    parser.add_argument("--model",      default="yolov8n-face.pt")
    parser.add_argument("--output",     default="captured_faces")
    parser.add_argument("--fps",        type=float, default=5.0)
    parser.add_argument("--conf",       type=float, default=0.4)
    parser.add_argument("--imgsz",      type=int,   default=1280)
    parser.add_argument("--no-boxes",   action="store_true")
    parser.add_argument("--save-crops", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gap_logger  = GapLogger(output_dir / "session_gaps.jsonl")
    grabber     = RTSPFrameGrabber(args.rtsp_url, gap_logger)

    print(f"[main] loading model: {args.model}")
    model = YOLO(args.model)

    grabber.start()

    frame_interval = 1.0 / args.fps
    next_tick      = time.time()
    start_time     = datetime.now()
    saved          = 0
    faces_total    = 0

    print(f"[main] running at ~{args.fps} FPS · saving to '{output_dir}' · Ctrl+C to stop")

    try:
        while True:
            now = time.time()
            if now < next_tick:
                time.sleep(min(0.01, next_tick - now))
                continue
            next_tick = now + frame_interval

            # Skip inference while grabber is not streaming
            if not grabber.is_streaming:
                continue

            frame = grabber.get_latest_frame()
            if frame is None:
                continue

            results = model.predict(frame, conf=args.conf,
                                    imgsz=args.imgsz, verbose=False)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                filename, count = save_detection(
                    frame, boxes, output_dir,
                    draw_boxes=not args.no_boxes,
                    save_crops=args.save_crops,
                )
                saved       += 1
                faces_total += count
                print(f"[main] saved {filename}  "
                      f"({count} face{'s' if count != 1 else ''}  |  "
                      f"total saved: {saved})")

    except KeyboardInterrupt:
        print("\n[main] shutting down...")
    finally:
        grabber.stop()
        write_session_summary(output_dir, start_time, saved, faces_total)


if __name__ == "__main__":
    main()