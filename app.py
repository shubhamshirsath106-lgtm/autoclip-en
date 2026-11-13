#!/usr/bin/env python3
"""
app.py - mobile-friendly web UI for autoclip_en.py

Run locally:   python app.py   ->  open http://localhost:7860 (on your phone use your PC's IP)
Production:    gunicorn -w 1 --threads 8 --timeout 600 -b 0.0.0.0:7860 app:app
NOTE: keep a single worker (-w 1); job state is kept in memory.
"""
import os
import shutil
import threading
import time
import uuid
from argparse import Namespace

from flask import Flask, jsonify, request, send_from_directory

import autoclip_en as core

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")
os.makedirs(BASE, exist_ok=True)
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
JOB_TTL = 2 * 3600  # delete jobs after 2 hours

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
JOBS = {}
LOCK = threading.Lock()


def cleanup():
    now = time.time()
    for name in os.listdir(BASE):
        path = os.path.join(BASE, name)
        if os.path.isdir(path) and now - os.path.getmtime(path) > JOB_TTL:
            shutil.rmtree(path, ignore_errors=True)
            JOBS.pop(name, None)


def set_job(job_id, **kw):
    with LOCK:
        JOBS[job_id].update(kw)


def worker(job_id, video, srt, args):
    out = os.path.join(BASE, job_id, "output")
    try:
        set_job(job_id, stage="Reading subtitles" if srt else "Transcribing audio", pct=5)
        cues = core.parse_srt(srt) if srt else core.transcribe(video)
        if not cues:
            raise RuntimeError("No transcript content found.")

        chunks = core.chunk_cues(cues, args.chunk_chars)

        def on_chunk(n, total):
            set_job(job_id, stage="Analyzing %d/%d" % (n, total), pct=10 + int(60 * (n - 1) / total))

        hl = core.find_highlights(chunks, args, core.video_duration(video), on_chunk)
        if not hl:
            raise RuntimeError("No highlights found. Try a lower score threshold (e.g. 0.5).")

        os.makedirs(out, exist_ok=True)
        paths, clips = [], []
        for i, h in enumerate(hl, 1):
            set_job(job_id, stage="Cutting clip %d/%d" % (i, len(hl)), pct=70 + int(25 * (i - 1) / len(hl)))
            name = "%02d_%s.mp4" % (i, core.safe_name(h["title"]))
            core.cut_clip(video, h, os.path.join(out, name))
            paths.append(os.path.join(out, name))
            clips.append({"file": name, "title": h["title"], "score": round(h["score"], 2),
                          "start": core.sec_to_ts(h["start"]), "end": core.sec_to_ts(h["end"])})

        collection = None
        top = sorted(range(len(hl)), key=lambda k: -hl[k]["score"])[:args.max_collection]
        if len(top) > 1:
            set_job(job_id, stage="Building collection", pct=96)
            core.make_collection([paths[k] for k in sorted(top)], os.path.join(out, "collection.mp4"))
            collection = "collection.mp4"

        os.remove(video)  # free disk space
        set_job(job_id, status="done", stage="Done", pct=100, clips=clips, collection=collection)
    except (Exception, SystemExit) as e:
        set_job(job_id, status="error", stage="Failed", error=str(e) or e.__class__.__name__)


@app.post("/start")
def start():
    cleanup()
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="Please choose a video file."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in VIDEO_EXT:
        return jsonify(error="Unsupported video type."), 400

    job_id = uuid.uuid4().hex
    folder = os.path.join(BASE, job_id)
    os.makedirs(folder)
    video = os.path.join(folder, "input" + ext)
    f.save(video)
    srt = None
    s = request.files.get("srt")
    if s and s.filename:
        srt = os.path.join(folder, "input.srt")
        s.save(srt)

    form = request.form

    def num(key, default, cast):
        try:
            return cast(form.get(key) or default)
        except ValueError:
            return cast(default)

    args = Namespace(
        base_url=form.get("base_url") or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=form.get("api_key") or os.getenv("LLM_API_KEY", ""),  # never written to disk
        model=form.get("model") or os.getenv("LLM_MODEL", "qwen2.5:7b"),
        chunk_chars=5000,
        min_score=min(max(num("min_score", 0.6, float), 0.0), 1.0),
        min_len=15,
        max_len=min(max(num("max_len", 90, int), 15), 300),
        max_collection=5, timeout=60, retries=3,
    )
    with LOCK:
        JOBS[job_id] = {"status": "running", "stage": "Queued", "pct": 1}
    threading.Thread(target=worker, args=(job_id, video, srt, args), daemon=True).start()
    return jsonify(id=job_id)


@app.get("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(status="error", error="Job not found or expired."), 404
    return jsonify(job)


@app.get("/files/<job_id>/<path:name>")
def files(job_id, name):
    if job_id not in JOBS:
        return "Not found", 404
    return send_from_directory(os.path.join(BASE, job_id, "output"), name,
                               as_attachment=request.args.get("dl") == "1")


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>AutoClip</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--text:#16181d;--muted:#6a7280;--line:#dfe3ea;--accent:#4f46e5;--err:#c62828}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#1a1d24;--text:#eef0f4;--muted:#9aa3b2;--line:#2b303a;--accent:#818cf8;--err:#ef9a9a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
padding:max(16px,env(safe-area-inset-top)) max(16px,env(safe-area-inset-right)) max(24px,env(safe-area-inset-bottom)) max(16px,env(safe-area-inset-left))}
main{max-width:640px;margin:0 auto}
h1{font-size:1.6rem;margin:.2em 0}
.sub{color:var(--muted);margin:0 0 1rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:14px}
label{display:block;font-weight:600;margin:12px 0 6px}
input,button{font:inherit;width:100%;min-height:48px;border-radius:10px;border:1px solid var(--line);
background:var(--bg);color:var(--text);padding:10px 12px}
input[type=file]{padding:10px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
details{margin-top:12px}
summary{cursor:pointer;min-height:44px;display:flex;align-items:center;font-weight:600}
button.primary{background:var(--accent);color:#fff;border:0;font-weight:700;margin-top:16px}
button:disabled{opacity:.6}
.bar{height:12px;background:var(--line);border-radius:99px;overflow:hidden;margin:10px 0}
.bar>div{height:100%;width:0;background:var(--accent);transition:width .4s}
.muted{color:var(--muted);font-size:.9rem}
.err{color:var(--err);font-weight:600}
video{width:100%;border-radius:10px;background:#000;max-height:60vh}
.clip h3{margin:0 0 4px;font-size:1.05rem}
a.btn{display:flex;align-items:center;justify-content:center;min-height:48px;margin-top:8px;border-radius:10px;
border:1px solid var(--accent);color:var(--accent);text-decoration:none;font-weight:600}
.hidden{display:none}
</style></head><body><main>
<h1>AutoClip</h1>
<p class="sub">Upload a long video. Get shareable highlight clips.</p>

<form id="form" class="card">
  <label for="video">Video</label>
  <input id="video" name="video" type="file" accept="video/*" required>
  <label for="srt">Subtitles (.srt, optional)</label>
  <input id="srt" name="srt" type="file" accept=".srt">
  <p class="muted">Without subtitles, speech is transcribed automatically (needs faster-whisper on the server).</p>
  <details>
    <summary>AI settings</summary>
    <label for="base_url">API base URL</label>
    <input id="base_url" name="base_url" inputmode="url" autocapitalize="off" placeholder="http://localhost:11434/v1">
    <label for="model">Model</label>
    <input id="model" name="model" autocapitalize="off" placeholder="qwen2.5:7b">
    <label for="api_key">API key (not stored)</label>
    <input id="api_key" name="api_key" type="password" autocomplete="off">
    <div class="row">
      <div><label for="min_score">Min score</label>
        <input id="min_score" name="min_score" type="number" step="0.05" min="0" max="1" value="0.6" inputmode="decimal"></div>
      <div><label for="max_len">Max clip (sec)</label>
        <input id="max_len" name="max_len" type="number" min="15" max="300" value="90" inputmode="numeric"></div>
    </div>
  </details>
  <button class="primary" id="go" type="submit">Create clips</button>
</form>

<section id="progress" class="card hidden">
  <strong id="stage">Starting...</strong>
  <div class="bar"><div id="fill"></div></div>
  <div id="msg" class="muted">Keep this page open.</div>
</section>

<section id="results"></section>

<script>
const $ = id => document.getElementById(id);
const form = $("form");
let timer = null;

form.addEventListener("submit", e => {
  e.preventDefault();
  $("results").innerHTML = "";
  $("go").disabled = true;
  $("progress").classList.remove("hidden");
  $("msg").className = "muted";
  $("msg").textContent = "Uploading...";
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/start");
  xhr.upload.onprogress = ev => {
    if (ev.lengthComputable) {
      const p = Math.round(ev.loaded / ev.total * 100);
      $("stage").textContent = "Uploading " + p + "%";
      $("fill").style.width = Math.min(p, 100) * 0.05 + "%";
    }
  };
  xhr.onload = () => {
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status === 200 && data.id) { $("msg").textContent = "Keep this page open."; poll(data.id); }
    else fail(data.error || "Upload failed (file may be too large).");
  };
  xhr.onerror = () => fail("Network error during upload.");
  xhr.send(new FormData(form));
});

function fail(text) {
  clearInterval(timer);
  $("stage").textContent = "Failed";
  $("msg").className = "err";
  $("msg").textContent = text;
  $("go").disabled = false;
}

function poll(id) {
  timer = setInterval(async () => {
    try {
      const r = await fetch("/status/" + id);
      const j = await r.json();
      $("stage").textContent = j.stage || "Working...";
      $("fill").style.width = (j.pct || 0) + "%";
      if (j.status === "done") { clearInterval(timer); $("go").disabled = false; show(id, j); }
      if (j.status === "error") fail(j.error || "Something went wrong.");
    } catch (_) { /* transient network blip on mobile: keep polling */ }
  }, 2000);
}

function card(id, file, title, sub) {
  const url = "/files/" + id + "/" + encodeURIComponent(file);
  const d = document.createElement("div");
  d.className = "card clip";
  const h = document.createElement("h3"); h.textContent = title;
  const p = document.createElement("div"); p.className = "muted"; p.textContent = sub;
  const v = document.createElement("video");
  v.controls = true; v.playsInline = true; v.preload = "metadata"; v.src = url;
  const a = document.createElement("a");
  a.className = "btn"; a.href = url + "?dl=1"; a.textContent = "Download";
  d.append(h, p, v, a);
  return d;
}

function show(id, j) {
  const box = $("results");
  if (j.collection) box.append(card(id, j.collection, "Best-of collection", "Top clips merged into one video"));
  j.clips.forEach(c => box.append(card(id, c.file, c.title, c.start + " - " + c.end + "  |  score " + c.score)));
}
</script>
</main></body></html>"""


@app.get("/")
def index():
    return PAGE


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
