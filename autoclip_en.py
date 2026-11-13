#!/usr/bin/env python3
"""
autoclip_en.py - minimal English AutoClip-style pipeline (single file, no pip deps).

Pipeline: video (+ SRT) -> chunk transcript -> LLM finds highlights -> score filter
          -> ffmpeg cuts clips -> optional merged collection.

Works with any OpenAI-compatible API (OpenAI, Ollama, LM Studio, etc.).

Examples:
  # Local Ollama
  python autoclip_en.py talk.mp4 --srt talk.srt

  # OpenAI
  export LLM_API_KEY=sk-...
  python autoclip_en.py talk.mp4 --srt talk.srt \
      --base-url https://api.openai.com/v1 --model gpt-4o-mini

Requires: Python 3.9+, ffmpeg on PATH. Optional: faster-whisper (if no --srt).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

SYSTEM_PROMPT = """You are an expert video editor. You receive a timestamped transcript chunk.
Find the most engaging, self-contained highlight segments (strong insights, stories,
surprising facts, emotional moments, clear explanations).

Rules:
- Each highlight must start and end on natural sentence boundaries.
- Use ONLY timestamps that appear in the transcript.
- Each highlight should be {min_len}-{max_len} seconds long.
- Score each from 0.0 to 1.0 (how shareable it is on its own).
- Write a short, catchy English title (max 10 words).

Return ONLY valid JSON, no markdown:
{{"highlights": [{{"start": "HH:MM:SS", "end": "HH:MM:SS", "title": "...", "score": 0.0, "reason": "..."}}]}}
If there is nothing worth clipping, return {{"highlights": []}}."""


# ---------- time helpers ----------
def ts_to_sec(ts):
    h, m, s = ts.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def sec_to_ts(sec):
    sec = max(0, int(sec))
    return "%02d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)


# ---------- subtitles ----------
SRT_RE = re.compile(
    r"(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n(.*?)(?=\n\s*\n|\Z)",
    re.S,
)


def parse_srt(path):
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().replace("\r\n", "\n")
    cues = []
    for m in SRT_RE.finditer(text + "\n"):
        body = " ".join(m.group(4).split())
        if body:
            cues.append({"start": ts_to_sec(m.group(2)), "end": ts_to_sec(m.group(3)), "text": body})
    return cues


def transcribe(video, model_size="small"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("No --srt given and faster-whisper is not installed: pip install faster-whisper")
    print("Transcribing (first run downloads the model)...")
    model = WhisperModel(model_size, compute_type="int8")
    segments, _ = model.transcribe(video)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def chunk_cues(cues, chunk_chars):
    chunks, cur, size = [], [], 0
    for c in cues:
        line = "[%s] %s" % (sec_to_ts(c["start"]), c["text"])
        if size + len(line) > chunk_chars and cur:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


# ---------- LLM ----------
def call_llm(base_url, api_key, model, system, user, timeout):
    body = json.dumps({
        "model": model,
        "temperature": 0.2,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", body, headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1:
        return {"highlights": []}
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return {"highlights": []}


def find_highlights(chunks, args, duration, progress=None):
    system = SYSTEM_PROMPT.format(min_len=args.min_len, max_len=args.max_len)
    found = []
    for n, chunk in enumerate(chunks, 1):
        print("Analyzing chunk %d/%d..." % (n, len(chunks)))
        if progress:
            progress(n, len(chunks))
        for attempt in range(args.retries):
            try:
                raw = call_llm(args.base_url, args.api_key, args.model, system, chunk, args.timeout)
                data = extract_json(raw)
                break
            except Exception as e:  # network / timeout / bad response
                print("  retry %d: %s" % (attempt + 1, e))
                data = {"highlights": []}
        for h in data.get("highlights", []):
            try:
                s, e = ts_to_sec(h["start"]), ts_to_sec(h["end"])
                score = float(h.get("score", 0))
            except (KeyError, ValueError):
                continue
            e = min(e, duration) if duration else e
            if score >= args.min_score and args.min_len <= e - s <= args.max_len * 1.5:
                found.append({"start": s, "end": e, "score": score,
                              "title": str(h.get("title", "Untitled"))[:80]})
    # drop overlaps, keeping the higher score
    found.sort(key=lambda x: -x["score"])
    kept = []
    for h in found:
        if all(h["end"] <= k["start"] or h["start"] >= k["end"] for k in kept):
            kept.append(h)
    return sorted(kept, key=lambda x: x["start"])


# ---------- ffmpeg ----------
def video_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:50] or "clip"


def cut_clip(video, h, out_path):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(h["start"]), "-i", video,
           "-t", str(h["end"] - h["start"]), "-c:v", "libx264", "-preset", "fast",
           "-crf", "22", "-c:a", "aac", out_path]
    subprocess.run(cmd, check=True)


def make_collection(paths, out_path):
    list_file = out_path + ".txt"
    with open(list_file, "w") as f:
        for p in paths:
            f.write("file '%s'\n" % os.path.abspath(p).replace("'", "'\\''"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", list_file, "-c", "copy", out_path], check=True)
    os.remove(list_file)


# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="English AutoClip-style highlight clipper")
    p.add_argument("video")
    p.add_argument("--srt", help="subtitle file; if omitted, faster-whisper is used")
    p.add_argument("--out", default="output")
    p.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    p.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "qwen2.5:7b"))
    p.add_argument("--chunk-chars", type=int, default=5000)
    p.add_argument("--min-score", type=float, default=0.6)
    p.add_argument("--min-len", type=int, default=15, help="min clip seconds")
    p.add_argument("--max-len", type=int, default=90, help="max clip seconds")
    p.add_argument("--max-collection", type=int, default=5, help="clips in merged collection")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cues = parse_srt(args.srt) if args.srt else transcribe(args.video)
    if not cues:
        sys.exit("No transcript content found.")

    highlights = find_highlights(chunk_cues(cues, args.chunk_chars), args, video_duration(args.video))
    if not highlights:
        sys.exit("No highlights found. Try a lower --min-score (e.g. 0.5).")

    paths = []
    for i, h in enumerate(highlights, 1):
        path = os.path.join(args.out, "%02d_%s.mp4" % (i, safe_name(h["title"])))
        print("Cutting %s-%s  %s (score %.2f)" % (sec_to_ts(h["start"]), sec_to_ts(h["end"]), h["title"], h["score"]))
        cut_clip(args.video, h, path)
        paths.append(path)

    top = sorted(zip(highlights, paths), key=lambda x: -x[0]["score"])[:args.max_collection]
    top_paths = [pp for _, pp in sorted(top, key=lambda x: x[0]["start"])]
    if len(top_paths) > 1:
        make_collection(top_paths, os.path.join(args.out, "collection.mp4"))

    with open(os.path.join(args.out, "highlights.json"), "w", encoding="utf-8") as f:
        json.dump(highlights, f, indent=2, ensure_ascii=False)
    print("Done. %d clips saved in %s/" % (len(paths), args.out))


if __name__ == "__main__":
    main()
