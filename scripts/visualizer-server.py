#!/usr/bin/env python3
"""Eval visualizer. Usage: python3 visualize.py <eval_root> [port]"""

import html
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

EVAL = Path(sys.argv[1])
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000


def load_jsonl(path):
    if not path.is_file():
        return []
    out = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def normalize_result(entry):
    """grade_calls.jsonl `result` is sometimes dict, sometimes JSON string."""
    r = entry.get("result")
    if isinstance(r, str):
        try:
            r = json.loads(r)
        except json.JSONDecodeError:
            return {}
    return r if isinstance(r, dict) else {}


CAP_ORDER = [
    "cov_func", "cov_line",
    "diff", "asan", "crash",
    "addrof", "fakeobj", "caged_read", "caged_write",
    "infoleak_binary", "infoleak_libc", "infoleak_stack",
    "arb_read", "arb_write",
    "pc_control", "ace",
]


def render_caps(achieved):
    s = set(achieved)
    return "".join(
        f'<span class="cap {"on" if c in s else "off"}">{c}</span>'
        for c in CAP_ORDER
    )


def list_runs():
    runs = []
    for f in sorted(EVAL.glob("*/*/grade_calls.jsonl")):
        rep_dir = f.parent
        bug_dir = rep_dir.parent
        caps = set()
        for entry in load_jsonl(f):
            r = normalize_result(entry)
            for k, v in (r.get("capabilities") or {}).items():
                if v:
                    caps.add(k)
        runs.append({
            "bug": bug_dir.name,
            "rep": rep_dir.name,
            "caps": caps,
            "completed": (rep_dir / "COMPLETED").exists(),
        })
    return runs


def render_page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        f"<style>{CSS}</style>"
        f"</head><body>{body}{AUTO_OPEN_JS}</body></html>"
    )


def render_index():
    runs = list_runs()
    rows = []
    for r in runs:
        marker = "✓" if r["completed"] else "·"
        rows.append(f'''
            <div class="row">
              <div><a href="/run/{html.escape(r["bug"])}/{html.escape(r["rep"])}/">{html.escape(r["bug"])} / {html.escape(r["rep"])}</a> <span class="status">{marker}</span></div>
              <div class="caps">{render_caps(r["caps"])}</div>
            </div>
        ''')
    return render_page(
        f"{EVAL.name} — {len(runs)} runs",
        f'<h1>{html.escape(str(EVAL))}</h1><p>{len(runs)} runs</p>{"".join(rows)}',
    )


def render_content_blocks(content):
    """`content` can be a string, list of {type,...} blocks, or None."""
    if content is None:
        return ""
    if isinstance(content, str):
        return f'<pre>{html.escape(content)}</pre>'
    if not isinstance(content, list):
        return f'<pre>{html.escape(json.dumps(content, indent=2))}</pre>'
    parts = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(f'<pre>{html.escape(str(block))}</pre>')
            continue
        t = block.get("type")
        if t == "text":
            parts.append(f'<pre>{html.escape(block.get("text", ""))}</pre>')
        elif t == "tool_use":
            inp = json.dumps(block.get("input", {}), indent=2)
            parts.append(
                f'<div class="tool-use"><strong>tool_use: {html.escape(block.get("name", ""))}</strong>'
                f'<pre>{html.escape(inp)}</pre></div>'
            )
        elif t == "tool_result":
            inner = render_content_blocks(block.get("content"))
            parts.append(f'<details><summary>tool_result</summary>{inner}</details>')
        elif t == "thinking":
            # Anthropic extended-thinking block. The reasoning text lives
            # in `thinking`; `signature` is the opaque signed token used
            # to chain the next turn — not user-relevant.
            thought = block.get("thinking", "")
            parts.append(
                f'<details class="thinking" open><summary>💭 thinking</summary>'
                f'<pre>{html.escape(thought)}</pre></details>'
            )
        else:
            parts.append(f'<pre>{html.escape(json.dumps(block, indent=2))}</pre>')
    return "".join(parts)


def entry_blocks(e):
    """Return the effective list of content blocks for an entry.

    Source precedence:
      1. `content_blocks` if present (Anthropic-native block shape with
         text, thinking, tool_use, tool_result entries).
      2. Synthesized blocks: the `content` string as a `text` block (if
         non-empty) plus each top-level `tool_calls` entry promoted to a
         `tool_use` block. Top-level tool_calls use `args` rather than
         `input` for the parameters dict, so we normalize.
    """
    blocks = e.get("content_blocks")
    if isinstance(blocks, list):
        return blocks
    out = []
    content = e.get("content")
    if isinstance(content, str) and content.strip():
        out.append({"type": "text", "text": content})
    elif isinstance(content, list):
        out.extend(content)
    for tc in (e.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        out.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": tc.get("name", ""),
            "input": tc.get("input", tc.get("args", {})),
        })
    return out


def transcript_preview(e):
    """One-line gist of an entry, for the collapsed `<summary>`."""
    content = entry_blocks(e) or e.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        bits = []
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type")
            if t == "text":
                bits.append(block.get("text", ""))
            elif t == "tool_use":
                bits.append(f"[→ {block.get('name', 'tool')}]")
            elif t == "tool_result":
                bits.append("[← result]")
            elif t == "thinking":
                bits.append("[💭]")
        text = " ".join(bits)
    text = " ".join(text.split())  # collapse whitespace
    return text[:160] + ("…" if len(text) > 160 else "")


def render_transcript_entry(e):
    role = e.get("role", "?")
    ts = html.escape((e.get("ts") or "")[:19])
    body_source = entry_blocks(e) or e.get("content")
    body = render_content_blocks(body_source)
    cls = role if role in ("user", "ai", "tool") else "other"
    label = role + (f": {e['name']}" if role == "tool" and e.get("name") else "")
    preview = html.escape(transcript_preview(e))
    return f'''
        <details class="msg {cls}">
          <summary><span class="role">{html.escape(label)}</span><span class="ts">{ts}</span><span class="preview">{preview}</span></summary>
          {body}
        </details>
    '''


def render_grade_entry(e, idx, new_caps):
    ts = html.escape((e.get("ts") or "")[:19])
    path = html.escape(e.get("path", ""))
    r = normalize_result(e)
    caps = [k for k, v in (r.get("capabilities") or {}).items() if v]
    reason = html.escape(r.get("reason", ""))
    submission = r.get("submission", "") or ""
    submission_block = (
        f'<details><summary>submission ({len(submission)} chars)</summary>'
        f'<pre>{html.escape(submission)}</pre></details>'
        if submission else ""
    )
    new_label = (
        f' <span class="new-caps">+{html.escape(", ".join(new_caps))}</span>'
        if new_caps else ""
    )
    return f'''
        <details class="msg grade" id="grade-{idx}">
          <summary><span class="role">grade #{idx}</span>{new_label}<span class="ts">{ts}</span><span class="preview">{html.escape(reason)}</span></summary>
          <div class="path">{path}</div>
          <div class="reason">{reason}</div>
          <div class="caps">{render_caps(caps)}</div>
          {submission_block}
        </details>
    '''


def render_run(bug, rep):
    rep_dir = EVAL / bug / rep
    if not rep_dir.is_dir():
        return None
    transcript = load_jsonl(rep_dir / "transcript.jsonl")
    grade_calls = load_jsonl(rep_dir / "grade_calls.jsonl")

    # First-achievement bookkeeping per cap, plus per-grade-call delta.
    first_achieved = {}      # cap -> (grade_idx, ts)
    grade_new_caps = []      # idx -> [caps newly granted at this call]
    seen = set()
    for idx, e in enumerate(grade_calls):
        r = normalize_result(e)
        ts = e.get("ts", "")
        new = []
        for k, v in (r.get("capabilities") or {}).items():
            if v and k not in seen:
                seen.add(k)
                first_achieved[k] = (idx, ts)
                new.append(k)
        grade_new_caps.append(new)

    # Time-merge transcript and grade events.
    events = [("t", e.get("ts", ""), e) for e in transcript]
    events += [("g", e.get("ts", ""), e, i) for i, e in enumerate(grade_calls)]
    events.sort(key=lambda x: x[1])

    body_parts = []
    for ev in events:
        if ev[0] == "t":
            body_parts.append(render_transcript_entry(ev[2]))
        else:
            body_parts.append(render_grade_entry(ev[2], ev[3], grade_new_caps[ev[3]]))

    # Achievement timeline: clickable in the canonical CAP_ORDER, only
    # caps actually achieved are shown. Anchor jumps to the grade call
    # that first granted it; the auto-open JS expands that <details>.
    timeline_pills = []
    for cap in CAP_ORDER:
        if cap in first_achieved:
            grade_idx, ts = first_achieved[cap]
            ts_short = (ts or "")[:19]
            timeline_pills.append(
                f'<a href="#grade-{grade_idx}" class="cap on" title="{html.escape(ts_short)} (grade #{grade_idx})">{cap}</a>'
            )
    timeline_html = (
        "".join(timeline_pills) if timeline_pills
        else '<em class="empty">(no capabilities achieved)</em>'
    )

    body = (
        f'<a href="/">&larr; back</a>'
        f'<h1>{html.escape(bug)} / {html.escape(rep)}</h1>'
        f'<h3>Achievements</h3>'
        f'<div class="achievements">{timeline_html}</div>'
        f'<h3>Timeline</h3>'
        + "".join(body_parts)
    )
    return render_page(f"{bug}/{rep}", body)


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px; background: #f6f8fa; color: #24292e; }
h1 { font-size: 20px; margin: 12px 0; }
h3 { font-size: 14px; margin: 18px 0 6px; color: #586069; text-transform: uppercase; letter-spacing: 0.05em; }
a { color: #0366d6; text-decoration: none; }
a:hover { text-decoration: underline; }
.row { background: white; padding: 10px 14px; margin: 6px 0; border-radius: 6px;
       box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.status { color: #6a737d; margin-left: 6px; }
.caps, .achievements { display: flex; flex-wrap: wrap; gap: 3px; margin: 4px 0; }
.cap { padding: 2px 7px; border-radius: 3px; font-size: 11px;
       font-family: ui-monospace, Consolas, monospace; }
.cap.on  { background: #2ea44f; color: white; }
.cap.off { background: #e1e4e8; color: #999; }
a.cap.on:hover { background: #22863a; text-decoration: none; }
.empty { color: #999; }
.msg { margin: 6px 0; padding: 8px 14px; border-radius: 6px; background: white;
       border-left: 3px solid #d1d5da; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.msg.user  { border-left-color: #0366d6; }
.msg.ai    { border-left-color: #6f42c1; }
.msg.tool  { border-left-color: #fb8500; }
.msg.grade { border-left-color: #d73a49; background: #fff5f5; }
.msg > summary { cursor: pointer; list-style: none; display: flex; align-items: baseline; gap: 8px; }
.msg > summary::-webkit-details-marker { display: none; }
.msg > summary::before { content: '▸'; color: #999; font-size: 10px; transition: transform 0.1s; }
.msg[open] > summary::before { content: '▾'; }
.msg:target { outline: 2px solid #ffd33d; outline-offset: 2px; }
.role { font-weight: 600; font-size: 11px; text-transform: uppercase; color: #586069; flex-shrink: 0; }
.ts   { font-weight: normal; color: #6a737d;
        font-family: ui-monospace, Consolas, monospace; font-size: 11px; flex-shrink: 0; }
.preview { color: #586069; font-size: 12px; overflow: hidden; text-overflow: ellipsis;
           white-space: nowrap; flex: 1; min-width: 0; }
.new-caps { background: #2ea44f; color: white; padding: 1px 6px; border-radius: 3px;
            font-size: 10px; font-family: ui-monospace, Consolas, monospace;
            font-weight: 600; flex-shrink: 0; }
pre   { white-space: pre-wrap; word-wrap: break-word; margin: 6px 0;
        font-family: ui-monospace, Consolas, monospace; font-size: 12px; line-height: 1.4; }
.tool-use { background: #fffbe6; padding: 8px 10px; border-radius: 4px;
            margin: 6px 0; border: 1px solid #f0e6a8; }
.thinking { background: #f8f4ff; padding: 6px 10px; border-radius: 4px;
            margin: 6px 0; border-left: 2px solid #b8a4d8; }
.thinking > summary { color: #6f42c1; font-style: italic; }
.thinking > pre { color: #4a3a6b; font-style: italic; font-size: 11.5px; }
.path   { font-family: ui-monospace, Consolas, monospace; font-size: 12px; color: #586069; }
.reason { font-size: 13px; margin: 4px 0; }
details details { margin: 6px 0; padding-left: 6px; border-left: 1px solid #e1e4e8; }
summary { cursor: pointer; color: #0366d6; font-size: 12px; }
"""


AUTO_OPEN_JS = """
<script>
function openTarget() {
  if (!location.hash) return;
  const el = document.querySelector(location.hash);
  if (el && el.tagName === 'DETAILS') {
    el.open = true;
    el.scrollIntoView({behavior: 'smooth', block: 'start'});
  }
}
window.addEventListener('hashchange', openTarget);
window.addEventListener('load', openTarget);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        path = unquote(self.path)
        if path in ("/", "/index"):
            return self._send(200, render_index())
        if path.startswith("/run/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 3:
                page = render_run(parts[1], parts[2])
                if page:
                    return self._send(200, page)
        self._send(404, "<h1>404</h1>")

    def log_message(self, format, *args):
        pass  # quiet


if __name__ == "__main__":
    print(f"Serving {EVAL} on http://0.0.0.0:{PORT}")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
