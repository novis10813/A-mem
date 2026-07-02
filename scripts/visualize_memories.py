#!/usr/bin/env python3
"""Render A-MEM cached memory pickle files as an HTML report."""

import argparse
import pickle
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_samples(sample_text: str) -> list[int]:
    samples: list[int] = []
    for part in sample_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            samples.extend(range(start, end + 1))
        else:
            samples.append(int(part))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize A-MEM cached memory notes")
    parser.add_argument(
        "--cache-dir",
        default="artifacts/caches/cached_memories_robust_ollama_llama3.2:latest",
        help="Path to cached memory directory, relative to repo root unless absolute",
    )
    parser.add_argument(
        "--output",
        default="artifacts/output/memory_viewer.html",
        help="Output HTML path, relative to repo root unless absolute",
    )
    parser.add_argument(
        "--samples",
        default="0-9",
        help='Sample indexes to load, for example "0-9" or "0,3,5"',
    )
    return parser.parse_args()


def load_sample(cache_dir: Path, sample_idx: int) -> dict:
    pkl_path = cache_dir / f"memory_cache_sample_{sample_idx}.pkl"
    with open(pkl_path, "rb") as f:
        return SafeUnpickler(f).load()


class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state: dict):
        self.__dict__.update(state)


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Stub

def note_to_dict(note) -> dict:
    if hasattr(note, '__dict__'):
        d = note.__dict__.copy()
        # convert lists/etc to serializable
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                d[k] = list(v)
        return d
    return {}

def build_html(all_samples: list[tuple[int, dict]]) -> str:
    # gather stats
    total_notes = sum(len(s) for _, s in all_samples)
    
    sample_sections = []
    for sample_idx, notes in all_samples:
        note_cards = []
        for i, (nid, note) in enumerate(list(notes.items())[:12]):  # show first 12 per sample
            d = note_to_dict(note)
            content = d.get("content", "")[:300]
            context = d.get("context", "")[:250]
            keywords = d.get("keywords", [])
            tags = d.get("tags", [])[:8]
            links = d.get("links", [])
            retrieval = d.get("retrieval_count", 0)
            timestamp = d.get("timestamp", "")
            evolution = d.get("evolution_history", [])
            evo_count = len(evolution)

            kw_html = "".join(f'<span class="kw-chip">{k}</span>' for k in keywords)
            tag_html = "".join(f'<span class="tag-chip">{t}</span>' for t in tags)

            note_cards.append(f"""
            <div class="note-card" data-sample="{sample_idx}" data-note="{i}">
              <div class="note-header">
                <div class="note-id">#{i+1}</div>
                <div class="note-meta">
                  {'<span class="badge badge-evo">🔄 Evolved ×' + str(evo_count) + '</span>' if evo_count else ''}
                </div>
              </div>
              <div class="note-content">{content}</div>
              <div class="note-context">
                <span class="label">Context:</span> {context}
              </div>
              <div class="chips-row">{kw_html}</div>
              <div class="chips-row tags-row">{tag_html}</div>
              <div class="note-footer">
                <div class="footer-stats">
                  <span>🔗 Links: {len(links)}</span>
                  <span>👁 Retrievals: {retrieval}</span>
                  <span>🕐 {timestamp}</span>
                </div>
              </div>
            </div>""")

        more = len(notes) - 12
        more_note = f'<div class="more-note">… and {more} more memory notes in this sample</div>' if more > 0 else ""
        
        sample_sections.append(f"""
        <section class="sample-section">
          <div class="sample-header">
            <h2>Sample {sample_idx}</h2>
            <div class="sample-stats">
              <span class="stat-pill">💾 {len(notes)} memories</span>
            </div>
          </div>
          <div class="notes-grid">
            {"".join(note_cards)}
          </div>
          {more_note}
        </section>""")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A-MEM Memory Viewer — Llama 3.2 (Ollama)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #21262d;
    --border: #30363d;
    --accent: #7c3aed;
    --accent2: #2563eb;
    --accent3: #059669;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --text-subtle: #6e7681;
    --kw: #7c3aed22;
    --kw-border: #7c3aed66;
    --kw-text: #c4b5fd;
    --tag: #05966922;
    --tag-border: #05966966;
    --tag-text: #6ee7b7;
    --radius: 12px;
    --radius-sm: 6px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding-bottom: 60px;
  }}

  /* ── Header ── */
  .site-header {{
    background: linear-gradient(135deg, #0d1117 0%, #1a0a2e 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 40px 48px 32px;
    position: relative;
    overflow: hidden;
  }}
  .site-header::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 30% 50%, #7c3aed18 0%, transparent 70%),
                radial-gradient(ellipse at 80% 20%, #2563eb12 0%, transparent 60%);
  }}
  .header-inner {{ position: relative; max-width: 1400px; margin: 0 auto; }}
  .header-badge {{
    display: inline-flex; align-items: center; gap: 8px;
    background: #7c3aed20; border: 1px solid #7c3aed44;
    color: #c4b5fd; font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 16px;
    text-transform: uppercase; letter-spacing: 0.08em;
  }}
  .header-badge .dot {{ width:6px; height:6px; border-radius:50%; background:#7c3aed; animation: pulse 2s infinite; }}
  @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.4}} }}
  h1 {{
    font-size: 36px; font-weight: 700; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e6edf3 0%, #c4b5fd 50%, #7c3aed 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }}
  .header-sub {{ color: var(--text-muted); font-size: 15px; margin-bottom: 24px; }}
  .header-stats {{
    display: flex; gap: 24px; flex-wrap: wrap;
  }}
  .hstat {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 12px 20px;
    display: flex; flex-direction: column; gap: 2px;
  }}
  .hstat .val {{ font-size: 22px; font-weight: 700; color: var(--text); }}
  .hstat .lbl {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }}

  /* ── Nav tabs ── */
  .nav-bar {{
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 0 48px; display: flex; gap: 0; overflow-x: auto;
    position: sticky; top: 0; z-index: 100;
  }}
  .nav-tab {{
    padding: 14px 20px; font-size: 14px; font-weight: 500;
    color: var(--text-muted); cursor: pointer; border: none;
    background: none; border-bottom: 2px solid transparent;
    transition: all .2s; white-space: nowrap;
  }}
  .nav-tab:hover {{ color: var(--text); }}
  .nav-tab.active {{ color: #c4b5fd; border-bottom-color: #7c3aed; }}

  /* ── Main content ── */
  .main {{ max-width: 1400px; margin: 0 auto; padding: 32px 48px; }}

  .sample-section {{ margin-bottom: 56px; }}
  .sample-header {{
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px; padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }}
  .sample-header h2 {{
    font-size: 20px; font-weight: 700;
    background: linear-gradient(90deg, #e6edf3, #8b949e);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .sample-stats {{ display: flex; gap: 8px; }}
  .stat-pill {{
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--text-muted); font-size: 13px; font-weight: 500;
    padding: 4px 12px; border-radius: 20px;
  }}

  /* ── Note grid ── */
  .notes-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 16px;
  }}
  .note-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    display: flex; flex-direction: column; gap: 10px;
    transition: border-color .2s, transform .2s, box-shadow .2s;
    animation: fadeIn .4s ease both;
  }}
  .note-card:hover {{
    border-color: #7c3aed55;
    transform: translateY(-2px);
    box-shadow: 0 8px 32px #7c3aed18;
  }}
  @keyframes fadeIn {{ from{{opacity:0;transform:translateY(8px)}} to{{opacity:1;transform:none}} }}

  .note-header {{ display: flex; align-items: center; justify-content: space-between; }}
  .note-id {{ font-size: 13px; font-weight: 600; color: var(--text-muted); }}
  .note-meta {{ display: flex; gap: 6px; align-items: center; }}
  .badge {{
    font-size: 11px; font-weight: 600; padding: 2px 8px;
    border-radius: 20px; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .badge-evo {{ background: #f59e0b18; border: 1px solid #f59e0b44; color: #fcd34d; }}

  .note-content {{
    font-size: 14px; line-height: 1.6; color: var(--text);
    background: var(--surface2); border-radius: var(--radius-sm);
    padding: 10px 12px; border-left: 3px solid #7c3aed;
  }}
  .note-context {{
    font-size: 13px; line-height: 1.5; color: var(--text-muted);
    font-style: italic;
  }}
  .note-context .label {{ font-style: normal; font-weight: 600; color: var(--text-subtle); }}

  .chips-row {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .kw-chip {{
    font-size: 12px; font-weight: 500;
    background: var(--kw); border: 1px solid var(--kw-border); color: var(--kw-text);
    padding: 2px 10px; border-radius: 20px;
  }}
  .tag-chip {{
    font-size: 11px;
    background: var(--tag); border: 1px solid var(--tag-border); color: var(--tag-text);
    padding: 2px 9px; border-radius: 20px;
  }}

  .note-footer {{ display: flex; flex-direction: column; gap: 8px; margin-top: 2px; }}
  .label {{ font-size: 11px; font-weight: 600; color: var(--text-subtle); text-transform: uppercase; letter-spacing: 0.06em; white-space: nowrap; }}
  .footer-stats {{ display: flex; gap: 14px; flex-wrap: wrap; }}
  .footer-stats span {{ font-size: 12px; color: var(--text-subtle); }}
  .more-note {{
    text-align: center; color: var(--text-subtle); font-size: 13px;
    padding: 12px; border: 1px dashed var(--border); border-radius: var(--radius);
    margin-top: 8px;
  }}

  /* Sample visibility */
  .sample-section.hidden {{ display: none; }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  /* Footer */
  .site-footer {{
    text-align: center; color: var(--text-subtle); font-size: 12px;
    margin-top: 60px; padding: 20px;
    border-top: 1px solid var(--border);
  }}
</style>
</head>
<body>

<header class="site-header">
  <div class="header-inner">
    <div class="header-badge">
      <span class="dot"></span>
      A-MEM Experiment Results
    </div>
    <h1>Agentic Memory Viewer</h1>
    <p class="header-sub">
      Model: <strong>ollama / llama3.2:latest</strong> &nbsp;·&nbsp;
      Generated: {now}
    </p>
    <div class="header-stats">
      <div class="hstat"><span class="val">{total_notes}</span><span class="lbl">Total Memory Notes</span></div>
      <div class="hstat"><span class="val">{len(all_samples)}</span><span class="lbl">Conversation Samples</span></div>
      <div class="hstat"><span class="val">RobustMemoryNote</span><span class="lbl">Note Type</span></div>
      <div class="hstat"><span class="val">Zettelkasten</span><span class="lbl">Architecture</span></div>
    </div>
  </div>
</header>

<nav class="nav-bar" id="navBar">
  <button class="nav-tab active" data-target="all" onclick="filterSample(this,'all')">All Samples</button>
  {"".join(f'<button class="nav-tab" data-target="{i}" onclick="filterSample(this,{i})">Sample {i}</button>' for i, _ in all_samples)}
</nav>

<main class="main">
  {"".join(sample_sections)}
</main>

<footer class="site-footer">
  A-MEM: Agentic Memory System &nbsp;·&nbsp; Paper: arXiv 2502.12110 &nbsp;·&nbsp; Experiment: LoCoMo Benchmark
</footer>

<script>
function filterSample(btn, target) {{
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.sample-section').forEach(s => {{
    if (target === 'all') {{ s.classList.remove('hidden'); }}
    else {{ s.classList.toggle('hidden', s.querySelector('h2').textContent.trim() !== 'Sample ' + target); }}
  }});
}}
// stagger card animations
document.querySelectorAll('.note-card').forEach((c, i) => {{
  c.style.animationDelay = (i % 12 * 0.04) + 's';
}});
</script>

</body>
</html>"""

def main():
    args = parse_args()
    cache_dir = repo_path(args.cache_dir)
    output_html = repo_path(args.output)
    samples = parse_samples(args.samples)

    print("📦 Loading memory caches...")
    all_samples = []
    for i in samples:
        try:
            notes = load_sample(cache_dir, i)
            all_samples.append((i, notes))
            print(f"  ✓ Sample {i}: {len(notes)} memory notes")
        except Exception as e:
            print(f"  ✗ Sample {i}: {e}")

    print(f"\n📊 Total notes across all samples: {sum(len(s) for _, s in all_samples)}")
    print("🎨 Building HTML report...")
    html = build_html(all_samples)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    print(f"✅ Saved to: {output_html}")

if __name__ == "__main__":
    main()
