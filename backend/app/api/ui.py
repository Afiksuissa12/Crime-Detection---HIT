from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Crime Detection - Test Console</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; padding: 24px; }

  h1   { font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
  .sub { font-size: 0.78rem; color: #475569; margin-bottom: 24px; }

  /* ── button row ── */
  .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
  button {
    padding: 8px 16px; border: none; border-radius: 6px;
    font-size: 0.82rem; font-weight: 600; cursor: pointer; transition: opacity .15s;
  }
  button:hover   { opacity: .8; }
  button:disabled { opacity: .35; cursor: not-allowed; }
  .btn-health   { background:#334155; color:#94a3b8; }
  .btn-violence { background:#7f1d1d; color:#fca5a5; }
  .btn-normal   { background:#14532d; color:#86efac; }
  .btn-stop     { background:#1e293b; color:#94a3b8; }
  .btn-status   { background:#1e3a5f; color:#93c5fd; }
  .btn-alerts   { background:#2d1b69; color:#c4b5fd; }
  .btn-upload   { background:#1a3040; color:#67e8f9; }
  .btn-run      { background:#0c2a1a; color:#6ee7b7; }

  /* ── drop zone ── */
  .drop-zone {
    border: 2px dashed #2d3748; border-radius: 10px;
    padding: 24px; text-align: center; cursor: pointer;
    color: #475569; font-size: 0.82rem; margin-bottom: 16px;
    transition: border-color .2s, background .2s;
  }
  .drop-zone.hover { border-color: #67e8f9; background: #0f2030; color: #a5f3fc; }
  .drop-zone .icon { font-size: 2rem; margin-bottom: 6px; }
  .drop-zone .filename { margin-top: 8px; color: #67e8f9; font-weight: 600; font-size: 0.8rem; }
  .progress { height: 4px; background: #1e293b; border-radius: 2px; margin-bottom: 16px; overflow: hidden; }
  .progress-bar { height: 100%; background: #67e8f9; width: 0; transition: width .3s; }
  #upload-msg { font-size: 0.75rem; color: #64748b; margin-bottom: 16px; min-height: 18px; }

  /* ── cards ── */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); gap: 14px; }
  .card { background: #1e2535; border: 1px solid #2d3748; border-radius: 10px; padding: 16px; }
  .card-title { font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: .08em; color: #475569; margin-bottom: 12px; }
  .field { display: flex; justify-content: space-between; align-items: flex-start;
           padding: 5px 0; border-bottom: 1px solid #1a2030; font-size: 0.82rem; }
  .field:last-child { border-bottom: none; }
  .field-key { color: #94a3b8; flex-shrink: 0; margin-right: 8px; }
  .field-val { font-weight: 600; color: #f1f5f9; text-align: right; word-break: break-word; }
  .empty { color: #334155; font-size: 0.8rem; text-align: center; padding: 10px 0; }

  /* ── badges ── */
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.7rem; font-weight:700; }
  .b-red    { background:#450a0a; color:#fca5a5; }
  .b-green  { background:#052e16; color:#86efac; }
  .b-blue   { background:#0c1a2e; color:#93c5fd; }
  .b-gray   { background:#1e293b; color:#94a3b8; }
  .b-orange { background:#431407; color:#fdba74; }

  /* ── alerts ── */
  .alerts-list { display:flex; flex-direction:column; gap:8px; }
  .alert-row { background:#151c2b; border:1px solid #2d3748; border-radius:6px; padding:10px 12px; }
  .alert-row .ah { display:flex; justify-content:space-between; margin-bottom:4px; }
  .alert-row .am { color:#64748b; font-size:.72rem; }

  /* ── ticker ── */
  #ticker { font-size:.72rem; color:#334155; margin-bottom:14px; min-height:16px; }

  /* hidden file input */
  #file-input { display:none; }
</style>
</head>
<body>

<h1>AI Crime Detection — Test Console</h1>
<p class="sub">Testing interface &nbsp;·&nbsp; <a href="/docs" style="color:#475569">/docs</a> &nbsp;·&nbsp; <a href="/redoc" style="color:#475569">/redoc</a></p>

<!-- ── existing controls ── -->
<div class="controls">
  <button class="btn-health"   onclick="checkHealth()">Check Health</button>
  <button class="btn-violence" onclick="startPreset('violence')">Start Violence Test Video</button>
  <button class="btn-normal"   onclick="startPreset('normal')">Start No-Violence Test Video</button>
  <button class="btn-stop"     onclick="stopVideo()">Stop Video</button>
  <button class="btn-status"   onclick="fetchStatus()">Get Status</button>
  <button class="btn-alerts"   onclick="fetchAlerts()">Get Alerts</button>
</div>

<!-- ── upload area ── -->
<div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
  <div class="icon">&#8681;</div>
  <div>Drag &amp; drop a video here or click to upload</div>
  <div class="filename" id="drop-label"></div>
</div>
<input type="file" id="file-input" accept=".mp4,.avi,.mov,.mkv">
<div class="progress"><div class="progress-bar" id="progress-bar"></div></div>
<div id="upload-msg"></div>

<div class="controls">
  <button class="btn-upload" id="btn-upload" onclick="uploadFile()" disabled>Upload Video</button>
  <button class="btn-run"    id="btn-run"    onclick="runUploaded()" disabled>Run Uploaded Video Test</button>
</div>

<div id="ticker"></div>

<!-- ── data cards ── -->
<div class="grid">
  <div class="card" id="card-health">
    <div class="card-title">System Health</div>
    <div class="empty">Press "Check Health"</div>
  </div>
  <div class="card" id="card-status">
    <div class="card-title">Video Status</div>
    <div class="empty">Press "Get Status"</div>
  </div>
  <div class="card" id="card-ai">
    <div class="card-title">Latest AI Result</div>
    <div class="empty">No data yet</div>
  </div>
  <div class="card" id="card-event">
    <div class="card-title">Latest Event Decision</div>
    <div class="empty">No data yet</div>
  </div>
  <div class="card" id="card-alerts" style="grid-column:1/-1">
    <div class="card-title">Alerts</div>
    <div class="empty">Press "Get Alerts"</div>
  </div>
</div>

<script>
// ── state ────────────────────────────────────────────────────────────────────
let _pollTimer   = null;
let _selectedFile = null;
let _uploadedSource = null;

// ── card titles ──────────────────────────────────────────────────────────────
const TITLES = {
  'card-health':'System Health', 'card-status':'Video Status',
  'card-ai':'Latest AI Result', 'card-event':'Latest Event Decision',
  'card-alerts':'Alerts',
};
function setCard(id, html) {
  document.getElementById(id).innerHTML =
    `<div class="card-title">${TITLES[id]}</div>${html}`;
}

// ── helpers ──────────────────────────────────────────────────────────────────
function fmt(v) { return (v == null) ? '—' : v; }
function fmtN(v, d=3) { return (v == null) ? '—' : Number(v).toFixed(d); }

function badge(val) {
  if (val == null) return '<span class="badge b-gray">—</span>';
  const s = String(val).toLowerCase();
  if (['violence_suspected','violence'].includes(s))    return `<span class="badge b-red">${val}</span>`;
  if (['normal','no_violence'].includes(s))             return `<span class="badge b-green">${val}</span>`;
  if (['true','running'].includes(s))                   return `<span class="badge b-blue">${val}</span>`;
  if (['false','stopped'].includes(s))                  return `<span class="badge b-gray">${val}</span>`;
  if (['crowd_gathering','person_fallen'].includes(s))  return `<span class="badge b-orange">${val}</span>`;
  return `<span class="badge b-gray">${val}</span>`;
}

function row(key, val, asBadge=false) {
  const v = asBadge ? badge(val) : `<span class="field-val">${fmt(val)}</span>`;
  return `<div class="field"><span class="field-key">${key}</span>${v}</div>`;
}

function ticker(msg) {
  document.getElementById('ticker').textContent =
    msg ? `${new Date().toLocaleTimeString()} — ${msg}` : '';
}

function uploadMsg(msg, color='#64748b') {
  const el = document.getElementById('upload-msg');
  el.style.color = color;
  el.textContent = msg;
}

// ── drag-and-drop ─────────────────────────────────────────────────────────────
const dz = document.getElementById('drop-zone');
const fi = document.getElementById('file-input');

dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('hover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('hover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('hover');
  const f = e.dataTransfer.files[0];
  if (f) selectFile(f);
});
fi.addEventListener('change', () => { if (fi.files[0]) selectFile(fi.files[0]); });

function selectFile(f) {
  const ext = f.name.split('.').pop().toLowerCase();
  if (!['mp4','avi','mov','mkv'].includes(ext)) {
    uploadMsg('Unsupported format. Use .mp4, .avi, .mov or .mkv', '#f87171');
    return;
  }
  _selectedFile = f;
  _uploadedSource = null;
  document.getElementById('drop-label').textContent = f.name;
  document.getElementById('btn-upload').disabled = false;
  document.getElementById('btn-run').disabled = true;
  document.getElementById('progress-bar').style.width = '0';
  uploadMsg('File selected — click "Upload Video" to upload');
}

// ── upload ────────────────────────────────────────────────────────────────────
async function uploadFile() {
  if (!_selectedFile) return;
  const btn = document.getElementById('btn-upload');
  btn.disabled = true;
  uploadMsg('Uploading…', '#67e8f9');
  document.getElementById('progress-bar').style.width = '60%';

  const form = new FormData();
  form.append('file', _selectedFile);

  try {
    const r = await fetch('/api/video/upload', { method: 'POST', body: form });
    const d = await r.json();
    document.getElementById('progress-bar').style.width = '100%';
    if (!r.ok) {
      uploadMsg('Upload failed: ' + (d.detail || r.status), '#f87171');
      btn.disabled = false;
      return;
    }
    _uploadedSource = d.source;
    uploadMsg(`Uploaded: ${d.filename}`, '#6ee7b7');
    document.getElementById('btn-run').disabled = false;
    ticker('Upload complete — click "Run Uploaded Video Test" to start');
  } catch(e) {
    uploadMsg('Upload error: ' + e, '#f87171');
    btn.disabled = false;
  }
}

// ── poll ──────────────────────────────────────────────────────────────────────
function startPoll() {
  if (_pollTimer) return;
  _pollTimer = setInterval(async () => {
    const s = await _fetchStatus(true);
    if (s && !s.is_running) stopPoll();
  }, 2000);
}

function stopPoll() {
  clearInterval(_pollTimer);
  _pollTimer = null;
  ticker('Video stopped — polling ended');
}

// ── API ───────────────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const d = await fetch('/health').then(r => r.json());
    setCard('card-health', row('status', d.status, true) + row('service', d.service));
    ticker('Health OK');
  } catch(e) { ticker('Health check failed'); }
}

async function startPreset(mode) {
  const source = mode === 'violence' ? 'data/videos/test.mp4' : 'data/videos/no_violence.mp4';
  await _startVideo(source);
}

async function runUploaded() {
  if (!_uploadedSource) return;
  await _startVideo(_uploadedSource);
}

async function _startVideo(source) {
  try {
    const r = await fetch('/api/video/start', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ source }),
    });
    const d = await r.json();
    ticker(d.message);
    renderStatus(d.status);
    startPoll();
  } catch(e) { ticker('Start failed: ' + e); }
}

async function stopVideo() {
  try {
    const d = await fetch('/api/video/stop', { method:'POST' }).then(r => r.json());
    ticker(d.message);
    stopPoll();
    await _fetchStatus(true);
  } catch(e) { ticker('Stop failed'); }
}

async function fetchStatus() { await _fetchStatus(false); }

async function _fetchStatus(silent=false) {
  try {
    const d = await fetch('/api/video/status').then(r => r.json());
    renderStatus(d);
    if (!silent) ticker('Status refreshed');
    return d;
  } catch(e) { if (!silent) ticker('Status failed'); return null; }
}

async function fetchAlerts() {
  try {
    const d = await fetch('/api/alerts').then(r => r.json());
    renderAlerts(d);
    ticker(`${d.length} alert(s) loaded`);
  } catch(e) { ticker('Alerts failed'); }
}

// ── renderers ─────────────────────────────────────────────────────────────────
function renderStatus(s) {
  setCard('card-status',
    row('is_running',          s.is_running, true) +
    row('source',              s.source) +
    row('frames_read',         s.frames_read) +
    row('frames_processed',    s.frames_processed) +
    row('ai_windows_analyzed', s.ai_windows_analyzed)
  );
  setCard('card-ai',
    row('video_label',      s.latest_video_label,      true) +
    row('video_confidence', fmtN(s.latest_video_confidence)) +
    row('final_label',      s.latest_final_label,      true) +
    row('final_confidence', fmtN(s.latest_final_confidence))
  );
  setCard('card-event',
    row('event_type',            s.latest_event_type,       true) +
    row('confidence',            fmtN(s.latest_event_confidence)) +
    row('severity',              s.latest_event_severity,   true) +
    row('reason',                s.latest_event_reason) +
    row('requires_human_review', s.requires_human_review,   true)
  );
}

function renderAlerts(alerts) {
  if (!alerts.length) {
    setCard('card-alerts', '<div class="empty">No alerts yet</div>');
    return;
  }
  const rows = alerts.map(a => `
    <div class="alert-row">
      <div class="ah">${badge(a.event_type)}<span style="color:#475569;font-size:.72rem">#${a.id}</span></div>
      <div class="am">
        confidence: ${fmt(a.confidence)} &nbsp;|&nbsp;
        status: ${fmt(a.status)} &nbsp;|&nbsp;
        ${new Date(a.created_at).toLocaleString()}
      </div>
    </div>`).join('');
  setCard('card-alerts', `<div class="alerts-list">${rows}</div>`);
}
</script>
</body>
</html>"""


@router.get("/test-ui", include_in_schema=False)
async def test_ui() -> HTMLResponse:
    return HTMLResponse(content=_HTML)
