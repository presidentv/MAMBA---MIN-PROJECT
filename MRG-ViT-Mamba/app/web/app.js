/* Front end for the disengagement detector.
 *
 * The page deliberately leads with the score and the evidence rather than a
 * verdict badge. At the operating point this checkpoint was evaluated on,
 * a "not engaged" call is right about one time in six, so a confident-looking
 * label would misrepresent the model to anyone reading the screen. What the
 * model is actually good at -- ranking clips by how likely they are to show
 * disengagement -- is what the gauge shows.
 */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const api = async (url, opts) => {
  const res = await fetch(url, opts);
  const body = await res.json().catch(() => ({ error: res.statusText }));
  if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
};

const pct = (x) => `${(x * 100).toFixed(1)}%`;

let MODEL = null;

/* ------------------------------------------------------------------ boot */
async function boot() {
  try {
    MODEL = await api('/api/model');
    renderModelBar(MODEL);
  } catch (err) {
    $('#hint').textContent = `could not reach the model: ${err.message}`;
    return;
  }

  let clips = [];
  try {
    clips = (await api('/api/clips')).clips;
  } catch { /* the page still works for uploads */ }

  const host = $('#cards');
  if (!clips.length) {
    host.innerHTML =
      '<p class="placeholder">No demo clips found. Drop video files into ' +
      '<code>app/demo/</code> and reload, or upload one above.</p>';
    $('#run-all').disabled = true;
  }
  clips.forEach((c) => host.appendChild(makeCard(c)));

  $('#run-all').addEventListener('click', runAll);
  $('#file').addEventListener('change', onUpload);
}

/* timm names the backbone "vit_base_patch16_224.augreg2_in21k_ft_in1k";
   the paper calls it ViT-B/16, and so should the page. */
function prettyBackbone(name) {
  const m = /^vit_(\w+?)_patch(\d+)/.exec(name || '');
  if (m) return `ViT-${m[1][0].toUpperCase()}/${m[2]}`;
  return (name || '').split('.')[0] || '—';
}

function renderModelBar(m) {
  $('#m-backbone').textContent = prettyBackbone(m.backbone);
  $('#m-classes').textContent = (m.classes || []).join(' / ');
  $('#m-frames').textContent = m.num_frames;
  $('#m-device').textContent = m.device;
  const t = m.test_metrics || {};
  $('#m-auc').textContent = t.roc_auc != null ? t.roc_auc.toFixed(4) : '—';

  $('#caveat-text').textContent =
    'The gauge is the model’s probability that a clip shows disengagement. ' +
    'It is a ranking, and it is reliable in aggregate rather than clip by clip: ' +
    'on held-out data it separates disengaged from engaged clips at ' +
    `${(t.roc_auc ?? 0).toFixed(3)} ROC-AUC, where 0.5 would be no signal at all.`;
  $('#c-found').textContent = t.found;
  $('#c-total').textContent = t.total_not_engaged;
  $('#c-fa').textContent = t.false_alarms;
  $('#c-const').textContent = pct(t.constant_engaged_accuracy ?? 0);
}

/* ------------------------------------------------------------------ card */
function makeCard(clip) {
  const node = $('#card-tpl').content.firstElementChild.cloneNode(true);
  $('.clip-name', node).textContent = clip.name;
  $('.player', node).src = `/api/clip/${clip.id}`;
  const btn = $('.analyse', node);
  btn.addEventListener('click', () => analyse(node, btn, () =>
    api('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clip: clip.id, explain: true }),
    })));
  node.dataset.clipId = clip.id;
  return node;
}

async function analyse(card, btn, call) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Analysing…';
  try {
    render(card, await call());
  } catch (err) {
    $('.result', card).classList.add('empty');
    $('.placeholder', card).textContent = `failed: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = label;
  }
}

async function runAll() {
  const btn = $('#run-all');
  btn.disabled = true;
  const original = btn.textContent;
  const cards = [...document.querySelectorAll('.card')];
  for (const [i, card] of cards.entries()) {
    btn.textContent = `Analysing ${i + 1} of ${cards.length}…`;
    // Sequential on purpose: one model on one 6 GB card. Firing these in
    // parallel would queue on the server's lock anyway and risk VRAM.
    await analyse(card, $('.analyse', card), () =>
      api('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clip: card.dataset.clipId, explain: true }),
      }));
  }
  btn.textContent = original;
  btn.disabled = false;
}

/* ---------------------------------------------------------------- upload */
async function onUpload(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const card = makeCard({ id: '', name: file.name });
  $('.player', card).src = URL.createObjectURL(file);
  const btn = $('.analyse', card);
  const call = () => api('/api/upload', {
    method: 'POST',
    headers: { 'X-Filename': file.name, 'Content-Type': 'application/octet-stream' },
    body: file,
  });
  btn.replaceWith(btn.cloneNode(true));          // drop the demo-clip handler
  const fresh = $('.analyse', card);
  fresh.addEventListener('click', () => analyse(card, fresh, call));
  $('#cards').prepend(card);
  ev.target.value = '';
  analyse(card, fresh, call);
}

/* ---------------------------------------------------------------- render */
function render(card, r) {
  const result = $('.result', card);
  result.classList.remove('empty');

  const classes = MODEL.classes || [];
  // detection_score is P(not engaged) -- the quantity the paper's ROC-AUC
  // ranks on. Fall back to the predicted class for a non-binary checkpoint.
  const score = r.detection_score != null
    ? r.detection_score
    : r.probabilities[r.predicted_class];
  const flagged = r.predicted_class.toLowerCase().includes('not');

  // The plain-language verdict, stated once and unambiguously, with the
  // number it came from kept beside it rather than hidden.
  const banner = $('.banner', card);
  banner.className = `banner ${flagged ? 'flag' : 'ok'}`;
  $('.banner-label', banner).textContent =
    flagged ? 'Disengaged' : 'Engaged';
  $('.banner-sub', banner).textContent =
    `the model scores this clip ${score.toFixed(3)} on P(disengaged), ` +
    `against a 0.500 decision threshold`;

  $('.gauge-fill', card).style.width = `${Math.max(2, score * 100)}%`;
  const lab = $('.label', card);
  lab.textContent = r.predicted_class;
  lab.className = `label ${flagged ? 'flag' : 'ok'}`;
  $('.score', card).textContent =
    `P(disengaged) ${score.toFixed(3)} · ${r.elapsed_s}s`;

  renderFilmstrip(card, r);
  renderStrip(card, r);
  renderGroups(card, r);
  renderQuality(card, r, classes);
}

/* ------------------------------------------------------------- filmstrip */
function renderFilmstrip(card, r) {
  const strip = $('.filmstrip', card);
  const detail = $('.framedetail', card);
  strip.innerHTML = '';
  detail.innerHTML = '';
  detail.classList.add('empty');

  const frames = r.frames || [];
  if (!frames.length) {
    $('.filmnote', card).textContent = 'No per-frame detail was returned.';
    return;
  }
  // Pooling weights sum to 1 across the clip, so the bars are scaled against
  // the largest rather than against 1, or every bar would be a sliver.
  const maxW = Math.max(...frames.map((f) => f.pooling_weight ?? f.reliability));

  frames.forEach((f) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'fr';
    if (!f.landmarks_found || !f.face_found) btn.classList.add('weak');
    const w = f.pooling_weight ?? f.reliability;
    btn.innerHTML =
      (f.image ? `<img src="${f.image}" alt="frame ${f.index + 1}">`
               : '<img alt="no image">') +
      `<span class="wbar"><span style="width:${(w / maxW) * 100}%"></span></span>` +
      `<span class="wnum">${(w * 100).toFixed(1)}%</span>`;
    btn.title = `frame ${f.index + 1}: reliability ${f.reliability.toFixed(3)}`;
    btn.addEventListener('click', () => {
      [...strip.children].forEach((c) => c.classList.remove('sel'));
      btn.classList.add('sel');
      showFrame(detail, f, r);
    });
    strip.appendChild(btn);
  });

  // Open on the frame the model leaned on most, so the panel sees something
  // meaningful without having to click.
  const lead = frames.reduce((a, b) =>
    (b.pooling_weight ?? b.reliability) > (a.pooling_weight ?? a.reliability) ? b : a);
  strip.children[lead.index].classList.add('sel');
  showFrame(detail, lead, r);
}

function showFrame(host, f, r) {
  host.classList.remove('empty');
  const w = f.pooling_weight ?? f.reliability;
  const comps = Object.entries(f.components)
    .map(([k, v]) => `<dt>${k.replace(/_/g, ' ')}</dt><dd>${v.toFixed(3)}</dd>`)
    .join('');
  const derived = f.derived.map((d) =>
    `<dt title="${d.note}">${d.label}</dt>` +
    (d.value === null
      ? '<dd class="na">no mesh</dd>'
      : `<dd>${d.value.toFixed(3)}</dd>`)).join('');

  const src = f.source_frame === null || f.source_frame === undefined
    ? '' : ` · source frame ${f.source_frame}`;
  const pose = f.head_pose_deg
    ? ` · pose ${f.head_pose_deg.join('° / ')}°` : '';
  const fallback = f.face_found ? ''
    : ` · no face detected, box from ${f.bbox_source}`;

  host.innerHTML =
    '<div class="fd">' +
    (f.image ? `<img src="${f.image}" alt="frame ${f.index + 1}">` : '<div></div>') +
    '<div>' +
      `<h5>Frame ${f.index + 1} of ${(r.frames || []).length}</h5>` +
      `<p class="meta">reliability ${f.reliability.toFixed(3)} · ` +
      `carries ${(w * 100).toFixed(1)}% of the clip${src}${pose}${fallback}</p>` +
      '<p class="sect">Reliability components</p>' +
      `<dl class="kv">${comps}</dl>` +
      '<p class="sect">Measured from this frame</p>' +
      `<dl class="kv">${derived}</dl>` +
    '</div></div>';
}

function renderStrip(card, r) {
  const strip = $('.strip', card);
  strip.innerHTML = '';
  const rel = r.per_frame_reliability || [];
  const mean = rel.reduce((a, b) => a + b, 0) / (rel.length || 1);
  rel.forEach((v, i) => {
    const bar = document.createElement('i');
    bar.style.height = `${Math.max(4, v * 100)}%`;
    if (v < mean * 0.85) bar.classList.add('low');
    bar.title = `frame ${i + 1}: reliability ${v.toFixed(3)}`;
    strip.appendChild(bar);
  });
  const lo = Math.min(...rel), hi = Math.max(...rel);
  $('.stripnote', card).textContent =
    `${rel.length} sampled frames · mean ${mean.toFixed(3)}, ` +
    `range ${lo.toFixed(3)}–${hi.toFixed(3)}. ` +
    'Every frame stays in the sequence; the weaker ones simply count for less.';
}

function renderGroups(card, r) {
  const host = $('.groups', card);
  host.innerHTML = '';
  const groups = r.explanation && r.explanation.landmark_groups;
  if (!groups) {
    $('.attr-note', card).textContent = 'No attribution was computed for this clip.';
    return;
  }
  const rows = Object.entries(groups).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const max = Math.max(...rows.map(([, v]) => Math.abs(v))) || 1;
  for (const [name, value] of rows) {
    const row = document.createElement('div');
    row.className = 'grow';
    const width = (Math.abs(value) / max) * 100;
    row.innerHTML =
      `<span class="gname">${name.replace(/_/g, ' ')}</span>` +
      `<span class="gbar"><span class="${value < 0 ? 'neg' : ''}" ` +
      `style="width:${width}%"></span></span>` +
      `<span class="gval">${value >= 0 ? '+' : ''}${value.toFixed(2)}</span>`;
    host.appendChild(row);
  }
  $('.attr-note', card).textContent =
    'Integrated gradients over the named landmark features, towards the ' +
    'predicted class. Blue pushes towards it, grey away from it. Across 512 ' +
    'test clips, eye openness and gaze geometry carry roughly twice the ' +
    'attribution per dimension of the deep ViT–Mamba block.';
}

function renderQuality(card, r, classes) {
  const q = r.quality || {};
  const list = $('.qlist', card);
  list.innerHTML = '';
  const add = (text, warn) => {
    const li = document.createElement('li');
    li.textContent = text;
    if (warn) li.className = 'warn';
    list.appendChild(li);
  };
  add(`face found in ${q.frames_with_face}/${q.num_frames} sampled frames, ` +
      `landmarks in ${q.frames_with_landmarks}`,
      q.frames_with_face < q.num_frames);
  if (q.resolution) {
    add(`source ${q.resolution[0]}×${q.resolution[1]} at ` +
        `${(q.fps || 0).toFixed(0)} fps` +
        (q.duration_s ? `, ${q.duration_s}s long` : '') +
        `, ${q.source_frame_count} frames`);
  }
  const probs = Object.entries(r.probabilities)
    .map(([k, v]) => `${k} ${v.toFixed(3)}`).join(' · ');
  add(probs);
  if (r.tta_views) {
    add(`averaged over ${r.tta_views.length} centre-crop views, as evaluated`);
  }
  (q.warnings || []).forEach((w) => add(w, true));
  if (r.explanation && r.explanation.disagrees_with_tta) {
    add(r.explanation.disagrees_with_tta, true);
  }
}

boot();
