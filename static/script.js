'use strict';

// ── Tab / segmented control ──────────────────────────────────────
const segBtns = document.querySelectorAll('.seg-btn');
const panels  = document.querySelectorAll('.panel');

segBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    segBtns.forEach(b => b.classList.remove('active'));
    panels.forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`${btn.dataset.tab}-panel`).classList.add('active');
    stopLive();
  });
});

// ── Webcam ───────────────────────────────────────────────────────
const video      = document.getElementById('video');
const overlay    = document.getElementById('overlay');
const startBtn   = document.getElementById('startBtn');
const captureBtn = document.getElementById('captureBtn');
const liveBtn    = document.getElementById('liveBtn');
const liveBtnText = document.getElementById('liveBtnText');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const webcamResults    = document.getElementById('webcamResults');

let stream     = null;
let liveTimer  = null;
let liveActive = false;

startBtn.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    videoPlaceholder.classList.add('hidden');
    startBtn.disabled   = true;
    captureBtn.disabled = false;
    liveBtn.disabled    = false;
  } catch (err) {
    showError('Could not access camera: ' + err.message);
  }
});

captureBtn.addEventListener('click', () => {
  const dataUrl = captureFrame();
  if (dataUrl) sendForAnalysis(dataUrl, webcamResults);
});

liveBtn.addEventListener('click', () => {
  liveActive = !liveActive;
  liveBtnText.textContent = `Live: ${liveActive ? 'ON' : 'OFF'}`;
  liveBtn.classList.toggle('live-on', liveActive);
  if (liveActive) scheduleLive();
  else stopLive();
});

function captureFrame() {
  if (!stream) return null;
  const canvas  = document.createElement('canvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.85);
}

function scheduleLive() {
  liveTimer = setTimeout(async () => {
    if (!liveActive) return;
    const dataUrl = captureFrame();
    if (dataUrl) await sendForAnalysis(dataUrl, webcamResults, true);
    if (liveActive) scheduleLive();
  }, 800);
}

function stopLive() {
  liveActive = false;
  clearTimeout(liveTimer);
  liveBtnText.textContent = 'Live: OFF';
  liveBtn.classList.remove('live-on');
}

// ── Upload ───────────────────────────────────────────────────────
const dropZone         = document.getElementById('dropZone');
const fileInput        = document.getElementById('fileInput');
const dropContent      = document.getElementById('dropContent');
const analyseUploadBtn = document.getElementById('analyseUploadBtn');
const uploadResults    = document.getElementById('uploadResults');

let uploadDataUrl = null;

dropZone.addEventListener('click', e => { if (e.target !== fileInput) fileInput.click(); });

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function loadFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    uploadDataUrl = e.target.result;
    dropContent.innerHTML = '';
    const img = document.createElement('img');
    img.src = uploadDataUrl;
    dropContent.appendChild(img);
    analyseUploadBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

analyseUploadBtn.addEventListener('click', () => {
  if (uploadDataUrl) sendForAnalysis(uploadDataUrl, uploadResults);
});

// ── API call ─────────────────────────────────────────────────────
const spinner  = document.getElementById('spinner');
const errorMsg = document.getElementById('error-msg');

async function sendForAnalysis(dataUrl, resultsPanel, silent = false) {
  if (!silent) showSpinner(true);
  hideError();

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();

    if (!res.ok) { showError(data.error || 'Server error'); return; }

    renderResults(data, resultsPanel, silent);
  } catch (err) {
    showError('Network error: ' + err.message);
  } finally {
    if (!silent) showSpinner(false);
  }
}

function renderResults(data, resultsPanel, silent) {
  // Always update the annotated image
  let imgWrapper = resultsPanel.querySelector('.annotated-result');
  if (!imgWrapper) {
    imgWrapper = document.createElement('div');
    imgWrapper.className = 'annotated-result';
    const img = document.createElement('img');
    img.alt = 'Annotated result';
    imgWrapper.appendChild(img);
    resultsPanel.innerHTML = '';
    resultsPanel.appendChild(imgWrapper);
  }
  imgWrapper.querySelector('img').src = data.annotated;

  if (!silent) {
    // Remove old face cards
    resultsPanel.querySelectorAll('.face-card, .no-face').forEach(el => el.remove());

    if (data.faces.length === 0) {
      const p = document.createElement('p');
      p.className = 'no-face';
      p.textContent = 'No faces detected.';
      resultsPanel.appendChild(p);
    } else {
      data.faces.forEach((face, idx) => resultsPanel.appendChild(buildCard(face, idx)));
    }
  }
}

function buildCard(face, idx) {
  const card = document.createElement('div');
  card.className = 'face-card';

  const header = document.createElement('div');
  header.className = 'face-header';

  const pill = document.createElement('span');
  pill.className = 'emotion-pill';
  pill.style.background = face.color;
  pill.textContent = face.emotion;

  const conf = document.createElement('span');
  conf.className = 'face-conf';
  conf.textContent = `Face ${idx + 1} · ${face.confidence}%`;

  header.appendChild(pill);
  header.appendChild(conf);
  card.appendChild(header);

  const sorted = Object.entries(face.scores).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([emotion, pct]) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <span class="bar-label">${emotion}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${face.color}"></div></div>
      <span class="bar-pct">${pct}%</span>`;
    card.appendChild(row);
  });

  return card;
}

// ── Helpers ──────────────────────────────────────────────────────
function showSpinner(show) { spinner.hidden = !show; }
function showError(msg)    { errorMsg.textContent = msg; errorMsg.hidden = false; }
function hideError()       { errorMsg.hidden = true; }
