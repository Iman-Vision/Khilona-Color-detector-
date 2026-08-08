const API_BASE = '';

const beltCounts = { A: 0, B: 0, C: 0 };
let cameraStream = null;
let cameraInterval = null;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// --- INIT ---
document.addEventListener('DOMContentLoaded', () => {
    checkModel();
    setupTabs();
    setupUpload();
    setupCamera();
    setupBatch();
    setupReset();
});

// --- MODEL CHECK ---
async function checkModel() {
    const statusEl = $('#modelStatus');
    try {
        const res = await fetch(`${API_BASE}/api/model_info`);
        const data = await res.json();
        statusEl.classList.add('loaded');
        const accText = data.test_accuracy != null
            ? ` | Acc: ${(data.test_accuracy * 100).toFixed(1)}%`
            : '';
        statusEl.querySelector('span:last-child').textContent = `Model loaded${accText}`;
    } catch {
        statusEl.classList.add('error');
        statusEl.querySelector('span:last-child').textContent = 'Model not loaded';
    }
}

// --- TABS ---
function setupTabs() {
    $$('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const target = tab.dataset.tab;
            $$('.tab-content').forEach(c => c.classList.remove('active'));
            $(`#tab-${target}`).classList.add('active');
        });
    });
}

// --- FILE UPLOAD ---
function setupUpload() {
    const zone = $('#uploadZone');
    const input = $('#fileInput');
    const previewContainer = $('#previewContainer');
    const previewImage = $('#previewImage');
    const clearBtn = $('#clearPreview');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    input.addEventListener('change', () => {
        if (input.files.length) handleFile(input.files[0]);
    });
    clearBtn.addEventListener('click', () => {
        previewContainer.style.display = 'none';
        zone.style.display = '';
        $('#resultDisplay').style.display = 'none';
        $('.result-placeholder').style.display = '';
        input.value = '';
    });
}

async function handleFile(file) {
    $('#previewContainer').style.display = '';
    $('#uploadZone').style.display = 'none';
    const reader = new FileReader();
    reader.onload = (e) => {
        $('#previewImage').src = e.target.result;
    };
    reader.readAsDataURL(file);

    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('image', file);
        const res = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        displayResult(data);
    } catch (err) {
        alert('Prediction failed: ' + err.message);
    } finally {
        showLoading(false);
    }
}

// --- CAMERA ---
function setupCamera() {
    $('#startCamera').addEventListener('click', startCamera);
    $('#stopCamera').addEventListener('click', stopCamera);
}

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
        });
        const video = $('#cameraVideo');
        video.srcObject = cameraStream;
        $('#cameraOverlay').classList.add('hidden');
        $('#startCamera').style.display = 'none';
        $('#stopCamera').style.display = '';

        cameraInterval = setInterval(captureFrame, 1000);
    } catch (err) {
        alert('Camera access denied: ' + err.message);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    clearInterval(cameraInterval);
    cameraInterval = null;
    $('#cameraVideo').srcObject = null;
    $('#cameraOverlay').classList.remove('hidden');
    $('#startCamera').style.display = '';
    $('#stopCamera').style.display = 'none';
}

async function captureFrame() {
    const video = $('#cameraVideo');
    const canvas = $('#cameraCanvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const frame = canvas.toDataURL('image/jpeg', 0.8);

    try {
        const res = await fetch(`${API_BASE}/api/predict_frame`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame })
        });
        const data = await res.json();
        if (data.color && data.color !== 'unknown') {
            displayResult(data, null);
        }
    } catch { /* ignore frame errors */ }
}

// --- BATCH ---
function setupBatch() {
    const zone = $('#batchUploadZone');
    const input = $('#batchFileInput');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleBatch(e.dataTransfer.files);
    });
    input.addEventListener('change', () => {
        if (input.files.length) handleBatch(input.files);
    });
}

async function handleBatch(files) {
    const list = $('#batchList');
    list.innerHTML = '';

    for (const file of files) {
        const item = document.createElement('div');
        item.className = 'batch-item';

        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);

        const info = document.createElement('div');
        info.className = 'batch-item-info';
        info.innerHTML = `<div class="batch-item-name">${file.name}</div><div class="batch-item-result">Processing...</div>`;

        item.appendChild(img);
        item.appendChild(info);
        list.appendChild(item);

        try {
            const formData = new FormData();
            formData.append('image', file);
            const res = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body: formData });
            const data = await res.json();
            if (data.color === 'unknown') {
                info.querySelector('.batch-item-result').textContent = `Unknown (${(data.confidence * 100).toFixed(1)}%)`;
            } else {
                info.querySelector('.batch-item-result').textContent =
                    `${data.color.toUpperCase()} -> Belt ${data.belt} (${(data.confidence * 100).toFixed(1)}%)`;
                incrementBelt(data.belt);
            }
        } catch {
            info.querySelector('.batch-item-result').textContent = 'Error';
        }
    }
}

// --- DISPLAY RESULT ---
function displayResult(data) {
    $('.result-placeholder').style.display = 'none';
    const display = $('#resultDisplay');
    display.style.display = '';

    if (data.preview) {
        $('#resultImage').src = data.preview;
    }

    const color = data.color;
    $('#resultColor').textContent = color.toUpperCase();
    $('#resultColor').className = `color-${color}`;
    $('#colorDot').className = `color-dot dot-${color}`;
    $('#resultBelt').textContent = data.belt;

    const conf = data.confidence;
    $('#confBar').style.width = `${conf * 100}%`;
    $('#confBar').style.background = conf > 0.8 ? 'var(--success)' : conf > 0.6 ? 'var(--yellow)' : 'var(--danger)';
    $('#confValue').textContent = `${(conf * 100).toFixed(1)}%`;

    const allConfEl = $('#allConfidences');
    allConfEl.innerHTML = '';
    if (data.all_confidences) {
        for (const [cls, val] of Object.entries(data.all_confidences)) {
            const chip = document.createElement('div');
            chip.className = 'conf-chip';
            chip.innerHTML = `<span class="dot dot-${cls}"></span>${cls}: ${(val * 100).toFixed(1)}%`;
            allConfEl.appendChild(chip);
        }
    }

    if (data.belt && data.belt !== 'None') {
        incrementBelt(data.belt);
        const beltEl = $(`#belt-${data.belt}`);
        if (beltEl) {
            beltEl.classList.add('flash');
            setTimeout(() => beltEl.classList.remove('flash'), 500);
        }
    }
}

function incrementBelt(belt) {
    if (beltCounts[belt] !== undefined) {
        beltCounts[belt]++;
        $(`#belt${belt}Count`).textContent = beltCounts[belt];
    }
}

function setupReset() {
    $('#resetCounters').addEventListener('click', () => {
        beltCounts.A = 0;
        beltCounts.B = 0;
        beltCounts.C = 0;
        $('#beltACount').textContent = '0';
        $('#beltBCount').textContent = '0';
        $('#beltCCount').textContent = '0';
    });
}

function showLoading(show) {
    $('#loadingOverlay').style.display = show ? '' : 'none';
}
