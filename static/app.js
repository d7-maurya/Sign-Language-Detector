const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const captureBtn = document.getElementById('captureBtn');
const cameraStatus = document.getElementById('cameraStatus');
const videoOverlay = document.getElementById('videoOverlay');

// Stats Elements
const predictionEl = document.getElementById('prediction');
const confPercent = document.getElementById('confPercent');
const confidenceFill = document.getElementById('confidenceFill');
const historyList = document.getElementById('historyList');
const statFrames = document.getElementById('statFrames');
const statHands = document.getElementById('statHands');
const statAvg = document.getElementById('statAvg');

let stream = null;
let autoInterval = null;
let stats = { frames: 0, hands: 0, totalConf: 0, countConf: 0 };

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        captureBtn.disabled = false;
        cameraStatus.textContent = "CAMERA ACTIVE";
        cameraStatus.classList.add('active');
        videoOverlay.classList.add('hidden');
        
        // Auto-predict loop
        autoInterval = setInterval(captureAndPredict, 1500);
        
    } catch (err) {
        console.error(err);
        alert("Camera access denied");
    }
}

function stopCamera() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); }
    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    captureBtn.disabled = true;
    cameraStatus.textContent = "CAMERA INACTIVE";
    cameraStatus.classList.remove('active');
    videoOverlay.classList.remove('hidden');
    clearInterval(autoInterval);
}

async function captureAndPredict() {
    if (!stream) return;
    
    // Stats: Frame count
    stats.frames++;
    statFrames.textContent = stats.frames;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const data = await res.json();

        if (data.success && data.prediction !== "No Hand") {
            // Stats: Hand Count & Avg
            stats.hands++;
            stats.totalConf += data.confidence;
            stats.countConf++;
            
            updateUI(data.prediction, data.confidence);
            updateStatsUI();
        } else {
             // Reset UI if no hand
             confidenceFill.style.width = '0%';
             confPercent.textContent = '0%';
        }
    } catch (err) { console.error(err); }
}

function updateUI(pred, conf) {
    predictionEl.textContent = pred;
    const rounded = Math.round(conf);
    confidenceFill.style.width = `${rounded}%`;
    confPercent.textContent = `${rounded}%`;
    
    // Add to history if different from last
    const firstChild = historyList.firstChild;
    if (!firstChild || firstChild.textContent !== pred) {
        const pill = document.createElement('div');
        pill.className = 'history-pill';
        pill.textContent = pred;
        historyList.prepend(pill);
        if (historyList.children.length > 10) historyList.lastChild.remove();
    }
}

function updateStatsUI() {
    statFrames.textContent = stats.frames;
    statHands.textContent = stats.hands;
    if (stats.countConf > 0) {
        statAvg.textContent = Math.round(stats.totalConf / stats.countConf) + "%";
    }
}

document.getElementById('clearHistory').addEventListener('click', () => {
    historyList.innerHTML = '';
    stats = { frames: 0, hands: 0, totalConf: 0, countConf: 0 };
    updateStatsUI();
});

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);