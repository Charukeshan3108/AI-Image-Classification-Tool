/**
 * Virtual Vision AI - Zero-Shot & Few-Shot CLIP Dashboard Engine
 * Handles frontend state, interactive tag chips, inference requests, sample presets,
 * connection diagnostics, and session history.
 */

const MODEL_ID = 'openai/clip-vit-base-patch32';
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_EXAMPLES = 8;
const HISTORY_KEY = 'clip-classification-history';
const API_URL_KEY = 'clip_api_url';

// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');
const imagePreview = document.getElementById('image-preview');
const previewImage = document.getElementById('preview-image');
const resultState = document.getElementById('result-state');
const runNote = document.getElementById('run-note');
const exampleFileInput = document.getElementById('example-file-input');
const candidateInput = document.getElementById('candidate-input');
const candidateTagInput = document.getElementById('candidate-tag-input');
const tagsContainer = document.getElementById('tags-container');
const tagCounter = document.getElementById('tag-counter');
const backendStatus = document.getElementById('backend-status');
const backendAlertBanner = document.getElementById('backend-alert-banner');
const topMatchHero = document.getElementById('top-match-hero');
const topMatchLabel = document.getElementById('top-match-label');
const topMatchScore = document.getElementById('top-match-score');

// State Variables
const examples = [];
let candidateTags = ['cat', 'dog', 'sports car', 'coffee cup', 'mountain landscape'];
let selectedFile = null;
let selectedPreviewUrl = null;
let currentMode = 'zero-shot';
let backendAvailable = false;
let isLoading = false;
let demoModeActive = false;
let backendPingMs = null;
let pollIntervalTimer = null;
let history = loadHistory();

// Built-in Category Presets
const PRESET_CATEGORIES = {
    animals: ['golden retriever', 'siamese cat', 'bald eagle', 'bengal tiger', 'giant panda', 'wild dolphin'],
    vehicles: ['electric sedan', 'sports car', 'mountain bicycle', 'commercial airliner', 'bullet train', 'cargo ship'],
    scenes: ['alpine mountain peak', 'tropical beach', 'dense pine forest', 'desert sand dunes', 'modern city skyline'],
    food: ['espresso coffee', 'wood-fired pizza', 'fresh sushi rolls', 'chocolate cake', 'green avocado toast'],
    objects: ['laptop computer', 'mechanical keyboard', 'leather backpack', 'sunglasses', 'smartphone']
};

/* ==========================================================================
   Helper & Button Builders (Backward Compatibility)
   ========================================================================== */

function addButton(parent, text, className, clickFunction, title) {
    if (!parent) return null;
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = text;
    button.className = className;
    button.onclick = clickFunction;
    if (title) button.title = title;
    parent.appendChild(button);
    return button;
}

function createButtons() {
    const navigation = document.getElementById('main-navigation');
    const modeButtons = document.getElementById('mode-buttons');

    if (navigation && !navigation.hasChildNodes()) {
        const navItems = [
            ['◈ Identify image', 'identify-panel'],
            ['⌁ Few-shot prototypes', 'few-shot-panel'],
            ['◌ Session history', 'history-panel']
        ];

        navItems.forEach(function (item, index) {
            addButton(navigation, item[0], 'nav-item' + (index === 0 ? ' active' : ''), function (event) {
                document.querySelectorAll('.nav-item').forEach(function (btn) { btn.classList.remove('active'); });
                event.currentTarget.classList.add('active');
                if (item[1] === 'few-shot-panel') setMode('few-shot');
                if (item[1] === 'identify-panel') setMode('zero-shot');
                const target = document.getElementById(item[1]);
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });
    }

    const resetHost = document.getElementById('reset-button-host');
    if (resetHost && !resetHost.hasChildNodes()) {
        addButton(resetHost, '↺ Reset All', 'action-btn', resetDashboard, 'Clear inputs and start fresh');
    }

    if (modeButtons && !modeButtons.hasChildNodes()) {
        addButton(modeButtons, 'Zero-Shot Classification', 'mode-button active', function () { setMode('zero-shot'); });
        addButton(modeButtons, 'Few-Shot Prototypes', 'mode-button', function () { setMode('few-shot'); });
    }

    const runHost = document.getElementById('run-button-host');
    if (runHost && !runHost.hasChildNodes()) {
        const runBtn = addButton(runHost, 'Identify image →', 'solid-button', identifyImage, 'Execute CLIP classification (Ctrl+Enter)');
        runBtn.id = 'run-inference-btn';
    }

    const browseHost = document.getElementById('browse-button-host');
    if (browseHost && !browseHost.hasChildNodes()) {
        addButton(browseHost, 'Browse Image Files', 'browse-button', function (e) {
            e.stopPropagation();
            fileInput.click();
        });
    }

    const addExampleHost = document.getElementById('add-example-host');
    if (addExampleHost && !addExampleHost.hasChildNodes()) {
        addButton(addExampleHost, '+ Add Example Image', 'add-example', function () {
            if (examples.length >= MAX_EXAMPLES) {
                showError('example-error', 'You can add at most ' + MAX_EXAMPLES + ' example images.');
                return;
            }
            const labelInput = document.getElementById('example-label-input');
            if (!labelInput.value.trim()) {
                showError('example-error', 'Enter a class label before choosing an example image.');
                labelInput.focus();
                return;
            }
            exampleFileInput.click();
        });
    }

    const exportJsonHost = document.getElementById('export-json-host');
    if (exportJsonHost && !exportJsonHost.hasChildNodes()) {
        addButton(exportJsonHost, 'JSON', 'export-button', function () { exportHistory('json'); }, 'Download actual session predictions as JSON');
    }

    const exportCsvHost = document.getElementById('export-csv-host');
    if (exportCsvHost && !exportCsvHost.hasChildNodes()) {
        addButton(exportCsvHost, 'CSV', 'export-button', function () { exportHistory('csv'); }, 'Download actual session predictions as CSV');
    }

    updateActionState();
}

/* ==========================================================================
   Mode & View Switching
   ========================================================================== */

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-button').forEach(function (button, index) {
        button.classList.toggle('active', (mode === 'zero-shot' && index === 0) || (mode === 'few-shot' && index === 1));
    });

    const candidateSettings = document.getElementById('candidate-settings');
    if (candidateSettings) candidateSettings.hidden = mode !== 'zero-shot';

    const modeStat = document.getElementById('mode-stat');
    if (modeStat) modeStat.textContent = mode === 'zero-shot' ? 'Zero-shot' : 'Few-shot prototypes';

    if (runNote) {
        runNote.textContent = mode === 'zero-shot' 
            ? 'Compare query image against text candidate labels' 
            : 'Compare query image against example class prototypes';
    }

    updateActionState();
}

function showError(id, message) {
    const element = document.getElementById(id);
    if (!element) return;
    element.textContent = message;
    element.hidden = !message;
}

function clearErrors() {
    ['upload-error', 'request-error', 'example-error'].forEach(function (id) { showError(id, ''); });
}

/* ==========================================================================
   Image Validation & Loading
   ========================================================================== */

function validImage(file) {
    const allowedExtensions = ['jpg', 'jpeg', 'png', 'webp'];
    const extension = file.name.split('.').pop().toLowerCase();
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowedExtensions.includes(extension) || (file.type && !allowedTypes.includes(file.type))) {
        return 'Choose a valid JPG, JPEG, PNG, or WEBP image.';
    }
    if (file.size > MAX_IMAGE_BYTES) return 'Image must be 10 MB or smaller.';
    if (file.size === 0) return 'The selected file is empty.';
    return '';
}

function loadImage(file) {
    if (!file) return;
    if (isLoading) {
        showError('upload-error', 'Please wait for current classification to finish before changing image.');
        return;
    }
    const error = validImage(file);
    if (error) {
        showError('upload-error', error);
        fileInput.value = '';
        selectedFile = null;
        if (selectedPreviewUrl) URL.revokeObjectURL(selectedPreviewUrl);
        selectedPreviewUrl = null;
        previewImage.removeAttribute('src');
        uploadZone.hidden = false;
        imagePreview.style.display = 'none';
        updateActionState();
        return;
    }

    clearErrors();
    if (selectedPreviewUrl) URL.revokeObjectURL(selectedPreviewUrl);
    selectedFile = file;
    selectedPreviewUrl = URL.createObjectURL(file);
    previewImage.src = selectedPreviewUrl;
    uploadZone.hidden = true;
    imagePreview.style.display = 'block';
    resultState.textContent = 'Image ready (' + (file.size / 1024).toFixed(0) + ' KB)';
    updateActionState();
}

function updateActionState() {
    const runButton = document.querySelector('#run-button-host button');
    const canRun = selectedFile && (backendAvailable || demoModeActive) && !isLoading;
    if (runButton) {
        runButton.disabled = !canRun;
        if (!backendAvailable && !demoModeActive) {
            runButton.title = 'Backend offline. Start Flask or enable Demo Mode.';
        } else if (!selectedFile) {
            runButton.title = 'Upload or select a query image first.';
        } else {
            runButton.title = 'Run CLIP Classification (Ctrl+Enter)';
        }
    }
    document.querySelectorAll('.export-button').forEach(function (button) {
        button.disabled = history.length === 0 || isLoading;
    });
}

/* ==========================================================================
   Backend API Connection & Diagnostics
   ========================================================================== */

function getApiBaseUrl() {
    const customUrl = localStorage.getItem(API_URL_KEY);
    if (customUrl) return customUrl.replace(/\/+$/, '');

    // If served through Flask port 5000, use relative path
    if ((window.location.protocol === 'http:' || window.location.protocol === 'https:') &&
        window.location.port === '5000') {
        return '';
    }

    // Default local Flask host
    return 'http://127.0.0.1:5000';
}

async function checkBackend() {
    const statusBtn = document.getElementById('backend-status-btn');
    const statusText = document.getElementById('backend-status');
    const topDot = document.getElementById('top-status-dot');
    const sidebarDot = document.getElementById('sidebar-dot');
    const sidebarStatus = document.getElementById('sidebar-model-status');
    const modelName = document.getElementById('model-name');
    const sidebarModel = document.getElementById('sidebar-model');
    const alertBanner = document.getElementById('backend-alert-banner');
    const footerOrigin = document.getElementById('footer-origin');

    const baseUrl = getApiBaseUrl();
    if (footerOrigin) footerOrigin.textContent = baseUrl || window.location.host;

    const startTime = performance.now();
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4000);

        const response = await fetch(baseUrl + '/api/health', {
            signal: controller.signal,
            headers: { 'Accept': 'application/json' }
        });
        clearTimeout(timeoutId);

        if (!response.ok) throw new Error('HTTP ' + response.status);
        const data = await response.json();
        backendPingMs = Math.round(performance.now() - startTime);

        backendAvailable = data.status === 'ok';
        if (modelName) modelName.textContent = data.model_id;
        if (sidebarModel) sidebarModel.textContent = data.model_id;

        if (statusText) statusText.textContent = `Connected (${backendPingMs}ms)`;
        if (statusBtn) {
            statusBtn.classList.remove('offline', 'loading');
            statusBtn.title = `Backend online at ${baseUrl || 'current origin'} (${backendPingMs}ms ping)`;
        }
        if (topDot) topDot.className = 'status-dot pulsing';
        if (sidebarDot) sidebarDot.className = 'status-dot pulsing';

        if (sidebarStatus) {
            sidebarStatus.textContent = data.model_loaded
                ? `Loaded · ${data.device || 'CPU'}`
                : 'Ready · weights load on first run';
        }

        if (alertBanner) alertBanner.classList.add('hidden');
    } catch (error) {
        backendAvailable = false;
        if (statusText) statusText.textContent = demoModeActive ? 'Demo Mode Active' : 'Backend Offline';
        if (statusBtn) {
            statusBtn.classList.add(demoModeActive ? 'loading' : 'offline');
            statusBtn.title = `Backend unavailable at ${baseUrl}. Click for settings.`;
        }
        if (topDot) topDot.className = 'status-dot ' + (demoModeActive ? 'loading' : 'offline');
        if (sidebarDot) sidebarDot.className = 'status-dot ' + (demoModeActive ? 'loading' : 'offline');

        if (sidebarStatus) {
            sidebarStatus.textContent = demoModeActive 
                ? 'Simulated inference' 
                : 'Run: .\\.venv\\Scripts\\python.exe app.py';
        }

        if (alertBanner && !demoModeActive) {
            alertBanner.classList.remove('hidden');
        }
    }
    updateActionState();
}

/* ==========================================================================
   Candidate Labels & Interactive Tag Chips
   ========================================================================== */

function renderTags() {
    if (!tagsContainer) return;
    // Remove existing chips except the input
    const chips = tagsContainer.querySelectorAll('.tag-chip');
    chips.forEach(chip => chip.remove());

    candidateTags.forEach((tag, index) => {
        const chip = document.createElement('span');
        chip.className = 'tag-chip';
        chip.textContent = tag;

        const removeBtn = document.createElement('span');
        removeBtn.className = 'remove-tag';
        removeBtn.innerHTML = '&times;';
        removeBtn.title = 'Remove label';
        removeBtn.onclick = function (e) {
            e.stopPropagation();
            candidateTags.splice(index, 1);
            syncTagsToTextarea();
            renderTags();
        };

        chip.appendChild(removeBtn);
        tagsContainer.insertBefore(chip, candidateTagInput);
    });

    if (tagCounter) {
        tagCounter.textContent = `${candidateTags.length} / 30 labels`;
        tagCounter.style.color = candidateTags.length > 30 ? 'var(--accent-rose)' : 'var(--text-muted)';
    }
}

function syncTagsToTextarea() {
    if (candidateInput) {
        candidateInput.value = candidateTags.join(', ');
    }
}

function syncTextareaToTags() {
    if (!candidateInput) return;
    const raw = candidateInput.value;
    const parsed = raw.split(/[\n,]+/)
        .map(t => t.trim().replace(/\s+/g, ' '))
        .filter(Boolean);

    // Deduplicate preserving case of first occurrence
    const seen = new Set();
    candidateTags = [];
    parsed.forEach(label => {
        if (!seen.has(label.toLowerCase()) && candidateTags.length < 30) {
            seen.add(label.toLowerCase());
            candidateTags.push(label);
        }
    });
    renderTags();
}

function addCandidateTag(label) {
    if (!label) return;
    const clean = label.trim().replace(/\s+/g, ' ');
    if (!clean) return;
    if (clean.length > 80) {
        showError('request-error', 'Labels must be 80 characters or fewer.');
        return;
    }
    if (candidateTags.some(t => t.toLowerCase() === clean.toLowerCase())) {
        // Flash tag if already exists
        return;
    }
    if (candidateTags.length >= 30) {
        showError('request-error', 'Maximum 30 candidate labels reached.');
        return;
    }
    candidateTags.push(clean);
    syncTagsToTextarea();
    renderTags();
    clearErrors();
}

function parseCandidateLabels() {
    syncTextareaToTags();
    if (candidateTags.length < 1) throw new Error('Enter at least 1 candidate label.');
    if (candidateTags.length > 30) throw new Error('Enter between 1 and 30 candidate labels.');
    return [...candidateTags];
}

/* ==========================================================================
   Sample Image Presets Generator (Instant 1-Click Testing)
   ========================================================================== */

function generateSampleImageBlob(type) {
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');

    if (type === 'dog') {
        // Golden Retriever aesthetic palette
        const grad = ctx.createLinearGradient(0, 0, 400, 300);
        grad.addColorStop(0, '#78350f');
        grad.addColorStop(1, '#d97706');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 400, 300);

        ctx.fillStyle = '#fef3c7';
        ctx.beginPath();
        ctx.arc(200, 150, 75, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#b45309';
        ctx.beginPath();
        ctx.ellipse(140, 150, 25, 45, Math.PI / 6, 0, Math.PI * 2);
        ctx.ellipse(260, 150, 25, 45, -Math.PI / 6, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#1e1b4b';
        ctx.beginPath();
        ctx.arc(175, 135, 9, 0, Math.PI * 2);
        ctx.arc(225, 135, 9, 0, Math.PI * 2);
        ctx.arc(200, 170, 14, 0, Math.PI * 2);
        ctx.fill();
    } else if (type === 'car') {
        // Sports car aesthetic
        const grad = ctx.createLinearGradient(0, 0, 400, 300);
        grad.addColorStop(0, '#0f172a');
        grad.addColorStop(1, '#334155');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 400, 300);

        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.roundRect(80, 140, 240, 60, 16);
        ctx.fill();

        ctx.fillStyle = '#b91c1c';
        ctx.beginPath();
        ctx.roundRect(130, 95, 140, 50, 10);
        ctx.fill();

        ctx.fillStyle = '#18181b';
        ctx.beginPath();
        ctx.arc(130, 205, 24, 0, Math.PI * 2);
        ctx.arc(270, 205, 24, 0, Math.PI * 2);
        ctx.fill();
    } else if (type === 'coffee') {
        // Espresso Cup aesthetic
        const grad = ctx.createLinearGradient(0, 0, 400, 300);
        grad.addColorStop(0, '#292524');
        grad.addColorStop(1, '#44403c');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 400, 300);

        ctx.fillStyle = '#f5f5f4';
        ctx.beginPath();
        ctx.arc(200, 160, 70, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#451a03';
        ctx.beginPath();
        ctx.arc(200, 160, 55, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#d97706';
        ctx.beginPath();
        ctx.arc(200, 160, 40, 0, Math.PI * 2);
        ctx.fill();
    } else {
        // Mountain Peak
        const grad = ctx.createLinearGradient(0, 0, 0, 300);
        grad.addColorStop(0, '#0284c7');
        grad.addColorStop(0.6, '#38bdf8');
        grad.addColorStop(1, '#f8fafc');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 400, 300);

        ctx.fillStyle = '#334155';
        ctx.beginPath();
        ctx.moveTo(80, 270);
        ctx.lineTo(200, 90);
        ctx.lineTo(320, 270);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.moveTo(170, 135);
        ctx.lineTo(200, 90);
        ctx.lineTo(230, 135);
        ctx.lineTo(200, 145);
        ctx.fill();
    }

    return new Promise(resolve => {
        canvas.toBlob(blob => {
            const file = new File([blob], `sample_${type}.png`, { type: 'image/png' });
            resolve(file);
        }, 'image/png');
    });
}

async function loadSamplePreset(type) {
    const file = await generateSampleImageBlob(type);
    loadImage(file);

    if (type === 'dog') {
        candidateTags = ['golden retriever', 'siamese cat', 'bengal tiger', 'brown bear'];
    } else if (type === 'car') {
        candidateTags = ['sports car', 'pickup truck', 'mountain bicycle', 'commercial airliner'];
    } else if (type === 'coffee') {
        candidateTags = ['espresso coffee', 'green tea', 'chocolate milkshake', 'glass of water'];
    } else {
        candidateTags = ['alpine mountain peak', 'tropical beach', 'dense pine forest', 'ocean waves'];
    }
    syncTagsToTextarea();
    renderTags();
    showError('request-error', '');
}

/* ==========================================================================
   Classification Inference Pipeline
   ========================================================================== */

function updateLoading(loading) {
    isLoading = loading;
    const runButton = document.querySelector('#run-button-host button');
    if (runButton) {
        runButton.textContent = loading ? 'Neural inference…' : 'Identify image →';
        runButton.setAttribute('aria-busy', String(loading));
    }
    document.querySelectorAll('button:not(#close-diagnostics-btn), input, textarea, select').forEach(function (control) {
        control.disabled = loading;
    });
    updateActionState();
}

async function identifyImage() {
    clearErrors();
    if (!selectedFile) {
        showError('upload-error', 'Choose or upload a query image first.');
        return;
    }

    if (!backendAvailable && !demoModeActive) {
        showError('request-error', 'Local Flask backend is offline. Run: .\\.venv\\Scripts\\python.exe app.py or switch on Demo Mode.');
        return;
    }

    const thresholdVal = Number(document.getElementById('confidence-threshold').value) / 100;
    const promptTemplate = document.getElementById('prompt-template').value;
    const queryFilename = selectedFile.name;

    updateLoading(true);
    resultState.textContent = 'Running CLIP inference…';

    // If Demo Mode is explicitly active, simulate realistic inference
    if (demoModeActive && !backendAvailable) {
        setTimeout(() => {
            simulateDemoInference(queryFilename, thresholdVal, promptTemplate);
            updateLoading(false);
        }, 650);
        return;
    }

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('threshold', String(thresholdVal));
    let endpoint;

    if (currentMode === 'zero-shot') {
        let labels;
        try {
            labels = parseCandidateLabels();
        } catch (error) {
            showError('request-error', error.message);
            updateLoading(false);
            return;
        }
        formData.append('labels', JSON.stringify(labels));
        formData.append('prompt_template', promptTemplate);
        endpoint = getApiBaseUrl() + '/api/classify/zero-shot';
    } else {
        if (examples.length === 0) {
            showError('request-error', 'Add at least one labeled example image before running few-shot classification.');
            updateLoading(false);
            return;
        }
        examples.forEach(function (example) {
            formData.append('examples', example.file);
            formData.append('example_labels', example.label);
        });
        endpoint = getApiBaseUrl() + '/api/classify/few-shot';
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        let data;
        const textResponse = await response.text();
        try {
            data = JSON.parse(textResponse);
        } catch (parseError) {
            throw new Error(`Server returned status ${response.status}: ${textResponse.slice(0, 100)}`);
        }

        if (!response.ok) {
            throw new Error(data.error?.message || `Classification failed with HTTP ${response.status}`);
        }

        renderResults(data);
        savePrediction(data, queryFilename);
        resultState.textContent = `${data.results.length} result${data.results.length === 1 ? '' : 's'} · ${data.mode}`;
        runNote.textContent = data.filtered_count 
            ? `${data.filtered_count} result(s) hidden by score threshold` 
            : 'Ranked by CLIP cosine similarity & softmax';
        await checkBackend();
    } catch (error) {
        resultState.textContent = 'Classification error';
        showError('request-error', error.message || 'Could not complete inference with local backend.');
    } finally {
        updateLoading(false);
    }
}

/* ==========================================================================
   Simulated Inference (For Demo Mode)
   ========================================================================== */

function simulateDemoInference(queryFilename, threshold, promptTemplate) {
    const labels = candidateTags.length > 0 ? candidateTags : ['sample label 1', 'sample label 2'];
    // Generate pseudo-logits
    const pseudoScores = labels.map((label, idx) => {
        const similarity = Math.max(0.1, 0.45 - idx * 0.08 + (Math.random() * 0.05));
        const logit = similarity * 10;
        return { label, similarity, logit };
    });

    const expScores = pseudoScores.map(item => Math.exp(item.logit));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const allResults = pseudoScores.map((item, idx) => ({
        label: item.label,
        similarity: parseFloat(item.similarity.toFixed(4)),
        logit: parseFloat(item.logit.toFixed(4)),
        score: parseFloat((expScores[idx] / sumExp).toFixed(4))
    })).sort((a, b) => b.score - a.score);

    const filtered = allResults.filter(r => r.score >= threshold);

    const mockData = {
        model_id: MODEL_ID,
        mode: currentMode,
        prompt_template: promptTemplate,
        threshold: threshold,
        results: filtered,
        all_results: allResults,
        filtered_count: allResults.length - filtered.length,
        score_note: "Demo Mode ranking score (simulated)"
    };

    renderResults(mockData);
    savePrediction(mockData, queryFilename + ' (Demo)');
    resultState.textContent = `${filtered.length} result${filtered.length === 1 ? '' : 's'} (Demo)`;
    runNote.textContent = 'Demo Mode output';
}

/* ==========================================================================
   Results Rendering & Spotlight Display
   ========================================================================== */

function renderResults(data) {
    const list = document.getElementById('results-list');
    list.replaceChildren();

    if (!data.results || data.results.length === 0) {
        if (topMatchHero) topMatchHero.style.display = 'none';
        const empty = document.createElement('p');
        empty.className = 'empty-state';
        empty.textContent = 'No predictions meet the current confidence threshold. Lower the threshold slider to view more rankings.';
        list.appendChild(empty);
        return;
    }

    // Top Result Spotlight
    const top = data.results[0];
    if (topMatchHero && topMatchLabel && topMatchScore) {
        topMatchLabel.textContent = top.label;
        topMatchScore.textContent = (top.score * 100).toFixed(1) + '%';
        topMatchHero.style.display = 'flex';
    }

    data.results.forEach(function (result, index) {
        const row = document.createElement('div');
        row.className = 'result-row';

        const rank = document.createElement('span');
        rank.className = 'rank';
        rank.textContent = `#${String(index + 1).padStart(2, '0')}`;

        const details = document.createElement('div');
        const name = document.createElement('div');
        name.className = 'result-name';

        const label = document.createElement('span');
        label.textContent = result.label;

        const scorePercent = (result.score * 100).toFixed(1) + '%';
        const scoreSpan = document.createElement('span');
        scoreSpan.textContent = scorePercent;

        name.append(label, scoreSpan);

        const bar = document.createElement('div');
        bar.className = 'bar';
        const fill = document.createElement('span');
        fill.style.width = Math.max(0, Math.min(100, result.score * 100)) + '%';
        bar.appendChild(fill);
        details.append(name, bar);

        const scoreText = document.createElement('span');
        scoreText.className = 'confidence';
        scoreText.textContent = scorePercent;
        scoreText.title = `Softmax Score: ${scorePercent} · Cosine: ${result.similarity.toFixed(4)} · Scaled Logit: ${result.logit.toFixed(4)}`;

        row.append(rank, details, scoreText);
        list.appendChild(row);
    });

    const caveat = document.getElementById('result-caveat');
    if (caveat) {
        caveat.textContent = `${data.score_note || 'Ranked by CLIP.'} Hover over percentages to view cosine similarity and scaled logits.`;
    }
}

/* ==========================================================================
   Session History & Export System
   ========================================================================== */

function loadHistory() {
    try {
        const parsed = JSON.parse(sessionStorage.getItem(HISTORY_KEY) || '[]');
        return Array.isArray(parsed) ? parsed : [];
    } catch (error) {
        return [];
    }
}

function savePrediction(data, queryFilename) {
    history.unshift({
        timestamp: new Date().toISOString(),
        model_id: data.model_id,
        mode: data.mode,
        query_name: queryFilename,
        prompt_template: data.prompt_template || null,
        threshold: data.threshold,
        results: data.results,
        all_results: data.all_results
    });
    history = history.slice(0, 50);
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistory();
}

function renderHistory() {
    const list = document.getElementById('history-list');
    if (!list) return;
    list.replaceChildren();

    const countStat = document.getElementById('session-prediction-count');
    if (countStat) countStat.textContent = String(history.length);

    const summary = document.getElementById('history-summary');
    if (summary) summary.textContent = `${history.length} prediction${history.length === 1 ? '' : 's'}`;

    if (history.length === 0) {
        const empty = document.createElement('p');
        empty.className = 'empty-state';
        empty.textContent = 'Actual predictions performed in this tab will appear here.';
        list.appendChild(empty);
        updateActionState();
        return;
    }

    history.slice(0, 10).forEach(function (entry) {
        const item = document.createElement('div');
        item.className = 'history-item';

        const info = document.createElement('div');
        const title = document.createElement('strong');
        const topResult = entry.results && entry.results[0];
        title.textContent = `${entry.query_name} → ${topResult ? topResult.label : 'None'}`;

        const date = document.createElement('small');
        date.textContent = `${new Date(entry.timestamp).toLocaleTimeString()} · ${entry.mode}`;
        info.append(title, date);

        const score = document.createElement('span');
        score.className = 'history-score';
        score.textContent = topResult ? (topResult.score * 100).toFixed(1) + '%' : '—';

        item.append(info, score);
        list.appendChild(item);
    });
    updateActionState();
}

function exportHistory(format) {
    if (!history.length) return;
    let content;
    let mimeType;
    let extension;
    if (format === 'json') {
        content = JSON.stringify({ model_id: MODEL_ID, exported_at: new Date().toISOString(), predictions: history }, null, 2);
        mimeType = 'application/json';
        extension = 'json';
    } else {
        const rows = [['timestamp', 'model_id', 'mode', 'query_name', 'prompt_template', 'threshold', 'rank', 'label', 'similarity', 'logit', 'score']];
        history.forEach(function (entry) {
            (entry.all_results || entry.results || []).forEach(function (result, index) {
                rows.push([
                    entry.timestamp,
                    entry.model_id,
                    entry.mode,
                    entry.query_name,
                    entry.prompt_template || '',
                    entry.threshold,
                    index + 1,
                    result.label,
                    result.similarity,
                    result.logit,
                    result.score
                ]);
            });
        });
        content = rows.map(row => row.map(csvCell).join(',')).join('\r\n');
        mimeType = 'text/csv;charset=utf-8';
        extension = 'csv';
    }
    const blob = new Blob([content], { type: mimeType });
    const link = document.createElement('a');
    const downloadUrl = URL.createObjectURL(blob);
    link.href = downloadUrl;
    link.download = `clip-predictions-${Date.now()}.${extension}`;
    link.click();
    setTimeout(function () { URL.revokeObjectURL(downloadUrl); }, 1000);
}

function csvCell(value) {
    let text = String(value ?? '');
    if (typeof value === 'string' && /^[\t\r\n ]*[=+@-]/.test(text)) text = "'" + text;
    return '"' + text.replace(/"/g, '""') + '"';
}

/* ==========================================================================
   Few-Shot Prototypes Management
   ========================================================================== */

function renderExamples() {
    const list = document.getElementById('example-list');
    if (!list) return;
    list.replaceChildren();

    const countStat = document.getElementById('example-count');
    if (countStat) countStat.textContent = String(examples.length);

    const countLabel = document.getElementById('example-count-label');
    if (countLabel) countLabel.textContent = `${examples.length} / ${MAX_EXAMPLES}`;

    if (examples.length === 0) {
        const empty = document.createElement('p');
        empty.className = 'empty-state';
        empty.textContent = 'No example images added yet.';
        list.appendChild(empty);
        return;
    }

    examples.forEach(function (example, index) {
        const row = document.createElement('div');
        row.className = 'example';

        const image = document.createElement('img');
        image.className = 'example-thumb';
        image.src = example.previewUrl;
        image.alt = example.label;

        const text = document.createElement('div');
        text.className = 'example-text';
        const label = document.createElement('strong');
        label.textContent = example.label;
        const filename = document.createElement('span');
        filename.textContent = example.file.name;
        text.append(label, filename);

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'remove';
        removeBtn.innerHTML = '&times;';
        removeBtn.title = 'Remove this example';
        removeBtn.onclick = function () {
            URL.revokeObjectURL(example.previewUrl);
            examples.splice(index, 1);
            renderExamples();
        };

        row.append(image, text, removeBtn);
        list.appendChild(row);
    });
}

/* ==========================================================================
   Dashboard Reset & Clipboard Paste
   ========================================================================== */

function resetDashboard() {
    clearErrors();
    fileInput.value = '';
    selectedFile = null;
    if (selectedPreviewUrl) URL.revokeObjectURL(selectedPreviewUrl);
    selectedPreviewUrl = null;
    previewImage.removeAttribute('src');
    uploadZone.hidden = false;
    imagePreview.style.display = 'none';

    candidateTags = ['cat', 'dog', 'sports car', 'coffee cup', 'mountain landscape'];
    syncTagsToTextarea();
    renderTags();

    const exampleLabel = document.getElementById('example-label-input');
    if (exampleLabel) exampleLabel.value = '';

    examples.forEach(example => URL.revokeObjectURL(example.previewUrl));
    examples.splice(0, examples.length);
    renderExamples();

    history = [];
    sessionStorage.removeItem(HISTORY_KEY);
    renderHistory();

    const list = document.getElementById('results-list');
    if (list) {
        list.innerHTML = '<p class="empty-state">Upload or pick a sample image and click "Identify image →" to see real-time CLIP rankings.</p>';
    }
    if (topMatchHero) topMatchHero.style.display = 'none';
    resultState.textContent = 'Waiting for image';
    setMode('zero-shot');
    updateActionState();
    checkBackend();
}

// Clipboard Paste Handler (Ctrl+V)
window.addEventListener('paste', function (event) {
    const items = (event.clipboardData || event.originalEvent.clipboardData).items;
    for (let index = 0; index < items.length; index++) {
        const item = items[index];
        if (item.kind === 'file' && item.type.startsWith('image/')) {
            const blob = item.getAsFile();
            loadImage(blob);
            break;
        }
    }
});

// Keyboard Shortcut: Ctrl+Enter to Run
window.addEventListener('keydown', function (event) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        const runBtn = document.querySelector('#run-button-host button');
        if (runBtn && !runBtn.disabled) {
            identifyImage();
        }
    }
});

/* ==========================================================================
   Event Listeners & Initialization
   ========================================================================== */

// File Upload Listeners
fileInput.addEventListener('change', function () {
    if (fileInput.files && fileInput.files[0]) {
        loadImage(fileInput.files[0]);
    }
});

uploadZone.addEventListener('click', function () {
    fileInput.click();
});

uploadZone.addEventListener('dragover', function (event) {
    event.preventDefault();
    uploadZone.classList.add('dragging');
});

uploadZone.addEventListener('dragleave', function () {
    uploadZone.classList.remove('dragging');
});

uploadZone.addEventListener('drop', function (event) {
    event.preventDefault();
    uploadZone.classList.remove('dragging');
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
        loadImage(event.dataTransfer.files[0]);
    }
});

const changeImgBtn = document.getElementById('change-image-btn');
if (changeImgBtn) {
    changeImgBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        fileInput.click();
    });
}

// Example File Upload Listener
exampleFileInput.addEventListener('change', function () {
    const file = exampleFileInput.files[0];
    if (!file) return;
    const error = validImage(file);
    const labelInput = document.getElementById('example-label-input');
    const label = labelInput ? labelInput.value.trim().replace(/\s+/g, ' ') : '';
    if (error) {
        showError('example-error', error);
    } else if (!label || label.length > 80) {
        showError('example-error', 'Enter a class label between 1 and 80 characters.');
    } else {
        showError('example-error', '');
        examples.push({ file: file, label: label, previewUrl: URL.createObjectURL(file) });
        if (labelInput) labelInput.value = '';
        renderExamples();
    }
    exampleFileInput.value = '';
});

// Confidence Threshold Slider
const thresholdSlider = document.getElementById('confidence-threshold');
if (thresholdSlider) {
    thresholdSlider.addEventListener('input', function (event) {
        const val = event.target.value + '%';
        const out = document.getElementById('threshold-value');
        if (out) {
            out.value = val;
            out.textContent = val;
        }
    });
}

// Candidate Tags Keydown (Enter or Comma)
if (candidateTagInput) {
    candidateTagInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            const val = candidateTagInput.value.replace(/,/g, '');
            addCandidateTag(val);
            candidateTagInput.value = '';
        }
    });

    candidateTagInput.addEventListener('blur', function () {
        if (candidateTagInput.value.trim()) {
            addCandidateTag(candidateTagInput.value.replace(/,/g, ''));
            candidateTagInput.value = '';
        }
    });
}

// Toggle Raw Textarea
const toggleRawBtn = document.getElementById('toggle-raw-btn');
if (toggleRawBtn && candidateInput) {
    toggleRawBtn.addEventListener('click', function () {
        const isShown = candidateInput.style.display === 'block';
        candidateInput.style.display = isShown ? 'none' : 'block';
        if (!isShown) candidateInput.focus();
    });
    candidateInput.addEventListener('input', function () {
        syncTextareaToTags();
    });
}

// Category Presets
document.querySelectorAll('.quick-category-btn').forEach(btn => {
    btn.addEventListener('click', function () {
        const cat = btn.getAttribute('data-cat');
        if (PRESET_CATEGORIES[cat]) {
            candidateTags = [...PRESET_CATEGORIES[cat]];
            syncTagsToTextarea();
            renderTags();
        }
    });
});

// Sample Image Presets
document.querySelectorAll('.preset-chip').forEach(chip => {
    chip.addEventListener('click', function () {
        const sampleType = chip.getAttribute('data-sample');
        if (sampleType) {
            loadSamplePreset(sampleType);
        }
    });
});

// Diagnostics Modal & URL Customization
const diagModal = document.getElementById('diagnostics-modal');
const openDiagBtn = document.getElementById('open-diagnostics-btn');
const closeDiagBtn = document.getElementById('close-diagnostics-btn');
const statusBtn = document.getElementById('backend-status-btn');
const saveApiBtn = document.getElementById('save-api-url-btn');
const apiUrlInput = document.getElementById('api-url-input');
const retryBackendBtn = document.getElementById('retry-backend-btn');
const copyStartCmd = document.getElementById('copy-start-cmd');
const demoModeBtn = document.getElementById('demo-mode-btn');

function openModal() {
    if (apiUrlInput) apiUrlInput.value = getApiBaseUrl();
    if (diagModal) diagModal.classList.add('open');
    updatePingDisplay();
}

function closeModal() {
    if (diagModal) diagModal.classList.remove('open');
}

async function updatePingDisplay() {
    const pingBox = document.getElementById('modal-ping-result');
    if (!pingBox) return;
    pingBox.textContent = 'Pinging API endpoint…';
    const baseUrl = getApiBaseUrl();
    const t0 = performance.now();
    try {
        const res = await fetch(baseUrl + '/api/health');
        const dt = Math.round(performance.now() - t0);
        if (res.ok) {
            pingBox.textContent = `✅ Connected (${dt}ms) · Status: OK`;
            pingBox.style.color = 'var(--accent-emerald)';
        } else {
            pingBox.textContent = `⚠️ Returned HTTP ${res.status}`;
            pingBox.style.color = 'var(--accent-amber)';
        }
    } catch (e) {
        pingBox.textContent = `❌ Offline: Could not connect to ${baseUrl}`;
        pingBox.style.color = 'var(--accent-rose)';
    }
}

if (statusBtn) statusBtn.addEventListener('click', openModal);
if (openDiagBtn) openDiagBtn.addEventListener('click', openModal);
if (closeDiagBtn) closeDiagBtn.addEventListener('click', closeModal);
if (diagModal) {
    diagModal.addEventListener('click', function (e) {
        if (e.target === diagModal) closeModal();
    });
}

if (saveApiBtn && apiUrlInput) {
    saveApiBtn.addEventListener('click', function () {
        const url = apiUrlInput.value.trim();
        if (url) {
            localStorage.setItem(API_URL_KEY, url);
        } else {
            localStorage.removeItem(API_URL_KEY);
        }
        checkBackend();
        updatePingDisplay();
    });
}

if (retryBackendBtn) {
    retryBackendBtn.addEventListener('click', function () {
        checkBackend();
    });
}

if (copyStartCmd) {
    copyStartCmd.addEventListener('click', function () {
        navigator.clipboard.writeText('.\\.venv\\Scripts\\python.exe app.py').then(() => {
            const small = copyStartCmd.querySelector('small');
            if (small) {
                small.textContent = '✓ copied!';
                setTimeout(() => { small.textContent = '📋 copy'; }, 1500);
            }
        });
    });
}

if (demoModeBtn) {
    demoModeBtn.addEventListener('click', function () {
        demoModeActive = !demoModeActive;
        demoModeBtn.classList.toggle('primary', demoModeActive);
        demoModeBtn.querySelector('span').textContent = demoModeActive ? '🧪 Demo Active' : '🧪 Demo Mode';
        checkBackend();
    });
}

// App Boot
createButtons();
syncTagsToTextarea();
renderTags();
renderHistory();
renderExamples();
checkBackend();

// Poll backend status
pollIntervalTimer = setInterval(function () {
    if (!isLoading) {
        checkBackend();
    }
}, 5000);
