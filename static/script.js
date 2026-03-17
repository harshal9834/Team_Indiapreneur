document.addEventListener('DOMContentLoaded', () => {
  /* Selectors */
  const dropZone        = document.getElementById('dropZone');
  const fileInput       = document.getElementById('fileInput');
  const uploadIdle      = document.getElementById('uploadIdle');
  const uploadPreview   = document.getElementById('uploadPreview');
  const previewImg      = document.getElementById('previewImg');
  const previewVid      = document.getElementById('previewVid');
  const previewAud      = document.getElementById('previewAud');
  const originalView    = document.getElementById('originalView');
  const heatmapToggle   = document.getElementById('heatmapToggle');
  const previewFilename = document.getElementById('previewFilename');
  const changeBtn       = document.getElementById('changeBtn');
  const analyzeBtn      = document.getElementById('analyzeBtn');

  const initialState    = document.getElementById('initialState');
  const loaderCard      = document.getElementById('loaderCard');
  const resultCard      = document.getElementById('resultCard');
  
  const procTime        = document.getElementById('procTime');
  const engineName      = document.getElementById('engineName');
  const confValue       = document.getElementById('confValue');
  const trustScore      = document.getElementById('trustScore');
  
  const explainList     = document.getElementById('explainList');
  const reasoningBox    = document.getElementById('reasoningBox');
  const techStackList   = document.getElementById('techStackList');
  const recList         = document.getElementById('recList');
  const auditTrail      = document.getElementById('auditTrail');
  const resetBtn        = document.getElementById('resetBtn');

  let selectedFile = null;

  /* File Handling */
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

  dropZone.addEventListener('dragover', e => { 
    e.preventDefault(); 
    dropZone.classList.add('drag-active'); 
  });
  ['dragleave', 'drop'].forEach(evt => dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-active')));
  dropZone.addEventListener('drop', e => { 
    e.preventDefault(); 
    handleFile(e.dataTransfer.files[0]); 
  });

  changeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    const reader = new FileReader();

    if (file.type.startsWith('image/')) {
      reader.onload = e => { 
        previewImg.src = e.target.result; 
        originalView.src = e.target.result;
        previewImg.classList.remove('hidden'); 
        previewVid.classList.add('hidden'); 
        previewAud.classList.add('hidden'); 
      };
      reader.readAsDataURL(file);
    } else if (file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file);
      previewVid.src = url;
      previewVid.classList.remove('hidden');
      previewImg.classList.add('hidden');
      previewAud.classList.add('hidden');
      /* For video, we don't have a static originalView yet in results, 
         using first frame/placeholder might be better but for now URL is fine */
      originalView.src = ""; 
    } else if (file.type.startsWith('audio/')) {
      previewAud.src = URL.createObjectURL(file);
      previewAud.classList.remove('hidden');
      previewImg.classList.add('hidden');
      previewVid.classList.add('hidden');
    }

    previewFilename.textContent = file.name;
    uploadIdle.classList.add('hidden');
    uploadPreview.classList.remove('hidden');
    analyzeBtn.disabled = false;
    
    /* Reset results if changing file */
    resultCard.classList.add('hidden');
    initialState.classList.remove('hidden');
  }

  /* Analysis */
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    analyzeBtn.disabled = true;
    initialState.classList.add('hidden');
    resultCard.classList.add('hidden');
    loaderCard.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const startTime = Date.now();
      const response = await fetch('/detect', { method: 'POST', body: formData });
      const data = await response.json();
      
      if (response.ok) {
        renderResult(data, selectedFile.name);
      } else {
        alert(data.error || "Analysis failed.");
        resetToInitial();
      }
    } catch (err) {
      console.error(err);
      alert('Network error - ensure backend is running.');
      resetToInitial();
    }
  });

  function renderResult(data, filename) {
    const f = data.forensic_analysis;
    const s = data.summary;
    const isFake = s.prediction === 'FAKE';
    const riskLevelStr = s.risk_level.toUpperCase();
    const riskClass = riskLevelStr.includes('HIGH') ? 'HIGH' : (riskLevelStr.includes('MEDIUM') ? 'MEDIUM' : 'LOW');

    loaderCard.classList.add('hidden');
    resultCard.classList.remove('hidden');

    // 1. Final Verdict Header
    document.getElementById('verdictTitle').textContent = f.verdict_header.title;
    document.getElementById('confidenceStatement').textContent = f.verdict_header.statement;
    const verdictCard = document.getElementById('verdictCard');
    verdictCard.className = `card verdict-card ${s.prediction}`;

    // 2. Summary Cards
    document.getElementById('statPrediction').textContent = s.prediction;
    document.getElementById('statConfidence').textContent = `${s.confidence}%`;
    document.getElementById('statRisk').textContent = s.risk_level;
    document.getElementById('statTrust').textContent = s.trust_score;

    // 3. Confidence Meter
    const fill = document.getElementById('signalMeterFill');
    const val = document.getElementById('signalValue');
    fill.style.width = `${data.fake_signal_strength}%`;
    val.textContent = `${data.fake_signal_strength}%`;

    // 4. Feature Analysis
    const featureList = document.getElementById('featureList');
    featureList.innerHTML = '';
    Object.entries(f.feature_analysis).forEach(([key, value]) => {
      const item = document.createElement('div');
      item.className = 'analysis-item';
      item.innerHTML = `
        <span class="analysis-key">${key}</span>
        <span class="analysis-val">${value}</span>
      `;
      featureList.appendChild(item);
    });

    // 5. Feature Table
    const tableBody = document.getElementById('featureTableBody');
    tableBody.innerHTML = '';
    f.feature_table.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${row.feature}</td>
        <td>${row.value}</td>
        <td>${row.interpretation}</td>
      `;
      tableBody.appendChild(tr);
    });

    // 6. Model Insights
    document.getElementById('cnnInsightText').textContent = f.model_insights.cnn;
    document.getElementById('temporalInsightText').textContent = f.model_insights.temporal;

    // 7. Key Observations
    const obsList = document.getElementById('observationsList');
    obsList.innerHTML = '';
    f.key_observations.forEach(obs => {
      const li = document.createElement('li');
      li.className = 'obs-item';
      li.textContent = obs;
      obsList.appendChild(li);
    });

    // 8. Final Explanation
    document.getElementById('finalExplanationText').textContent = f.explanation;

    // 9. Recommendations
    recList.innerHTML = '';
    const recCard = document.getElementById('recommendationCard');
    recCard.className = `card recommendation-card ${isFake ? 'FAKE' : 'SAFE'}`;
    data.recommendation.forEach(rec => {
      const div = document.createElement('div');
      div.className = 'rec-item';
      div.innerHTML = `<span class="rec-bullet">🛡️</span> <span>${rec}</span>`;
      recList.appendChild(div);
    });

    // 10. System Tags
    const tagsDiv = document.getElementById('systemTags');
    const artifactColor = isFake ? 'var(--color-danger)' : 'var(--color-safe)';
    tagsDiv.innerHTML = `
      <div class="sys-tag"><strong>AI Artifact:</strong> <span style="color:${artifactColor}">${f.system_tags.ai_artifact}</span></div>
      <div class="sys-tag"><strong>Type:</strong> ${f.system_tags.analysis_type}</div>
      <div class="sys-tag"><strong>Mode:</strong> ${f.system_tags.processing_mode}</div>
    `;

    // Metrics Panel Update
    procTime.textContent = data.system_info.processing_time;
    engineName.textContent = data.system_info.engine;
    confValue.textContent = `${s.confidence}%`;
    trustScore.textContent = `${s.trust_score}/100`;

    // Visual Evidence
    originalView.src = previewImg.src || (previewVid.src ? previewVid.poster : '');
    const heatmapOverlay = document.getElementById('heatmapOverlay');
    if (data.heatmap_base64) {
      heatmapOverlay.src = data.heatmap_base64;
      heatmapToggle.disabled = false;
      heatmapToggle.checked = true;
      heatmapOverlay.classList.remove('hidden');
    } else {
      heatmapToggle.disabled = true;
      heatmapOverlay.classList.add('hidden');
    }

    // Technology Attribution
    techStackList.innerHTML = '';
    data.stack_contribution.forEach((item) => {
      const div = document.createElement('div');
      div.className = 'tech-contribution-item';
      div.innerHTML = `
        <div class="tech-name">${item.tech}</div>
        <div class="tech-role">${item.role}</div>
      `;
      techStackList.appendChild(div);
    });

    // Audit Entry
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${timeStr}</td>
      <td class="filename">${filename}</td>
      <td><span style="color:${isFake ? 'var(--color-danger)' : 'var(--color-safe)'}; font-weight:800">${isFake ? 'FAKE' : 'REAL'}</span></td>
      <td><span class="status-cell ${riskClass}">${riskLevelStr}</span></td>
    `;
    auditTrail.prepend(row);
  }

  heatmapToggle.addEventListener('change', () => {
    document.getElementById('heatmapOverlay').classList.toggle('hidden', !heatmapToggle.checked);
  });

  function resetToInitial() {
    loaderCard.classList.add('hidden');
    resultCard.classList.add('hidden');
    initialState.classList.remove('hidden');
    analyzeBtn.disabled = false;
  }

  resetBtn.addEventListener('click', () => {
    uploadIdle.classList.remove('hidden');
    uploadPreview.classList.add('hidden');
    previewImg.classList.add('hidden');
    previewVid.classList.add('hidden');
    previewAud.classList.add('hidden');
    analyzeBtn.disabled = true;
    resetToInitial();
    
    /* Clear audit if user starts completely fresh session? 
       Usually audit persists in enterprise apps until logout/refresh. */
  });

  /* View Switching (Placeholder for Sidebar) */
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
    });
  });
});

function downloadReport() {
  window.open("/download-report-csv", "_blank");
}

function downloadPDF() {
  window.open("/download-report-pdf", "_blank");
}

function downloadDetailedReport() {
  window.open("/download-latest-report", "_blank");
}
