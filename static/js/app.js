document.addEventListener('DOMContentLoaded', () => {
  // ── Auto-dismiss flash alerts ──
  setTimeout(() => {
    document.querySelectorAll('.alert').forEach(el => {
      el.style.transition = 'opacity 0.3s ease';
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 300);
    });
  }, 5000);

  // ── Dashboard / History: Clickable experiment rows ──
  document.querySelectorAll('.experiment-row, .history-row').forEach(row => {
    row.addEventListener('click', (e) => {
      if (e.target.closest('a') || e.target.closest('button')) return;
      const href = row.dataset.href;
      if (href) window.location.href = href;
    });
    row.style.cursor = 'pointer';
  });

  // ── Builder: Mode toggle ──
  window.toggleMode = function(mode) {
    const runModeInput = document.getElementById('run_mode_input');
    if (runModeInput) runModeInput.value = mode;

    const featureSection = document.getElementById('feature-selection');
    const modelSection = document.getElementById('model-selection');

    if (mode === 'manual') {
      if (featureSection) featureSection.style.display = 'block';
      if (modelSection) modelSection.style.display = 'block';
      onTargetChange();
    } else {
      if (featureSection) featureSection.style.display = 'none';
      if (modelSection) modelSection.style.display = 'none';
    }
  };

  // ── Builder: Target column change → detect problem type ──
  window.onTargetChange = function() {
    const targetCol = document.getElementById('target_column');
    const modelHint = document.getElementById('model-hint');
    const clfModels = document.getElementById('classification-models');
    const regModels = document.getElementById('regression-models');

    if (!targetCol || !targetCol.value) {
      if (clfModels) clfModels.style.display = 'none';
      if (regModels) regModels.style.display = 'none';
      if (modelHint) {
        modelHint.textContent = 'Select a target column above to see available algorithms.';
        modelHint.style.display = 'block';
      }
      return;
    }

    const selectedOption = targetCol.options[targetCol.selectedIndex];
    const colName = selectedOption.text;

    const featureTable = document.querySelector('.feature-table tbody');
    let isNumeric = false;
    let uniqueCount = 0;

    if (featureTable) {
      const rows = featureTable.querySelectorAll('tr');
      for (const row of rows) {
        const nameCell = row.querySelector('td:nth-child(2)');
        if (nameCell && nameCell.textContent.trim() === colName) {
          const typeCell = row.querySelector('td:nth-child(3) .dtype-badge');
          if (typeCell) {
            isNumeric = typeCell.classList.contains('dtype-numeric');
          }
          const uniqueCell = row.querySelector('td:nth-child(4) .unique-count');
          if (uniqueCell) uniqueCount = parseInt(uniqueCell.textContent) || 0;
          break;
        }
      }
    }

    const isClassification = !isNumeric || uniqueCount <= 10;

    if (isClassification) {
      if (clfModels) clfModels.style.display = 'block';
      if (regModels) regModels.style.display = 'none';
      if (modelHint) modelHint.style.display = 'none';
    } else {
      if (clfModels) clfModels.style.display = 'none';
      if (regModels) regModels.style.display = 'block';
      if (modelHint) modelHint.style.display = 'none';
    }
  };

  // ── Builder: Select All / Deselect All features ──
  window.selectAllFeatures = function(checked) {
    const checkboxes = document.querySelectorAll('.feature-checkbox');
    checkboxes.forEach(cb => cb.checked = checked);
    const selectAllBox = document.getElementById('select-all-features');
    if (selectAllBox) selectAllBox.checked = checked;
  };

  // ── Builder: Show log panel when form submits — NON-BLOCKING ──
  const trainForm = document.getElementById('train-form');
  const trainBtn = document.getElementById('train-btn');
  const logPanel = document.getElementById('training-logs');
  const logTerminal = document.getElementById('log-terminal');

  if (trainForm && trainBtn) {
    trainBtn.addEventListener('click', function() {
      // Show log panel immediately for visual feedback
      if (logPanel) logPanel.style.display = 'block';
      if (logTerminal) {
        logTerminal.innerHTML = '<div class="log-line log-info">⏳ Initializing pipeline...</div>';
      }
      // Change button text visually but DO NOT disable or prevent submission
      trainBtn.innerHTML = '<span class="spinner"></span> Training in progress...';
      trainBtn.style.opacity = '0.85';
      // Form submits normally after this handler completes
    });
  }

  // ── Results: Animate performance bars on load ──
  const perfBars = document.querySelectorAll('.perf-bar-fill[data-width]');
  if (perfBars.length > 0) {
    setTimeout(() => {
      perfBars.forEach(bar => {
        const width = bar.dataset.width;
        if (width !== undefined && width !== '') {
          const w = parseFloat(width);
          const clamped = Math.max(0, Math.min(100, w));
          bar.style.width = clamped + '%';
        }
      });
    }, 300);
  }

  // ── Results: Smooth scroll to sections ──
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  // ── Initialize mode on page load ──
  const modeAuto = document.getElementById('mode_auto');
  if (modeAuto && modeAuto.checked) {
    toggleMode('auto');
  }
});