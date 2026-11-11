(function () {
  const I = window.INFO, R = I.results, names = Object.keys(R);
  const cs = getComputedStyle(document.documentElement);
  const mute = cs.getPropertyValue('--mute').trim(), line = cs.getPropertyValue('--line').trim();
  Chart.defaults.color = mute; Chart.defaults.borderColor = line;
  Chart.defaults.font.family = 'system-ui, sans-serif';
  const pal = ['#4f8cff', '#2fc58a', '#f5b84a', '#ff5d6c', '#a78bfa', '#22d3ee'];
  const mk = (id, cfg) => new Chart(document.getElementById(id), cfg);

  mk('cMetrics', {
    type: 'bar',
    data: {
      labels: names,
      datasets: [['Precision', 'precision'], ['Recall', 'recall'], ['F1', 'f1'], ['ROC-AUC', 'roc_auc']]
        .map(([l, k], i) => ({ label: l, data: names.map(n => R[n][k]), backgroundColor: pal[i], borderRadius: 3 }))
    },
    options: { scales: { y: { min: 0, max: 1 } }, plugins: { legend: { position: 'bottom' } } }
  });

  mk('cRoc', {
    type: 'line',
    data: {
      datasets: Object.keys(I.roc).map((n, i) => ({
        label: n, borderColor: pal[i], backgroundColor: pal[i], borderWidth: 2, pointRadius: 0,
        data: I.roc[n].fpr.map((x, j) => ({ x, y: I.roc[n].tpr[j] }))
      })).concat([{ label: 'Chance', borderColor: mute, borderDash: [5, 5], borderWidth: 1, pointRadius: 0, data: [{ x: 0, y: 0 }, { x: 1, y: 1 }] }])
    },
    options: {
      parsing: false, scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False positive rate' } },
        y: { min: 0, max: 1, title: { display: true, text: 'True positive rate' } }
      }, plugins: { legend: { position: 'bottom' } }
    }
  });

  mk('cHour', {
    type: 'bar',
    data: { labels: [...Array(24).keys()], datasets: [{ data: I.by_hour, backgroundColor: I.by_hour.map(v => v > 20 ? '#ff5d6c' : '#4f8cff'), borderRadius: 3 }] },
    options: { plugins: { legend: { display: false } } }
  });

  mk('cCat', {
    type: 'bar',
    data: { labels: I.by_category.labels.map(s => s.replace('_', ' ')), datasets: [{ data: I.by_category.values, backgroundColor: '#f5b84a', borderRadius: 3 }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } } }
  });

  mk('cImp', {
    type: 'bar',
    data: { labels: I.importance.labels, datasets: [{ data: I.importance.values, backgroundColor: '#2fc58a', borderRadius: 3 }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } } }
  });

  mk('cBma', {
    type: 'doughnut',
    data: { labels: Object.keys(I.bma_weights), datasets: [{ data: Object.values(I.bma_weights).map(v => +(v * 100).toFixed(1)), backgroundColor: pal, borderColor: 'transparent' }] },
    options: { plugins: { legend: { position: 'bottom' }, tooltip: { callbacks: { label: c => ` ${c.label}: ${c.raw}%` } } } }
  });

  const cm = document.getElementById('cm'), sel = document.getElementById('cmSel');
  function drawCM() {
    const [[tn, fp], [fn, tp]] = R[sel.value].confusion;
    cm.innerHTML = `<div class="ok"><b>${tn}</b>True negative</div><div class="no"><b>${fp}</b>False positive</div>
                    <div class="no"><b>${fn}</b>False negative (missed fraud)</div><div class="ok"><b>${tp}</b>True positive</div>`;
  }
  sel.addEventListener('change', drawCM); sel.value = 'Voting Classifier'; drawCM();
})();
