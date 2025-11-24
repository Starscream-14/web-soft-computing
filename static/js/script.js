document.addEventListener('DOMContentLoaded', function () {
	const navToggle = document.getElementById('navToggle');
	const siteNav = document.getElementById('siteNav');
	const navLinks = document.querySelectorAll('.nav-link');

	if (navToggle && siteNav) {
		navToggle.addEventListener('click', function () {
			const open = siteNav.classList.toggle('open');
			navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
		});

		window.addEventListener('keydown', function (e) {
			if (e.key === 'Escape') {
				siteNav.classList.remove('open');
				navToggle.setAttribute('aria-expanded', 'false');
			}
		});
	}

	navLinks.forEach(function (link) {
		link.addEventListener('click', function (e) {
			const href = link.getAttribute('href');
			if (href && href.startsWith('#')) {
				e.preventDefault();
				const target = document.querySelector(href);
				if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
			}

			if (siteNav) {
				siteNav.classList.remove('open');
				if (navToggle) navToggle.setAttribute('aria-expanded', 'false');
			}
		});
	});

	const sections = Array.from(document.querySelectorAll('main .section'));
	function setActiveLinkOnScroll() {
		const scrollPos = window.scrollY + window.innerHeight / 3;
		sections.forEach(function (sec) {
			const top = sec.offsetTop;
			const bottom = top + sec.offsetHeight;
			const id = sec.id;
			const link = document.querySelector('.nav-link[href="#' + id + '"]');
			if (!link) return;
			if (scrollPos >= top && scrollPos < bottom) {
				link.classList.add('active');
			} else {
				link.classList.remove('active');
			}
		});
	}

	setActiveLinkOnScroll();
	window.addEventListener('scroll', setActiveLinkOnScroll, { passive: true });

  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  const API_BASE = 'http://localhost:5000/api';

  function showLoading(resultId) {
    const el = document.getElementById(resultId);
    if (el) el.innerHTML = '<em>Memproses...</em>';
  }

  function showResult(resultId, html) {
    const el = document.getElementById(resultId);
    if (el) el.innerHTML = html;
  }

  function showError(resultId, message) {
    const el = document.getElementById(resultId);
    if (el) el.innerHTML = `<span style="color:#ff6b6b">Error: ${message}</span>`;
  }

  const fuzzyForm = document.getElementById('fuzzyForm');
  if (fuzzyForm) {
    fuzzyForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      showLoading('fuzzyResult');

      const formData = new FormData(fuzzyForm);
      const data = {
        temperature: parseFloat(formData.get('temperature')),
        humidity: parseFloat(formData.get('humidity'))
      };

      try {
        const response = await fetch(`${API_BASE}/fuzzy`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Backend tidak merespons');

        const result = await response.json();
        const html = `
          <strong>Hasil Fuzzy Logic:</strong><br/>
          Skor Kenyamanan: <strong>${result.comfort_score}/100</strong> (${result.comfort_label})<br/>
          <small>${result.explanation}</small>
        `;
        showResult('fuzzyResult', html);
      } catch (error) {
        showError('fuzzyResult', error.message + ' — Pastikan backend berjalan di port 5000');
      }
    });
  }

  const annForm = document.getElementById('annForm');
  if (annForm) {
    annForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      showLoading('annResult');

      const formData = new FormData(annForm);
      const data = {
        x1: parseFloat(formData.get('x1')),
        x2: parseFloat(formData.get('x2'))
      };

      try {
        const response = await fetch(`${API_BASE}/ann`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Backend tidak merespons');

        const result = await response.json();
        const html = `
          <strong>Hasil ANN:</strong><br/>
          Output Prediksi: <strong>${result.output}</strong><br/>
          <small>${result.explanation}</small>
        `;
        showResult('annResult', html);
      } catch (error) {
        showError('annResult', error.message + ' — Pastikan backend berjalan di port 5000');
      }
    });
  }

  const gaForm = document.getElementById('gaForm');
  if (gaForm) {
    gaForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      showLoading('gaResult');

      const formData = new FormData(gaForm);
      const data = {
        target: formData.get('target'),
        generations: parseInt(formData.get('generations'))
      };

      try {
        const response = await fetch(`${API_BASE}/genetic`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Backend tidak merespons');

        const result = await response.json();
        let html = `<strong>Hasil Algoritma Genetika:</strong><br/>`;
        if (result.success) {
          html += `Target "<strong>${result.target}</strong>" ditemukan dalam <strong>${result.generations}</strong> generasi!<br/>`;
        } else {
          html += `Terbaik: "<strong>${result.best_individual}</strong>" (fitness ${result.fitness}/${result.target.length})<br/>`;
        }
        html += `<small>${result.explanation}</small>`;
        showResult('gaResult', html);
      } catch (error) {
        showError('gaResult', error.message + ' — Pastikan backend berjalan di port 5000');
      }
    });
  }

  const ga2Form = document.getElementById('ga2Form');
  if (ga2Form) {
    ga2Form.addEventListener('submit', async function (e) {
      e.preventDefault();
      showLoading('ga2Result');

      const formData = new FormData(ga2Form);
      const data = {
        pop_size: parseInt(formData.get('pop_size')),
        generations: parseInt(formData.get('generations')),
        crossover_rate: parseFloat(formData.get('crossover_rate')),
        mutation_rate: parseFloat(formData.get('mutation_rate')),
        capacity: parseInt(formData.get('capacity')),
        elitism: formData.get('elitism') === 'on' || formData.get('elitism') === 'true'
      };

      try {
        const response = await fetch(`${API_BASE}/genetic2`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Backend tidak merespons');

        const result = await response.json();
        if (result.error) throw new Error(result.error);

        let html = `<strong>Hasil Knapsack GA:</strong><br/>`;
        if (result.success) {
          html += `Nilai terbaik: <strong>${result.total_value}</strong> (berat ${result.total_weight}/${result.capacity})<br/>`;
          html += `Item terpilih: <strong>${result.chosen_items.join(', ') || '-'}</strong><br/>`;
        } else {
          html += `Tidak berhasil menemukan solusi (coba ubah parameter).`;
        }

        html += `<details><summary>Ringkasan Generasi (terakhir)</summary><pre>${JSON.stringify(result.history, null, 2)}</pre></details>`;
        showResult('ga2Result', html);
      } catch (error) {
        showError('ga2Result', error.message + ' — Pastikan backend berjalan di port 5000');
      }
    });
  }

  const tspRunForm = document.getElementById('tspRunForm');
  if (tspRunForm) {
    tspRunForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      showLoading('tspResult');

      const fd = new FormData(tspRunForm);
      const data = {
        pop_size: parseInt(fd.get('pop_size')),
        generations: parseInt(fd.get('generations')),
        pc: parseFloat(fd.get('pc')),
        pm: parseFloat(fd.get('pm')),
        elite_size: parseInt(fd.get('elite_size'))
      };

      try {
        const resp = await fetch(`${API_BASE}/tsp`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });
        if (!resp.ok) throw new Error('Backend tidak merespons');
        const result = await resp.json();
        if (result.error) throw new Error(result.error);

        let html = `<strong>Hasil TSP GA:</strong><br/>`;
        if (result.success) {
          html += `Rute terbaik: <strong>${result.best_route.join(' → ')}</strong><br/>`;
          html += `Jarak: <strong>${result.best_distance}</strong><br/>`;
          html += `<details><summary>History (terakhir)</summary><pre>${JSON.stringify(result.history, null, 2)}</pre></details>`;
        } else {
          html += `Tidak ada hasil.`;
        }

        showResult('tspResult', html);
      } catch (err) {
        showError('tspResult', err.message + ' — Pastikan backend berjalan di port 5000');
      }
    });
  }
});