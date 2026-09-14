/* Force-directed knowledge-graph view.
 *
 * Hand-rolled rather than d3-from-a-CDN on purpose: this app is offline-first
 * and makes a point of never calling out, so a runtime CDN fetch would both
 * break air-gapped use and leak a request. The simulation below is the standard
 * three forces — repulsion, spring, centering — which is all a graph this size
 * needs.
 *
 * Encoding follows the usual node-link conventions: node size is degree
 * (how connected), colour is entity type, and edges are plain lines. No
 * arrowheads — direction adds clutter at this density and the predicate is
 * available on hover, where it can be read.
 */
(function () {
  const canvas = document.getElementById('kgCanvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const tip = document.getElementById('kgTip');
  const statusEl = document.getElementById('kgStatus');
  const searchEl = document.getElementById('kgFilter');

  // Named colours for the types this extractor emits most; anything else gets a
  // stable hashed hue rather than a shared grey, so an unforeseen type is still
  // distinguishable instead of silently merging with every other unknown.
  const TYPE_COLORS = {
    person:       '#f472b6',
    project:      '#8ab4f8',
    tool:         '#34d399',
    technology:   '#2dd4bf',
    concept:      '#c084fc',
    field:        '#a78bfa',
    decision:     '#fbbf24',
    event:        '#f59e0b',
    organization: '#fb923c',
    org:          '#fb923c',
    product:      '#38bdf8',
    role:         '#e879f9',
    place:        '#22d3ee',
  };
  function hashHue(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
    return h;
  }
  const colorFor = (t) => {
    const k = (t || 'concept').toLowerCase();
    return TYPE_COLORS[k] || `hsl(${hashHue(k)} 55% 62%)`;
  };

  let nodes = [], edges = [], adjacency = new Map();
  let view = { x: 0, y: 0, k: 1 };
  let hover = null, dragging = null, panning = null;
  let filter = '';
  let hideIsolated = false;
  let raf = null, alpha = 1;

  function size() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return rect;
  }

  // ── simulation ───────────────────────────────────────────────────────────
  function step(rect) {
    const cx = rect.width / 2, cy = rect.height / 2;
    const k = alpha;

    // Repulsion — O(n²), fine at the ~120 nodes this view caps at.
    for (let i = 0; i < nodes.length; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const b = nodes[j];
        let dx = b.x - a.x, dy = b.y - a.y;
        let d2 = dx * dx + dy * dy || 0.01;
        if (d2 > 90000) continue;                 // ignore distant pairs
        const f = (2600 * k) / d2;
        const d = Math.sqrt(d2);
        const fx = (dx / d) * f, fy = (dy / d) * f;
        a.vx -= fx; a.vy -= fy; b.vx += fx; b.vy += fy;
      }
    }
    // Springs along edges
    for (const e of edges) {
      const a = e.s, b = e.t;
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 0.01;
      const f = ((d - 90) * 0.02 * k);
      const fx = (dx / d) * f, fy = (dy / d) * f;
      a.vx += fx; a.vy += fy; b.vx -= fx; b.vy -= fy;
    }
    // Gentle pull to centre so the graph doesn't drift off-screen
    for (const n of nodes) {
      n.vx += (cx - n.x) * 0.0015 * k;
      n.vy += (cy - n.y) * 0.0015 * k;
      if (n === dragging) continue;
      n.x += (n.vx *= 0.82);
      n.y += (n.vy *= 0.82);
    }
    alpha = Math.max(0.02, alpha * 0.995);
  }

  function radius(n) { return 4 + Math.min(14, Math.sqrt(n.degree || 0) * 3.2); }

  function matches(n) {
    return !filter || n.label.toLowerCase().includes(filter);
  }
  // Entities with no extracted relation carry no shape; they are worth showing
  // by default (their absence from the graph is itself information) but must be
  // dismissible when they crowd out the connected core.
  function visible(n) { return !hideIsolated || (n.degree || 0) > 0; }

  // ── render ───────────────────────────────────────────────────────────────
  function draw(rect) {
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.save();
    ctx.translate(view.x, view.y);
    ctx.scale(view.k, view.k);

    const neighbours = hover ? (adjacency.get(hover.id) || new Set()) : null;

    for (const e of edges) {
      if (!e.s || !e.t) continue;
      const lit = hover && (e.s === hover || e.t === hover);
      const dim = filter && !(matches(e.s) || matches(e.t));
      // --accent when highlighted, --muted-2 otherwise (canvas can't read vars).
      ctx.strokeStyle = lit ? 'rgba(210,168,107,0.85)'
                            : dim ? 'rgba(110,102,90,0.10)' : 'rgba(110,102,90,0.30)';
      ctx.lineWidth = lit ? 1.6 : 0.8;
      ctx.beginPath();
      ctx.moveTo(e.s.x, e.s.y);
      ctx.lineTo(e.t.x, e.t.y);
      ctx.stroke();
    }

    for (const n of nodes) {
      if (!visible(n)) continue;
      const r = radius(n);
      const isHover = n === hover;
      const near = neighbours && neighbours.has(n.id);
      const dim = (filter && !matches(n)) || (hover && !isHover && !near);
      ctx.globalAlpha = dim ? 0.22 : 1;
      ctx.fillStyle = colorFor(n.type);
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fill();
      if (isHover) {
        ctx.strokeStyle = '#f6f0e1'; ctx.lineWidth = 1.5; ctx.stroke();  // --fg-strong
      }
      // Label only the nodes worth reading at this zoom, so the view stays legible.
      if (!dim && (isHover || r > 8 || filter)) {
        ctx.globalAlpha = dim ? 0.2 : 0.85;
        ctx.fillStyle = '#ece5d5';   // --fg
        ctx.font = `${isHover ? 600 : 400} ${11 / view.k + 1}px Inter, system-ui, sans-serif`;
        ctx.fillText(n.label, n.x + r + 4, n.y + 4);
      }
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  function loop() {
    const rect = canvas.getBoundingClientRect();
    step(rect);
    draw(rect);
    raf = requestAnimationFrame(loop);
  }

  // ── interaction ──────────────────────────────────────────────────────────
  function toGraph(px, py) {
    return { x: (px - view.x) / view.k, y: (py - view.y) / view.k };
  }
  function pick(px, py) {
    const p = toGraph(px, py);
    let best = null, bestD = Infinity;
    for (const n of nodes) {
      if (!visible(n)) continue;
      const d = (n.x - p.x) ** 2 + (n.y - p.y) ** 2;
      const r = radius(n) + 6;
      if (d < r * r && d < bestD) { best = n; bestD = d; }
    }
    return best;
  }

  canvas.addEventListener('mousemove', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    if (dragging) {
      const p = toGraph(px, py);
      dragging.x = p.x; dragging.y = p.y; alpha = Math.max(alpha, 0.35);
      return;
    }
    if (panning) {
      view.x += px - panning.x; view.y += py - panning.y;
      panning = { x: px, y: py };
      return;
    }
    hover = pick(px, py);
    canvas.style.cursor = hover ? 'pointer' : 'grab';
    if (hover) {
      const rels = (relationsFor.get(hover.id) || []).slice(0, 4);
      tip.innerHTML =
        `<strong>${esc(hover.label)}</strong><span class="kg-tip-type">${esc(hover.type)}</span>` +
        `<div class="kg-tip-meta">${hover.degree} connection${hover.degree === 1 ? '' : 's'} · ${hover.mentions} mention${hover.mentions === 1 ? '' : 's'}</div>` +
        (rels.length ? `<ul class="kg-tip-rels">${rels.map(r => `<li>${esc(r)}</li>`).join('')}</ul>` : '');
      tip.style.left = (px + 14) + 'px';
      tip.style.top = (py + 14) + 'px';
      tip.hidden = false;
    } else {
      tip.hidden = true;
    }
  });

  canvas.addEventListener('mousedown', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    const n = pick(px, py);
    if (n) { dragging = n; canvas.style.cursor = 'grabbing'; }
    else { panning = { x: px, y: py }; canvas.style.cursor = 'grabbing'; }
  });
  window.addEventListener('mouseup', () => { dragging = null; panning = null; canvas.style.cursor = 'grab'; });

  canvas.addEventListener('dblclick', () => {
    if (hover) window.location.href = '/graph?e=' + encodeURIComponent(hover.label);
  });

  canvas.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
    const before = toGraph(px, py);
    view.k = Math.min(3, Math.max(0.25, view.k * (ev.deltaY < 0 ? 1.12 : 0.89)));
    const after = toGraph(px, py);
    view.x += (after.x - before.x) * view.k;
    view.y += (after.y - before.y) * view.k;
  }, { passive: false });

  if (searchEl) {
    searchEl.addEventListener('input', () => {
      filter = searchEl.value.trim().toLowerCase();
      alpha = Math.max(alpha, 0.2);
    });
  }
  const isoEl = document.getElementById('kgHideIsolated');
  if (isoEl) {
    isoEl.addEventListener('change', () => {
      hideIsolated = isoEl.checked;
      alpha = Math.max(alpha, 0.35);
    });
  }

  const relationsFor = new Map();
  const esc = (s) => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  // ── load ─────────────────────────────────────────────────────────────────
  fetch('/api/graph/data?limit=120')
    .then(r => r.json())
    .then(data => {
      const rect = size();
      const byId = new Map();
      nodes = (data.nodes || []).map((n, i) => {
        const angle = (i / Math.max(1, data.nodes.length)) * Math.PI * 2;
        const o = Object.assign({}, n, {
          x: rect.width / 2 + Math.cos(angle) * 180 + (Math.random() - 0.5) * 40,
          y: rect.height / 2 + Math.sin(angle) * 180 + (Math.random() - 0.5) * 40,
          vx: 0, vy: 0,
        });
        byId.set(o.id, o);
        return o;
      });
      edges = (data.edges || []).map(e => ({ s: byId.get(e.source), t: byId.get(e.target), predicate: e.predicate }));
      adjacency = new Map(nodes.map(n => [n.id, new Set()]));
      for (const e of edges) {
        if (!e.s || !e.t) continue;
        adjacency.get(e.s.id).add(e.t.id);
        adjacency.get(e.t.id).add(e.s.id);
        const a = relationsFor.get(e.s.id) || []; a.push(`${e.s.label} ${e.predicate} ${e.t.label}`); relationsFor.set(e.s.id, a);
        const b = relationsFor.get(e.t.id) || []; b.push(`${e.s.label} ${e.predicate} ${e.t.label}`); relationsFor.set(e.t.id, b);
      }
      if (statusEl) {
        const iso = nodes.filter(n => !n.degree).length;
        statusEl.textContent = nodes.length
          ? `${nodes.length} entities · ${edges.length} relations`
            + (iso ? ` · ${iso} with no extracted relation` : '')
            + ' — drag to move, scroll to zoom, double-click to open'
          : 'No entities yet. Ingest documents, then build the graph.';
      }
      // Legend reflects only the types actually present.
      const present = [...new Set(nodes.map(n => (n.type || 'concept').toLowerCase()))].sort();
      const legend = document.getElementById('kgLegend');
      if (legend) {
        legend.innerHTML = present.map(t =>
          `<span class="kg-legend-item"><i style="background:${colorFor(t)}"></i>${esc(t)}</span>`).join('');
      }
      window.addEventListener('resize', () => { size(); });
      loop();
    })
    .catch(err => { if (statusEl) statusEl.textContent = 'Could not load the graph: ' + err.message; });
})();
