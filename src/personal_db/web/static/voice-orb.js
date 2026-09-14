/* Voice orb — the call's state, rendered as one object.
 *
 * Two things make an orb worth more than a spinner, and both come from how
 * voice UIs actually fail:
 *
 *   1. It is driven by real audio amplitude, not a fixed animation. When it
 *      moves with your voice you know the mic is live and reaching the server;
 *      when it is still while you talk, something is broken. A looping
 *      animation would look identical either way.
 *   2. Listening, thinking and speaking are visually distinct, because the
 *      slow parts of this pipeline are invisible — a local model can take ten
 *      seconds to answer, and unexplained silence reads as a hang.
 *
 * Amplitude comes from whichever stream is relevant: the microphone while the
 * caller speaks, the bot's own audio while it replies.
 */
(function () {
  // Drawn from the app's own palette (warm dark: gold, steel blue, sage) rather
  // than generic neon, and following its existing phase colours — retrieval is
  // steel blue and summarising is gold elsewhere in the UI, so they are here too.
  const STATE_COLORS = {
    idle:         ['#6e665a', '#4a443b'],   // muted-2
    connected:    ['#95b585', '#5f7a53'],   // ok / sage
    listening:    ['#95b585', '#5f7a53'],
    hearing:      ['#b4d0a4', '#6f8d61'],   // brighter sage: your voice
    transcribing: ['#9a9081', '#6e665a'],   // muted: a brief mechanical step
    thinking:     ['#8fb6c7', '#527489'],   // link / phase-retrieve
    searching:    ['#d2a86b', '#8f6f40'],   // accent / phase-summarize
    speaking:     ['#e4cfa6', '#a68a5c'],   // warm light: the assistant talking
    error:        ['#d97a7a', '#8f4444'],   // err
  };

  class VoiceOrb {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.state = 'idle';
      this.level = 0;        // smoothed amplitude 0..1
      this.phase = 0;
      this.audioCtx = null;
      this.analysers = [];   // {node, buf, active()}
      this.raf = null;
      this.muted = false;
    }

    _ensureCtx() {
      if (!this.audioCtx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        this.audioCtx = new AC();
      }
      // Browsers start the context suspended until a user gesture; the call
      // button is that gesture, so this resolves immediately in practice.
      if (this.audioCtx.state === 'suspended') this.audioCtx.resume().catch(() => {});
      return this.audioCtx;
    }

    /** Watch a MediaStream. `when` decides if this source counts right now. */
    addStream(stream, when) {
      try {
        const ctx = this._ensureCtx();
        const src = ctx.createMediaStreamSource(stream);
        const node = ctx.createAnalyser();
        node.fftSize = 512;
        node.smoothingTimeConstant = 0.6;
        src.connect(node);            // analyser only — playback stays with <audio>
        this.analysers.push({ node, buf: new Uint8Array(node.frequencyBinCount), when });
      } catch (e) {
        // An orb that cannot read audio still shows state; never break the call.
        console.warn('voice orb: audio analysis unavailable', e);
      }
    }

    setState(s) { this.state = s || 'idle'; }
    setMuted(m) { this.muted = !!m; }

    _amplitude() {
      let peak = 0;
      for (const a of this.analysers) {
        if (a.when && !a.when(this.state)) continue;
        a.node.getByteFrequencyData(a.buf);
        let sum = 0;
        for (let i = 0; i < a.buf.length; i++) sum += a.buf[i];
        peak = Math.max(peak, (sum / a.buf.length) / 255);
      }
      return Math.min(1, peak * 2.2);
    }

    start() { if (!this.raf) this._loop(); }
    stop() {
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = null;
      this.analysers = [];
      if (this.audioCtx) { this.audioCtx.close().catch(() => {}); this.audioCtx = null; }
    }

    _loop() {
      const c = this.canvas, ctx = this.ctx;
      const rect = c.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      if (c.width !== rect.width * dpr) {
        c.width = rect.width * dpr; c.height = rect.height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const w = rect.width, h = rect.height, cx = w / 2, cy = h / 2;
      ctx.clearRect(0, 0, w, h);

      const target = this.muted ? 0 : this._amplitude();
      this.level += (target - this.level) * 0.25;      // smooth, so it breathes
      this.phase += 0.045;

      const [c1, c2] = STATE_COLORS[this.state] || STATE_COLORS.idle;
      const base = Math.min(w, h) * 0.26;
      // Busy states breathe on their own; listening states move with the voice.
      const busy = (this.state === 'thinking' || this.state === 'searching');
      const swell = busy ? (Math.sin(this.phase) * 0.5 + 0.5) * 0.22
                         : this.level * 0.55;
      const r = base * (1 + swell);

      // Outer halo tracks amplitude — the part you notice from across the room.
      const halo = ctx.createRadialGradient(cx, cy, r * 0.4, cx, cy, r * 2.1);
      halo.addColorStop(0, c1 + '55');
      halo.addColorStop(1, 'transparent');
      ctx.fillStyle = halo;
      ctx.beginPath(); ctx.arc(cx, cy, r * 2.1, 0, Math.PI * 2); ctx.fill();

      const body = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.35, r * 0.15, cx, cy, r);
      body.addColorStop(0, c1);
      body.addColorStop(1, c2);
      ctx.fillStyle = body;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();

      // Muted is stated outright, not implied by stillness: a silent orb and a
      // deaf one must not look the same.
      if (this.muted) {
        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx - r * 0.6, cy + r * 0.6);
        ctx.lineTo(cx + r * 0.6, cy - r * 0.6);
        ctx.stroke();
      }

      this.raf = requestAnimationFrame(() => this._loop());
    }
  }

  window.VoiceOrb = VoiceOrb;
})();
