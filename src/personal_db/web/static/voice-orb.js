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
      this.wave = null;
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

    /** Draw a bar waveform to a second canvas from the same analysers.
     *
     * Shares this instance's audio graph and rAF loop on purpose: a second
     * AudioContext would be a second device-level capture, and browsers cap
     * how many a page may hold. */
    attachWave(canvas) {
      this.wave = canvas ? { canvas, ctx: canvas.getContext('2d') } : null;
    }

    /** Frequency bins of whichever source is live, or null. */
    _spectrum() {
      for (const a of this.analysers) {
        if (a.when && !a.when(this.state)) continue;
        a.node.getByteFrequencyData(a.buf);
        return a.buf;
      }
      return null;
    }

    _drawWave() {
      const w = this.wave;
      if (!w) return;
      const rect = w.canvas.getBoundingClientRect();
      if (!rect.width) return;
      const dpr = window.devicePixelRatio || 1;
      if (w.canvas.width !== rect.width * dpr) {
        w.canvas.width = rect.width * dpr;
        w.canvas.height = rect.height * dpr;
      }
      const ctx = w.ctx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const bins = this.muted ? null : this._spectrum();
      const [c1] = STATE_COLORS[this.state] || STATE_COLORS.idle;
      // Same rule as the orb: synthesise only where there is genuinely no
      // signal to show, never where silence is the information.
      let energy = 0;
      if (bins) {
        for (let i = 0; i < bins.length; i++) energy += bins[i];
        energy /= bins.length;                    // mean bin, 0..255
      }
      const working = this.state === 'thinking' || this.state === 'searching';
      const talkingSilently = this.state === 'speaking' && energy < 2;
      const synth = !this.muted && (working || talkingSilently);
      const BAR = 3, GAP = 3, step = BAR + GAP;
      const n = Math.max(1, Math.floor(rect.width / step));
      const mid = rect.height / 2;
      ctx.fillStyle = c1;

      for (let i = 0; i < n; i++) {
        let v;
        if (synth) {
          // A travelling ripple, tapered from the centre so it shares the
          // mirrored shape of real speech rather than looking like a
          // different instrument.
          const d = Math.abs(i - (n - 1) / 2) / Math.max(1, (n - 1) / 2);
          v = (0.20 + 0.34 * Math.abs(Math.sin(this.phase * 0.9 + i * 0.26)))
              * (1 - d * 0.72);
        } else if (bins) {
          // Mirrored around the centre. Mapping bar index straight onto bins
          // put all the speech energy (which lives in the low bins) at the
          // left and left the rest flat — it read as a broken meter rather
          // than a voice. Distance from centre selects the bin instead, so
          // loud speech blooms outward from the middle.
          const d = Math.abs(i - (n - 1) / 2) / Math.max(1, (n - 1) / 2);
          const lo = Math.floor(d * (bins.length * 0.42));
          v = bins[Math.min(lo, bins.length - 1)] / 255;
        } else {
          v = 0.04;
        }
        // A floor keeps the line visible in silence, so "idle" and "broken"
        // still look different.
        const h = Math.max(2, v * rect.height * 0.92);
        ctx.globalAlpha = (bins || synth) ? 0.35 + v * 0.65 : 0.25;
        ctx.fillRect(i * step, mid - h / 2, BAR, h);
      }
      ctx.globalAlpha = 1;
    }

    _amplitude() {
      // Zero from a suspended context is indistinguishable from zero from a
      // silent room, and it is the more likely cause, so retry rather than
      // trusting the single resume() at construction.
      if (this.audioCtx && this.audioCtx.state === 'suspended') {
        this.audioCtx.resume().catch(() => {});
      }
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
      this.wave = null;
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
      // "Working" and "talking" are states the caller cannot verify by ear the
      // way they can verify their own voice, and both were rendering as a dead
      // circle: during a tool call the mic analyser is deliberately inactive,
      // and a remote WebRTC stream does not always expose levels to Web Audio.
      // Motion is synthesised for those two, and never for listening/hearing —
      // there, stillness is the true signal that the mic is not reaching us.
      const talking = this.state === 'speaking';
      const busy = (this.state === 'thinking' || this.state === 'searching')
                 || (talking && this.level < 0.02);
      // A slow baseline breath always runs. Without it a quiet room renders an
      // absolutely static circle, which reads as "the call has hung" — the one
      // thing this orb exists to rule out.
      const breath = (Math.sin(this.phase * 0.55) * 0.5 + 0.5) * 0.06;
      const swell = busy ? (Math.sin(this.phase) * 0.5 + 0.5) * 0.22
                         : breath + this.level * 0.55;
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

      // The waveform is decoration; the orb is the status light. A throw in
      // the former must never stop the latter, because the rAF chain is only
      // continued below — one exception here would freeze the orb for good and
      // look exactly like a dead call.
      try { this._drawWave(); } catch (e) { this.wave = null; console.warn('voice wave off', e); }

      this.raf = requestAnimationFrame(() => this._loop());
    }
  }

  window.VoiceOrb = VoiceOrb;
})();
