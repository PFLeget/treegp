/* One init function per slide; lazily initialized on first visit. */
"use strict";
const DEMOS = (() => {
  const V = window.VISIT || null;
  const inited = {};
  const $ = (id) => document.getElementById(id);
  const fmt = (x, d = 2) => Number(x).toFixed(d);
  const e0 = (arr) => arr.slice(0, 4).reduce((a, b) => a + b, 0) / 4; // E-mode at the smallest separations (mean of the first 4 bins)
  const BASE = { margin: { l: 50, r: 10, t: 30, b: 40 }, font: { size: 12 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#fff" };
  const CFG = { displayModeBar: false, responsive: true };
  // matplotlib "seismic": diverging, white at zero, readable with colour-vision deficiencies
  const SEISMIC = [[0, "rgb(0,0,77)"], [0.25, "rgb(0,0,255)"], [0.5, "rgb(255,255,255)"], [0.75, "rgb(255,0,0)"], [1, "rgb(128,0,0)"]];
  const BLUE = "#1f5fbf", ORANGE = "#e67e22", GREY = "#7f7f7f", BLACK = "#222";
  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

  function heat(div, z, x, y, opts = {}) {
    const zmax = opts.zmax !== undefined ? opts.zmax : NUM.absmax(z.flat());
    const data = [{ type: "heatmap", z, x, y, colorscale: opts.colorscale || SEISMIC, zmin: opts.zmin !== undefined ? opts.zmin : -zmax, zmax, showscale: opts.showscale !== false, colorbar: { thickness: 10, len: 0.9, title: { text: opts.ctitle || "", side: "right" } }, hovertemplate: opts.hover || "%{x:.3f}, %{y:.3f}<br>%{z:.2f}<extra></extra>" }];
    const layout = Object.assign({}, BASE, { title: { text: opts.title || "", font: { size: opts.tfont || 13 } }, xaxis: { title: opts.xt !== undefined ? opts.xt : "u (deg)", scaleanchor: opts.square === false ? undefined : "y", constrain: "domain", automargin: true }, yaxis: { title: opts.yt !== undefined ? opts.yt : "v (deg)", constrain: "domain", automargin: true } }, opts.layout || {});
    if (opts.traces) data.push(...opts.traces);
    Plotly.react(div, data, layout, CFG);
  }
  const bind = (ids, fn) => ids.forEach((id) => { const el = $(id); if (el) el.addEventListener("input", fn); el && el.addEventListener("click", fn); });
  const toggle = (ids, active) => ids.forEach((id) => $(id).classList.toggle("active", id === active));
  const starsTrace = (xs, ys, size = 3) => ({ x: xs, y: ys, mode: "markers", marker: { size, color: BLACK }, hoverinfo: "skip", showlegend: false });
  // pointer position in data coordinates (works anywhere in the plot area, not only on a trace point)
  const pointerXY = (gd, ev) => {
    const fl = gd._fullLayout; if (!fl || !fl.xaxis) return null;
    const bb = gd.getBoundingClientRect();
    return { x: fl.xaxis.p2d(ev.clientX - bb.left - fl.xaxis._offset), y: fl.yaxis.p2d(ev.clientY - bb.top - fl.yaxis._offset) };
  };
  const ebTraces = (eb, which) => {
    const t = [
      { x: eb.r, y: eb.E_before, name: "E-mode, before", mode: "lines+markers", line: { color: BLUE } },
      { x: eb.r, y: eb.B_before, name: "B-mode, before", mode: "lines+markers", line: { color: ORANGE } },
    ];
    if (which !== "before") {
      t.push({ x: eb.r, y: eb.E_after, name: "E-mode, TURBO-GP corrected", mode: "lines+markers", line: { color: BLUE, dash: "dot" } });
      t.push({ x: eb.r, y: eb.B_after, name: "B-mode, TURBO-GP corrected", mode: "lines+markers", line: { color: ORANGE, dash: "dot" } });
      if (eb.E_after_l21) t.push({ x: eb.r, y: eb.E_after_l21, name: "E-mode, L21 corrected", mode: "lines", line: { color: GREY, dash: "dash" } });
    }
    return t;
  };
  const ebLayout = (title) => Object.assign({}, BASE, { title: { text: title, font: { size: 13 } }, xaxis: { type: "log", title: "separation (deg)" }, yaxis: { title: "ξ (mas²)" }, legend: { x: 0.55, y: 0.95, font: { size: 10 } } });
  // real patch helpers (exported by make_data.py): the full box and its central display window
  const patchGrid = (p) => ({ x0: p.x0, y0: p.y0, h: p.h, nx: p.nx, ny: p.ny });
  const patchAxes = (p) => ({ x: NUM.linspace(p.x0, p.x0 + (p.nx - 1) * p.h, p.nx), y: NUM.linspace(p.y0, p.y0 + (p.ny - 1) * p.h, p.ny) });
  const windowOf = (img, p) => { // central win x win nodes of a (ny, nx) image, as 2d array + axes
    const z = []; for (let y = p.off; y < p.off + p.win; y++) { const row = []; for (let x = p.off; x < p.off + p.win; x++) row.push(img[y * p.nx + x]); z.push(row); }
    return { z, x: NUM.linspace(p.x0 + p.off * p.h, p.x0 + (p.off + p.win - 1) * p.h, p.win), y: NUM.linspace(p.y0 + p.off * p.h, p.y0 + (p.off + p.win - 1) * p.h, p.win) };
  };
  const inWindow = (p, S) => { const x0 = p.x0 + p.off * p.h, x1 = x0 + (p.win - 1) * p.h, y0 = p.y0 + p.off * p.h, y1 = y0 + (p.win - 1) * p.h; const xs = [], ys = []; S.xs.forEach((u, i) => { if (u >= x0 && u <= x1 && S.ys[i] >= y0 && S.ys[i] <= y1) { xs.push(u); ys.push(S.ys[i]); } }); return { xs, ys }; };
  const fullGrid = (c) => ({ x0: c.grid.x0, y0: c.grid.y0, h: c.grid.h, nx: c.grid.nx, ny: c.grid.ny }); // the solver grid of the whole visit
  // the N training stars: positions shared, centred values / errors / solved weights per component
  const trainOf = (comp) => ({ xs: V.train.u, ys: V.train.v, y: V.comp[comp].train.y, sig: V.comp[comp].train.sig, alpha: V.comp[comp].train.alpha });
  const starsIn = (comp, x0, x1, y0, y1) => { // those inside [x0, x1) x [y0, y1)
    const t = trainOf(comp), xs = [], ys = [], y = [], sig = [], alpha = [];
    t.xs.forEach((u, i) => { const v = t.ys[i]; if (u >= x0 && u < x1 && v >= y0 && v < y1) { xs.push(u); ys.push(v); y.push(t.y[i]); sig.push(t.sig[i]); alpha.push(t.alpha[i]); } });
    return { xs, ys, y, sig, alpha };
  };
  const patchStars = (comp, p) => starsIn(comp, p.x0, p.x0 + (p.nx - 1) * p.h, p.y0, p.y0 + (p.ny - 1) * p.h);
  const convolvers = {};
  const realConvolver = (comp, pad) => { // tabulated real kernel image on the patch grid (cached: one 1024x1024 FFT)
    const key = comp + (pad ? "p" : "n");
    if (!convolvers[key]) { const c = V.comp[comp]; convolvers[key] = NUM.makeConvolverFromImage(patchGrid(c.patch), Float64Array.from(c.kimg.flat()), c.grid.kernel_side, pad); }
    return convolvers[key];
  };
  const fullConvolver = (comp) => { // same, on the solver grid of the whole visit (2048x2048 padded FFT, ~0.4 s once)
    const key = comp + "full";
    if (!convolvers[key]) { const c = V.comp[comp]; convolvers[key] = NUM.makeConvolverFromImage(fullGrid(c), Float64Array.from(c.kimg.flat()), c.grid.kernel_side, true); }
    return convolvers[key];
  };
  const blockAxes = (g, r, ds) => ({ x: NUM.linspace(g.x0 + (ds - 1) / 2 * g.h, g.x0 + ((r.nx - 1) * ds + (ds - 1) / 2) * g.h, r.nx), y: NUM.linspace(g.y0 + (ds - 1) / 2 * g.h, g.y0 + ((r.ny - 1) * ds + (ds - 1) / 2) * g.h, r.ny) }); // block centres

  // =============================================================== 1. the field
  function field() {
    let comp = "u", show = "raw";
    const draw = () => {
      const d = V.disp;
      const base = comp === "u" ? d.du : d.dv, gp = comp === "u" ? d.gpu : d.gpv;
      const vals = show === "raw" ? base : show === "gp" ? gp : base.map((b, i) => b - gp[i]);
      const zmax = show === "res" ? 40 : 60;
      Plotly.react("field-scatter", [{ type: "scattergl", mode: "markers", x: d.u, y: d.v, marker: { size: 4, color: vals, colorscale: SEISMIC, cmin: -zmax, cmax: zmax, colorbar: { thickness: 10, title: { text: `d${comp} (mas)`, side: "right" } } }, hovertemplate: "%{x:.3f}, %{y:.3f}<br>%{marker.color:.1f} mas<extra></extra>" }],
        Object.assign({}, BASE, { title: { text: `visit ${V.meta.visit}: ${show === "raw" ? "residual" : show === "gp" ? "GP prediction" : "residual after correction"} d${comp}, ${d.u.length} validation stars`, font: { size: 13 } }, xaxis: { title: "u (deg)", scaleanchor: "y" }, yaxis: { title: "v (deg)" } }), CFG);
      Plotly.react("field-eb", ebTraces(V.eb, show === "raw" ? "before" : "all"), ebLayout("E/B-mode correlation functions (validation stars)"), CFG);
      $("field-text").innerHTML = `E-mode variance at the smallest separations: <b>${fmt(e0(V.eb.E_before), 0)} mas²</b> before, <b>${fmt(e0(V.eb.E_after), 0)} mas²</b> after TURBO-GP (B-mode ${fmt(e0(V.eb.B_before), 0)} → ${fmt(e0(V.eb.B_after), 0)}${V.eb.E_after_l21 ? `; L21 on the same training stars: E ${fmt(e0(V.eb.E_after_l21), 0)}` : ""}). ${V.meta.n_train} training stars, ${V.meta.n_val} validation stars; the per-star photon noise (~${fmt(V.comp.u.noise_median, 0)} mas) stays in the scatter but not in the correlation function.`;
    };
    bind(["field-u", "field-v"], (e) => { comp = e.target.id.endsWith("u") ? "u" : "v"; toggle(["field-u", "field-v"], e.target.id); draw(); });
    bind(["field-raw", "field-gp", "field-res"], (e) => { show = e.target.id.split("-")[1]; toggle(["field-raw", "field-gp", "field-res"], e.target.id); draw(); });
    draw();
  }

  // =============================================================== 2. GRF and its 2-pcf
  function grf() {
    const n = 128, size = 3.5, h = size / n, XI0 = 400;
    let r = NUM.rng(7);
    const draw = () => {
      const ell = +$("grf-ell").value, g1 = +$("grf-g1").value, g2 = +$("grf-g2").value, noise = +$("grf-noise").value;
      $("grf-ell-v").textContent = fmt(ell); $("grf-g1-v").textContent = fmt(g1); $("grf-g2-v").textContent = fmt(g2); $("grf-noise-v").textContent = noise;
      const kfun = (lx, ly) => XI0 * NUM.vk(NUM.anisoRadius(lx, ly, g1, g2), ell);
      const kimg = NUM.kernelImage(n, h, kfun);
      const rr = NUM.rng(r.seed);
      const fld = NUM.grf(n, kimg, rr);
      const ax = NUM.linspace(0, size, n);
      heat("grf-field", NUM.to2d(fld, n, n), ax, ax, { title: "one realization of the field y(u, v) (mas)", zmax: 50, ctitle: "y (mas)" });
      const lagMax = 0.6, c = Math.round(lagMax / h);
      let z, lags;
      if ($("grf-measure").checked) {
        const ns = 1500, xs = new Float64Array(ns), ys = new Float64Array(ns);
        for (let i = 0; i < ns; i++) { xs[i] = rr.uniform() * size; ys[i] = rr.uniform() * size; }
        const f = NUM.sampleImage(fld, n, h, xs, ys);
        for (let i = 0; i < ns; i++) f[i] += noise * rr.normal();
        const pix = 2 * h, npix = 2 * Math.round(lagMax / pix);
        const pc = NUM.pairCount2pcf(xs, ys, f, npix, pix);
        z = NUM.to2d(pc.map, npix, npix); lags = NUM.linspace(-lagMax, lagMax, npix);
      } else {
        const ac = NUM.autocorr(fld, n), m = n / 2;
        z = []; lags = [];
        for (let y = m - c; y < m + c; y++) { const row = []; for (let x = m - c; x < m + c; x++) row.push(ac[y * n + x]); z.push(row); }
        for (let i = -c; i < c; i++) lags.push(i * h);
        if (noise > 0) z[c][c] += noise * noise; // white noise only at zero lag
      }
      heat("grf-xi", z, lags, lags, { title: "its 2-point correlation function ξ(Δu, Δv) (mas²)", xt: "Δu (deg)", yt: "Δv (deg)", zmax: XI0, ctitle: "ξ (mas²)" });
    };
    r.seed = 7;
    bind(["grf-ell", "grf-g1", "grf-g2", "grf-noise", "grf-measure"], draw);
    $("grf-new").addEventListener("click", () => { r.seed = Math.floor(Math.random() * 1e9); draw(); });
    draw();
  }

  // =============================================================== 3. GP in 1D
  function gp1d() {
    const XI0 = 400, xmin = 0, xmax = 2;
    let r = NUM.rng(3), xs = [], ys = [], truth = null;
    const xg = NUM.linspace(xmin, xmax, 300);
    const makeTruth = () => { // smooth random function, correlation length ~0.2
      const rr = NUM.rng(r.seed);
      const comps = []; for (let k = 0; k < 12; k++) comps.push({ a: rr.normal() * 8, f: 0.5 + k * 0.7, p: rr.uniform() * 6.28 });
      truth = (x) => comps.reduce((s, c) => s + c.a * Math.cos(2 * Math.PI * c.f * x + c.p), 0);
    };
    const newStars = () => {
      const rr = NUM.rng(r.seed + 1), n = +$("gp1d-n").value, sig = +$("gp1d-sig").value;
      xs = []; ys = [];
      for (let i = 0; i < n; i++) { const x = xmin + rr.uniform() * (xmax - xmin); xs.push(x); ys.push(truth(x) + sig * rr.normal()); }
    };
    const draw = () => {
      const ell = +$("gp1d-ell").value, sig = +$("gp1d-sig").value;
      $("gp1d-ell-v").textContent = fmt(ell); $("gp1d-sig-v").textContent = sig; $("gp1d-n-v").textContent = xs.length;
      const kfun = (d) => XI0 * NUM.vk(d, ell);
      const res = NUM.gp1d(xs, ys, xs.map(() => sig), kfun, xg);
      const traces = [
        { x: xg, y: xg.map(truth), name: "true field", mode: "lines", line: { color: GREY, dash: "dot" } },
        { x: xg.concat([...xg].reverse()), y: Array.from(res.mean, (m, i) => m + res.sd[i]).concat(Array.from(res.mean, (m, i) => m - res.sd[i]).reverse()), fill: "toself", fillcolor: "rgba(31,95,191,0.15)", line: { width: 0 }, name: "±1σ (Eq. 5)", hoverinfo: "skip" },
        { x: xg, y: Array.from(res.mean), name: "GP prediction (Eq. 4)", mode: "lines", line: { color: BLUE, width: 3 } },
        { x: xs, y: ys, error_y: { type: "constant", value: sig, width: 2 }, name: "stars (measured)", mode: "markers", marker: { color: ORANGE, size: 8 } },
      ];
      if ($("gp1d-bumps").checked) xs.forEach((xj, j) => traces.push({ x: xg, y: xg.map((x) => res.alpha[j] * kfun(Math.abs(x - xj))), mode: "lines", line: { color: "rgba(60,60,60,0.45)", width: 1 }, showlegend: j === 0, name: "bumps α_j ξ(x − x_j)", hoverinfo: "skip" }));
      Plotly.react("gp1d-main", traces, Object.assign({}, BASE, { title: { text: "one component of the field along a line", font: { size: 13 } }, xaxis: { title: "x (deg)", range: [xmin, xmax] }, yaxis: { title: "displacement (mas)", range: [-60, 60] }, legend: { x: 0.01, y: 0.99, font: { size: 10 }, bgcolor: "rgba(255,255,255,0.7)" } }), CFG);
      Plotly.react("gp1d-alpha", [{ type: "bar", x: xs, y: Array.from(res.alpha), marker: { color: "#555" }, width: 0.012 }], Object.assign({}, BASE, { margin: { l: 50, r: 10, t: 22, b: 30 }, title: { text: "the weights α (Eq. 3), one per star (mas⁻¹)", font: { size: 12 } }, xaxis: { range: [xmin, xmax], title: "x (deg)", automargin: true }, yaxis: { title: "α (mas⁻¹)", automargin: true } }), CFG);
    };
    r.seed = 3; makeTruth(); newStars(); draw();
    bind(["gp1d-ell", "gp1d-sig", "gp1d-bumps"], draw);
    $("gp1d-n").addEventListener("input", () => { newStars(); draw(); });
    $("gp1d-new").addEventListener("click", () => { r.seed = Math.floor(Math.random() * 1e9); makeTruth(); newStars(); draw(); });
    $("gp1d-main").on("plotly_click", (ev) => { const p = ev.points[0]; xs.push(p.x); ys.push(p.y); draw(); });
  }

  // =============================================================== 4. kernel origin
  function kernel() {
    let comp = "u";
    const draw = () => {
      const c = V.comp[comp];
      const z = NUM.absmax(c.raw.flat());
      if (c.vk) {
        const lag = c.vk.lag;
        heat("kernel-meas", c.vk.measured, lag, lag, { title: "measured ξ, L21 binning (mas²)", zmax: z, showscale: false, xt: "Δu (deg)", yt: "Δv (deg)" });
        heat("kernel-fit", c.vk.fit, lag, lag, { title: "L21: von Kármán fit to it (mas²)", zmax: z, showscale: false, xt: "Δu (deg)", yt: "Δv (deg)" });
        heat("kernel-resid", c.vk.measured.map((row, i) => row.map((v, j) => v - c.vk.fit[i][j])), lag, lag, { title: "measured − fit (mas²)", zmax: z / 3, xt: "Δu (deg)", yt: "Δv (deg)", ctitle: "mas²" });
      }
      heat("kernel-raw", c.raw, c.lag, c.lag, { title: "measured ξ, 20\" pixels, raw (mas²)", zmax: z, showscale: false, xt: "Δu (deg)", yt: "Δv (deg)" });
      heat("kernel-clean", c.clean, c.lag, c.lag, { title: "cleaned ξ = the kernel (mas²)", zmax: z, xt: "Δu (deg)", yt: "Δv (deg)", ctitle: "ξ (mas²)" });
      $("kernel-text").innerHTML = `L21: fit \\(\\phi, \\ell, g_1, g_2\\) of a sheared von Kármán profile to the map (plus a constant), then evaluate a Bessel function for every pair of stars. ${c.vk ? `Fit time here: ${fmt(c.vk.t_fit, 1)} s.` : ""}<br>G25 / TURBO-GP: no functional form, no fit, no Bessel function; the measured map, cleaned, is the kernel. ξ(0): raw ${fmt(c.xi0_raw, 0)} → cleaned ${fmt(c.xi0_clean, 0)} mas² (the zero-lag pixel of the raw map is one noisy pixel that also carries the photon-noise term; the cleaned value is set by the retained Fourier modes).`;
      renderMathInElement($("kernel-text"), { delimiters: [{ left: "\\(", right: "\\)", display: false }] });
    };
    bind(["kernel-u", "kernel-v"], (e) => { comp = e.target.id.endsWith("u") ? "u" : "v"; toggle(["kernel-u", "kernel-v"], e.target.id); draw(); });
    draw();
  }

  // =============================================================== 5. cleaning, live
  function clean() {
    let comp = "u", win = "hann";
    const draw = () => {
      const c = V.comp[comp], n = c.npix, pix = V.meta.pixel_deg;
      const rad = +$("clean-rad").value, g1 = +$("clean-g1").value, g2 = +$("clean-g2").value, thr = +$("clean-thr").value;
      $("clean-rad-v").textContent = fmt(rad, 3); $("clean-g1-v").textContent = fmt(g1); $("clean-g2-v").textContent = fmt(g2); $("clean-thr-v").textContent = fmt(thr, 2);
      const raw = Float64Array.from(c.raw.flat());
      const w = NUM.apodWindow(n, rad / pix, win, g1, g2);
      const ap = raw.map((v, i) => v * w[i]);
      const pk = NUM.corr2power(ap, n);
      const th = NUM.thresholdSpectrum(pk, thr);
      const out = NUM.power2corr(th.pk, n);
      const z = NUM.absmax(raw);
      heat("clean-raw", NUM.to2d(ap, n, n), c.lag, c.lag, { title: "raw ξ × window (mas²)", zmax: z, showscale: false, xt: "Δu (deg)", yt: "Δv (deg)" });
      // central quarter of the spectrum (the kept modes live at low frequency), log|P| with the kept modes outlined
      const q = n / 4, kc = [], lz = [], kz = [];
      for (let y = n / 2 - q; y < n / 2 + q; y++) { const r1 = [], r2 = []; for (let x = n / 2 - q; x < n / 2 + q; x++) { r1.push(Math.log10(1 + Math.abs(pk[y * n + x]))); r2.push(th.pk[y * n + x] > 0 ? 1 : 0); } lz.push(r1); kz.push(r2); }
      for (let i = -q; i < q; i++) kc.push(i / (n * pix));
      heat("clean-power", lz, kc, kc, { title: `|P̂_k| (log10), central quarter; kept: ${th.kept} of ${n * n} modes`, xt: "k_u (1/deg)", yt: "k_v (1/deg)", colorscale: "Viridis", zmin: 0, zmax: Math.max(...lz.flat()), showscale: false, traces: [{ type: "contour", z: kz, x: kc, y: kc, contours: { start: 0.5, end: 0.5, size: 1, coloring: "none" }, line: { color: "#fff", width: 2 }, showscale: false, hoverinfo: "skip" }] });
      heat("clean-out", NUM.to2d(out, n, n), c.lag, c.lag, { title: "cleaned ξ, the kernel (mas²)", zmax: z, xt: "Δu (deg)", yt: "Δv (deg)", ctitle: "ξ (mas²)" });
      $("clean-text").innerHTML = `threshold = median + ${thr} × (half the 16–84% spread) = ${fmt(th.thresh, 1)}; ${th.kept} modes kept, all with \\(\\hat P_k > 0\\): the cleaned map is positive semi-definite on its own ${n}×${n} grid, and band-limited (white outline: the kept modes). ξ(0): ${fmt(raw[(n / 2) * n + n / 2], 0)} → ${fmt(out[(n / 2) * n + n / 2], 0)} mas². treegp measured the anisotropy of the first-pass map as g = (${fmt(c.g_meas[0])}, ${fmt(c.g_meas[1])}) and applied g = (${fmt(c.g_app[0])}, ${fmt(c.g_app[1])}).`;
      renderMathInElement($("clean-text"), { delimiters: [{ left: "\\(", right: "\\)", display: false }] });
    };
    bind(["clean-rad", "clean-g1", "clean-g2", "clean-thr"], draw);
    bind(["clean-hann", "clean-bh"], (e) => { win = e.target.id.endsWith("hann") ? "hann" : "bh"; toggle(["clean-hann", "clean-bh"], e.target.id); draw(); });
    bind(["clean-u", "clean-v"], (e) => { comp = e.target.id.endsWith("u") ? "u" : "v"; toggle(["clean-u", "clean-v"], e.target.id); draw(); });
    $("clean-auto").addEventListener("click", () => { $("clean-g1").value = V.comp[comp].g_app[0]; $("clean-g2").value = V.comp[comp].g_app[1]; draw(); });
    draw();
  }

  // =============================================================== 6. why it breaks (toy)
  function psd() {
    const XI0 = 400, ell = 0.1, N = 60, box = 1.0; // ell = half-width, DP2-like
    const rr = NUM.rng(11), px = [], py = [];
    for (let i = 0; i < N; i++) { px.push(rr.uniform() * box); py.push(rr.uniform() * box); }
    const Rs = NUM.linspace(0.1, 1.0, 28);
    const kern = (r, R, taper) => (r >= R ? 0 : XI0 * NUM.vk(r, ell) * (taper ? NUM.hann(r / R) : 1));
    const minEig = (R, taper, sig) => {
      const K = new Float64Array(N * N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) K[i * N + j] = kern(Math.hypot(px[i] - px[j], py[i] - py[j]), R, taper) + (i === j ? sig * sig : 0);
      return NUM.jacobiEigenvalues(K, N)[0];
    };
    const cache = {};
    const draw = () => {
      const R = +$("psd-R").value, taper = $("psd-taper").checked, sig = +$("psd-sig").value;
      $("psd-R-v").textContent = fmt(R); $("psd-sig-v").textContent = fmt(sig, 1);
      const m = 4096, L = 16, dx = L / m, xs = NUM.linspace(-L / 2, L / 2 - dx, m);
      const re = Float64Array.from(xs, (x) => kern(Math.abs(x), R, taper)), im = new Float64Array(m);
      Plotly.react("psd-kernel", [{ x: xs, y: Array.from(re), mode: "lines", name: "ξ(r)", line: { color: BLUE } }], Object.assign({}, BASE, { margin: { l: 50, r: 10, t: 26, b: 34 }, title: { text: `kernel (half-width ${ell}°) truncated at R = ${fmt(R)}°`, font: { size: 12 } }, xaxis: { range: [-1.2, 1.2], title: "r (deg)" }, yaxis: { title: "ξ(r) (mas²)" } }), CFG);
      const sh = NUM.fftshift(re, m, 1);
      NUM.fft1d(sh, im, false);
      const kk = NUM.linspace(0, 1 / (2 * dx), m / 2), spec = Array.from(sh.subarray(0, m / 2), (v) => v * dx);
      Plotly.react("psd-spectrum", [{ x: kk, y: spec, mode: "lines", name: "FT", line: { color: BLUE } }, { x: [0, 8], y: [0, 0], mode: "lines", line: { color: "#000", width: 0.5 }, showlegend: false }], Object.assign({}, BASE, { margin: { l: 50, r: 10, t: 26, b: 34 }, title: { text: "its Fourier transform (negative lobes = not a valid kernel)", font: { size: 12 } }, xaxis: { range: [0, 20], title: "k (1/deg)" }, yaxis: { range: [-0.15 * Math.max(...spec), 0.4 * Math.max(...spec)], title: "P(k) (mas² deg)" } }), CFG);
      const key = `${taper}-${sig}`;
      if (!cache[key]) cache[key] = Rs.map((Rv) => minEig(Rv, taper, sig));
      const cur = minEig(R, taper, sig);
      Plotly.react("psd-eig", [
        { x: Rs, y: cache[key], mode: "lines+markers", line: { color: BLUE }, name: `smallest eigenvalue of K + D (N = ${N} stars in 1°×1°)` },
        { x: [R], y: [cur], mode: "markers", marker: { size: 14, color: ORANGE }, name: `R = ${fmt(R)}°: λ_min = ${fmt(cur, 1)} mas²` },
        { x: [0.1, 1.0], y: [0, 0], mode: "lines", line: { color: "#000", width: 0.5 }, showlegend: false },
      ], Object.assign({}, BASE, { title: { text: "smallest eigenvalue of K + D between the stars vs truncation radius R", font: { size: 12 } }, xaxis: { title: "truncation radius R (deg)" }, yaxis: { title: "λ_min (mas²)" }, legend: { x: 0.3, y: 0.05, font: { size: 10 } } }), CFG);
    };
    bind(["psd-R", "psd-taper", "psd-sig"], draw);
    draw();
  }

  // =============================================================== 6b. why it breaks, real visit
  function psdreal() {
    let comp = "u";
    $("psdr-n").textContent = V.meta.n_eig;
    const draw = () => {
      const c = V.comp[comp], e = c.eig, cut = c.spec_cut;
      const k = cut.map((_, i) => i / (c.grid.lx * c.grid.h));
      const mn = Math.min(...cut);
      Plotly.react("psdr-cut", [
        { x: k, y: cut, mode: "lines+markers", marker: { size: 4 }, line: { color: BLUE }, name: "P̂_k of the padded kernel image" },
        { x: [0, k[k.length - 1]], y: [0, 0], mode: "lines", line: { color: "#000", width: 0.8 }, showlegend: false },
      ], Object.assign({}, BASE, { title: { text: `P̂_k of the d${comp} kernel on the ${c.grid.lx}² grid, along k_v = 0 (peak ${Number(c.spec_peak).toExponential(1)} off scale)`, font: { size: 10 } }, xaxis: { title: "k_u (1/deg)" }, yaxis: { title: "P̂_k (mas²)", range: [1.6 * mn, -4 * mn] }, showlegend: false }), CFG);
      const idx = e.tab.map((_, i) => i + 1);
      Plotly.react("psdr-eig", [
        { x: idx, y: e.tab, mode: "markers", marker: { color: ORANGE, size: 7 }, name: `tabulated kernel K (${e.n_neg_tab} negative)` },
        { x: idx, y: e.tabD, mode: "markers", marker: { color: BLACK, size: 7, symbol: "diamond" }, name: `K + D, with the noise (${e.n_neg_tabD} negative)` },
        { x: idx, y: e.grid, mode: "markers", marker: { color: BLUE, size: 7, symbol: "square" }, name: `TURBO-GP grid kernel W G Wᵀ (${e.n_neg_grid} negative)` },
        { x: [0, idx.length + 1], y: [0, 0], mode: "lines", line: { color: "#000", width: 0.8 }, showlegend: false },
      ], Object.assign({}, BASE, { title: { text: `the ${idx.length} lowest eigenvalues (mas²) of the covariance of ${V.meta.n_eig} real stars`, font: { size: 12 } }, xaxis: { title: "rank (lowest first)" }, yaxis: { title: "λ (mas²)" }, legend: { x: 0.35, y: 0.08, font: { size: 10 } } }), CFG);
      $("psdr-text").innerHTML = `The cleaned map is positive semi-definite on its own 20" grid, but between stars it is <em>interpolated</em> and <em>cut</em> at ±0.3°: zero-padded on the solver grid its spectrum shows the negative lobes of slide 6 (deepest: ${fmt(c.spectrum_min, 0)}, i.e. ${fmt(100 * c.neg_power, 1)}% of the positive power). For the ${V.meta.n_eig} training stars within ${fmt(e.radius, 2)}° of the field centre (the real density), the tabulated kernel gives <b>${e.n_neg_tab}</b> negative eigenvalues, down to <b>${fmt(e.tab[0], 1)} mas²</b>. The measurement noise helps (smallest σ here: ${fmt(e.min_sig, 1)} mas) but ${e.n_neg_tabD > 0 ? `still leaves <b>${e.n_neg_tabD}</b> negative (${fmt(e.tabD[0], 1)} mas²): Cholesky fails, and the dense path has to clip the eigenvalues at O(N³) cost, as G25 do.` : `is enough on this component; it is not on the other, nor in general.`} The TURBO-GP grid kernel \\(\\mathbf{W}\\mathbf{G}\\mathbf{W}^T\\), built on the clipped spectrum, has none: its lowest eigenvalue is ${fmt(e.grid[0], 2)} mas².`;
      renderMathInElement($("psdr-text"), { delimiters: [{ left: "\\(", right: "\\)", display: false }] });
    };
    bind(["psdr-u", "psdr-v"], (e) => { comp = e.target.id.endsWith("u") ? "u" : "v"; toggle(["psdr-u", "psdr-v"], e.target.id); draw(); });
    draw();
  }

  // =============================================================== 7. grid and W (toy)
  function grid() {
    const g = { x0: 0, y0: 0, h: 1, nx: 9, ny: 9 };
    let star = { x: 3.3, y: 4.6 };
    const nodesX = [], nodesY = [];
    for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) { nodesX.push(x); nodesY.push(y); }
    const rr = NUM.rng(5), sx = [], sy = [], sv = [];
    for (let i = 0; i < 40; i++) { sx.push(rr.uniform() * 7.9); sy.push(rr.uniform() * 7.9); sv.push(rr.normal() * 10); }
    const draw = () => {
      const w = NUM.bilinearWeights(star.x, star.y, g);
      Plotly.react("grid-one", [
        { x: nodesX, y: nodesY, mode: "markers", marker: { size: 7, color: "#bbb" }, name: "grid nodes (spacing h)", hoverinfo: "skip" },
        { x: w.map((q) => q.ix), y: w.map((q) => q.iy), mode: "markers+text", marker: { size: w.map((q) => 10 + 40 * q.w), color: BLUE, opacity: 0.6 }, text: w.map((q) => q.w.toFixed(2)), textposition: "top center", name: "the 4 weights", hoverinfo: "skip" },
        { x: [star.x], y: [star.y], mode: "markers", marker: { size: 14, color: ORANGE, symbol: "star" }, name: "a star", hoverinfo: "skip" },
      ], Object.assign({}, BASE, { dragmode: false, title: { text: "one star and its four nodes", font: { size: 13 } }, xaxis: { range: [-0.5, 8.5], title: "x / h", scaleanchor: "y", dtick: 1, fixedrange: true }, yaxis: { range: [-0.5, 8.5], title: "y / h", dtick: 1, fixedrange: true }, legend: { orientation: "h", y: -0.18, font: { size: 10 } } }), CFG);
      $("grid-weights").innerHTML = `star at (${fmt(star.x)}, ${fmt(star.y)}) h → nodes ` + w.map((q) => `(${q.ix}, ${q.iy}): <b>${q.w.toFixed(3)}</b>`).join(", ") + ` ; sum = ${w.reduce((s, q) => s + q.w, 0).toFixed(3)}`;
    };
    const img = NUM.spread(sx, sy, sv, g);
    heat("grid-many", NUM.to2d(img, g.nx, g.ny), nodesX.slice(0, g.nx), nodesY.filter((_, i) => i % g.nx === 0), { title: "40 stars with values v (mas) deposited on the grid: Wᵀv", xt: "x / h", yt: "y / h", zmax: 20, ctitle: "v, Wᵀv (mas)", traces: [{ x: sx, y: sy, mode: "markers", marker: { size: 6, color: sv, colorscale: SEISMIC, cmin: -20, cmax: 20, line: { width: 1, color: "#000" } }, name: "stars", hoverinfo: "skip" }] });
    draw();
    // click or drag anywhere in the plot area (not only on a trace point)
    const gd = $("grid-one"); let dragging = false;
    const move = (ev) => { const p = pointerXY(gd, ev); if (!p) return; star = { x: clamp(p.x, 0.01, 7.99), y: clamp(p.y, 0.01, 7.99) }; draw(); };
    gd.addEventListener("mousedown", (ev) => { dragging = true; move(ev); ev.preventDefault(); }, true);
    gd.addEventListener("mousemove", (ev) => { if (dragging) move(ev); }, true);
    window.addEventListener("mouseup", () => { dragging = false; });
  }

  // =============================================================== 7b. the real grid, whole visit + linked zoom
  function gridreal() {
    const DS = 8, NODE_VIEW = 48, MARK_VIEW = 320, HEAT_CELLS = 160;
    let comp = "u", range = null, F = null; // range: current zoom of the left panel (deg), null = whole visit
    const build = () => {
      const c = V.comp[comp], g = fullGrid(c), t = trainOf(comp);
      const img = NUM.spread(t.xs, t.ys, t.y, g);
      let touched = 0; for (let i = 0; i < img.length; i++) if (img[i] !== 0) touched++;
      F = { c, g, t, img, touched, red: NUM.blockReduce(img, g.nx, g.ny, DS, "sum") };
    };
    const drawFull = () => {
      const { g, t, red } = F, ax = blockAxes(g, red, DS);
      heat("gridr-full", NUM.to2d(red.img, red.nx, red.ny), ax.x, ax.y, {
        title: `Wᵀy (mas), whole ${g.nx}×${g.ny} grid, summed over ${DS}×${DS} nodes (${fmt(DS * g.h * 60, 1)}' blocks)`, zmax: 30, ctitle: "Wᵀy per block (mas)",
        hover: "%{x:.3f}, %{y:.3f}<br>%{z:.1f} mas<extra></extra>", tfont: 11, layout: { dragmode: "zoom", uirevision: comp },
      });
    };
    const drawZoom = () => {
      const { g, img } = F;
      let ix0 = 0, iy0 = 0, nx = g.nx, ny = g.ny;
      if (range) {
        ix0 = clamp(Math.floor((range.x0 - g.x0) / g.h), 0, g.nx - 2); iy0 = clamp(Math.floor((range.y0 - g.y0) / g.h), 0, g.ny - 2);
        nx = clamp(Math.ceil((range.x1 - g.x0) / g.h), ix0 + 1, g.nx - 1) - ix0 + 1; ny = clamp(Math.ceil((range.y1 - g.y0) / g.h), iy0 + 1, g.ny - 1) - iy0 + 1;
      }
      const x0 = g.x0 + ix0 * g.h, y0 = g.y0 + iy0 * g.h, x1 = x0 + (nx - 1) * g.h, y1 = y0 + (ny - 1) * g.h;
      const S = starsIn(comp, x0 - g.h, x1 + g.h, y0 - g.h, y1 + g.h);
      const extent = `${fmt(nx * g.h * 60, 0)}' × ${fmt(ny * g.h * 60, 0)}'`;
      if (nx <= MARK_VIEW && ny <= MARK_VIEW) { // marker view: every node with a deposit (and, when few, the empty nodes too)
        const nodes = nx <= NODE_VIEW && ny <= NODE_VIEW, sc = nodes ? 1 : 0.45;
        const ex = [], ey = [], fx = [], fy = [], fv = [], fs = [];
        for (let iy = iy0; iy < iy0 + ny; iy++) for (let ix = ix0; ix < ix0 + nx; ix++) {
          const v = img[iy * g.nx + ix], x = g.x0 + ix * g.h, y = g.y0 + iy * g.h;
          if (v === 0) { if (nodes) { ex.push(x); ey.push(y); } } else { fx.push(x); fy.push(y); fv.push(v); fs.push(sc * (5 + 18 * Math.min(Math.abs(v) / 40, 1))); }
        }
        Plotly.react("gridr-zoom", [
          { x: ex, y: ey, mode: "markers", marker: { size: 4, color: "#ccc" }, name: nodes ? "empty nodes" : "empty nodes (not drawn)", hoverinfo: "skip", visible: nodes ? true : "legendonly" },
          { x: fx, y: fy, mode: "markers", marker: { size: fs, color: fv, colorscale: SEISMIC, cmin: -40, cmax: 40, line: { width: 0.5, color: "#555" }, colorbar: { thickness: 10, len: 0.8, title: { text: "Wᵀy (mas)", side: "right" } } }, name: "nodes with a deposit", hovertemplate: "node %{x:.4f}, %{y:.4f}<br>%{marker.color:.1f} mas<extra></extra>" },
          { x: S.xs, y: S.ys, mode: "markers", marker: { size: nodes ? 11 : 6, color: BLACK, symbol: "star" }, name: "stars", text: S.y.map((v) => `y = ${fmt(v, 1)} mas`), hovertemplate: "star %{x:.4f}, %{y:.4f}<br>%{text}<extra></extra>" },
        ], Object.assign({}, BASE, { title: { text: `${nx}×${ny} nodes (${extent}): ${fx.length} nodes with a deposit, ${S.xs.length} stars`, font: { size: 11 } }, xaxis: { title: "u (deg)", range: [x0 - g.h, x1 + g.h], scaleanchor: "y", automargin: true }, yaxis: { title: "v (deg)", range: [y0 - g.h, y1 + g.h], automargin: true }, legend: { orientation: "h", y: -0.18, font: { size: 10 } } }), CFG);
      } else { // image view: block-summed so that each displayed cell stays visible
        const sub = new Float64Array(nx * ny);
        for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) sub[y * nx + x] = img[(iy0 + y) * g.nx + ix0 + x];
        const ds = Math.max(1, Math.ceil(Math.max(nx, ny) / HEAT_CELLS));
        const r = ds > 1 ? NUM.blockReduce(sub, nx, ny, ds, "sum") : { img: sub, nx, ny };
        const ax = blockAxes({ x0, y0, h: g.h }, r, ds);
        heat("gridr-zoom", NUM.to2d(r.img, r.nx, r.ny), ax.x, ax.y, { title: `${nx}×${ny} nodes (${extent}), ${ds > 1 ? `summed over ${ds}×${ds} nodes` : "every node"}, ${S.xs.length} stars`, zmax: 40, ctitle: "Wᵀy (mas)", hover: "%{x:.4f}, %{y:.4f}<br>%{z:.1f} mas<extra></extra>", tfont: 11, traces: ds > 1 ? [] : [starsTrace(S.xs, S.ys, 4)] });
      }
    };
    const drawText = () => {
      const { g, touched } = F, ng = g.nx * g.ny, N = V.meta.n_train;
      $("gridr-text").innerHTML = `Real numbers: h = ${fmt(g.h * 3600, 0)}" (half the 20" pixel of the map). The solver grid of this visit has ${g.nx}×${g.ny} = ${fmt(ng / 1e6, 2)} million nodes for ${N} training stars: \\(\\mathbf{W}\\) has exactly 4N = ${4 * N} non-zero entries (${touched} distinct nodes touched, <b>${fmt(100 * touched / ng, 1)}%</b> of the grid) and is never formed as a dense matrix. Values are the centred residuals y (mas) the solver works with. <b>Draw a box on the left map</b> to zoom the right panel on it: below ${MARK_VIEW} nodes a side every node with a deposit is drawn, below ${NODE_VIEW} the empty nodes too; double-click the left map to reset.`;
      renderMathInElement($("gridr-text"), { delimiters: [{ left: "\\(", right: "\\)", display: false }] });
    };
    const draw = () => { build(); drawFull(); drawZoom(); drawText(); };
    bind(["gridr-u", "gridr-v"], (e) => { comp = e.target.id.endsWith("u") ? "u" : "v"; toggle(["gridr-u", "gridr-v"], e.target.id); range = null; draw(); });
    draw();
    const gd = $("gridr-full");
    gd.on("plotly_relayout", () => { // box zoom / double-click reset on the left map
      const fl = gd._fullLayout; if (!fl || !fl.xaxis) return;
      range = fl.xaxis.autorange && fl.yaxis.autorange ? null : { x0: fl.xaxis.range[0], x1: fl.xaxis.range[1], y0: fl.yaxis.range[0], y1: fl.yaxis.range[1] };
      drawZoom();
    });
  }

  // =============================================================== 8. convolution (toy)
  function conv() {
    const n = 64, size = 3.5, g = { x0: 0, y0: 0, h: size / n, nx: n, ny: n }, XI0 = 400;
    let seed = 21;
    const draw = () => {
      const ell = +$("conv-ell").value, pad = $("conv-pad").checked;
      $("conv-ell-v").textContent = fmt(ell);
      const rr = NUM.rng(seed), ns = 40, xs = [], ys = [], vs = [];
      for (let i = 0; i < ns; i++) { xs.push(rr.uniform() * size * 0.97); ys.push(rr.uniform() * size * 0.97); vs.push(1); }
      const f = NUM.spread(xs, ys, vs, g);
      const ks = Math.round(2.5 * ell / g.h);
      const cv = NUM.makeConvolver(g, (lx, ly) => XI0 * NUM.vk(Math.hypot(lx, ly), ell), ks, pad);
      const out = cv.apply(f, true);
      const ax = NUM.linspace(0, size, n);
      heat("conv-in", NUM.to2d(f, n, n), ax, ax, { title: "f = Wᵀv on the 64×64 grid (v = 1, no unit)", zmax: 1, zmin: 0, showscale: false, colorscale: "Greys" });
      const kc = NUM.fftshift(cv.kimg, cv.L, cv.L), m = cv.L / 2, c = ks + 2, kz = [], kl = [];
      for (let y = m - c; y < m + c; y++) { const row = []; for (let x = m - c; x < m + c; x++) row.push(kc[y * cv.L + x]); kz.push(row); }
      for (let i = -c; i < c; i++) kl.push(i * g.h);
      heat("conv-kernel", kz, kl, kl, { title: `kernel image κ (mas²), support ±${fmt(ks * g.h)}°, FFT grid ${cv.L}×${cv.L}`, zmax: XI0, showscale: false, xt: "Δu (deg)", yt: "Δv (deg)" });
      heat("conv-out", NUM.to2d(out, n, n), ax, ax, { title: pad ? "κ * f (mas²), padded: no wrap" : "κ * f (mas²), NOT padded: stars near one edge leak to the opposite edge", ctitle: "κ * f (mas²)", traces: [starsTrace(xs, ys, 4)] });
    };
    bind(["conv-pad", "conv-ell"], draw);
    $("conv-new").addEventListener("click", () => { seed = Math.floor(Math.random() * 1e9); draw(); });
    draw();
  }

  // =============================================================== 8b. convolution with the real kernel
  function convreal() {
    let comp = "u";
    const draw = () => {
      const c = V.comp[comp], p = c.patch, g = patchGrid(p), pad = true, S = patchStars(comp, p);
      $("convr-n").textContent = S.xs.length;
      const t0 = performance.now();
      const cv = realConvolver(comp, pad);
      const f = NUM.spread(S.xs, S.ys, S.alpha, g);
      const out = cv.apply(f, true);
      const dt = performance.now() - t0;
      const wf = windowOf(f, p), wo = windowOf(out, p), st = inWindow(p, S);
      const za = NUM.absmax(f);
      heat("convr-in", wf.z, wf.x, wf.y, { title: `Wᵀα in the window (mas⁻¹)`, zmax: za, showscale: false, traces: [starsTrace(st.xs, st.ys, 3)] });
      const side = c.grid.kernel_side, kl = NUM.linspace(-(side >> 1) * p.h, (side >> 1) * p.h, side);
      heat("convr-kernel", c.kimg, kl, kl, { title: `κ: cleaned d${comp} map at h (mas²)`, xt: "Δu (deg)", yt: "Δv (deg)", showscale: false });
      heat("convr-out", wo.z, wo.x, wo.y, { title: `κ * Wᵀα (mas), live (${cv.L}² FFT, ${fmt(dt, 0)} ms)`, zmax: 40, showscale: false, traces: [starsTrace(st.xs, st.ys, 3)] });
      heat("convr-ref", p.mean, wo.x, wo.y, { title: "treegp mean field ŷ (mas)", zmax: 40, ctitle: "ŷ (mas)", traces: [starsTrace(st.xs, st.ys, 3)] });
      let d2 = 0, r2 = 0; for (let i = 0; i < p.win; i++) for (let j = 0; j < p.win; j++) { const d = wo.z[i][j] - p.mean[i][j]; d2 += d * d; r2 += p.mean[i][j] ** 2; }
      $("convr-text").innerHTML = `Same operation as the toy, with the real ingredients: the solved weights α of the ${S.xs.length} training stars in a ${fmt(p.nx * p.h, 2)}° box (the ${V.meta.patch_deg}° window plus a margin of one kernel support, so that every star that influences the window is present) are deposited on the real 10" grid and convolved with the real kernel image. The result matches treegp's mean field to <b>${fmt(100 * Math.sqrt(d2 / r2), 1)}%</b> rms (the rest comes from the one-time clip of the kernel spectrum, done here on a 1024² padded grid instead of treegp's 1728² one): this is Eq. 13 computed from scratch, in your browser.`;
    };
    bind(["convr-u", "convr-v"], (e) => { comp = e.target.id.endsWith("u") ? "u" : "v"; toggle(["convr-u", "convr-v"], e.target.id); draw(); });
        draw();
  }

  // =============================================================== 9. PSD by construction
  function bochner() {
    let comp = "u", seed = 4;
    const n = 64, size = 3.5, g = { x0: 0, y0: 0, h: size / n, nx: n, ny: n }, XI0 = 400;
    const draw = () => {
      const c = V.comp[comp], sc = c.spec_crop;
      const ky = NUM.linspace(-sc.rows / 2, sc.rows / 2 - 1, sc.rows).map((v) => v / (c.grid.ly * c.grid.h)), kx = NUM.linspace(0, sc.cols - 1, sc.cols).map((v) => v / (c.grid.lx * c.grid.h));
      heat("boch-spec", sc.vals, kx, ky, { title: `REAL: padded spectrum P̂_k, ${c.grid.lx}² grid, sign·log10(1+|P̂_k|)`, xt: "k_u (1/deg)", yt: "k_v (1/deg)", zmax: Math.max(...sc.vals.flat()), square: false, ctitle: "sign·log10(1+|P̂_k|)" });
      $("boch-text").innerHTML = `Zero-padding the kernel image on the ${c.grid.lx}×${c.grid.ly} FFT grid is a truncation again: the negative modes carry ${fmt(100 * c.neg_power, 1)}% of the positive power (deepest mode ${fmt(c.spectrum_min, 0)} against a peak set by ξ(0) times the kernel support; about half of all modes are tiny negative numbers at frequencies where the band-limited kernel has no power at all). They are clipped to zero <em>once</em>, at the cost of one FFT. Then Eq. 11 holds for any weights v on the stars (the variance of Σ vᵢ yᵢ is a sum of non-negative terms), \\(\\mathbf{K}\\) is positive semi-definite for every configuration of stars, \\(\\mathbf{C} = \\mathbf{K} + \\mathbf{D}\\) is positive definite, and a failure of the linear algebra is impossible rather than repaired. Effective ξ(0) after the clip: ${fmt(c.xi0_eff, 0)} mas² (cleaned map: ${fmt(c.xi0_clean, 0)}).`;
      renderMathInElement($("boch-text"), { delimiters: [{ left: "\\(", right: "\\)", display: false }] });
      // toy: sharply truncated kernel so that the unclipped spectrum has visible negative modes
      const rr = NUM.rng(seed), ns = 120, xs = [], ys = [];
      for (let i = 0; i < ns; i++) { xs.push(rr.uniform() * size * 0.97); ys.push(rr.uniform() * size * 0.97); }
      const ell = 0.5, ks = Math.round(0.9 * ell / g.h);
      const cv = NUM.makeConvolver(g, (lx, ly) => XI0 * NUM.vk(Math.hypot(lx, ly), ell), ks, true);
      const L = cv.L, kq = [], before = [], after = [];
      for (let i = 0; i <= L / 2; i++) { kq.push(i / (L * g.h)); before.push(cv.spectrum[i]); after.push(cv.clipped[i]); }
      const peak = Math.max(...before), negPow = -cv.spectrum.filter((v) => v < 0).reduce((a, b) => a + b, 0) / cv.spectrum.filter((v) => v > 0).reduce((a, b) => a + b, 0);
      Plotly.react("boch-hist", [
        { x: kq, y: before, mode: "lines", name: "spectrum of the padded kernel image, as is", line: { color: ORANGE } },
        { x: kq, y: after, mode: "lines", name: "after the clip P̂ ← max(P̂, 0)", line: { color: BLUE, dash: "dot", width: 3 } },
      ], Object.assign({}, BASE, { title: { text: `TOY: kernel cut sharply at ${fmt(ks * g.h)}°: ${cv.nneg} of ${L * L} modes < 0 (${fmt(100 * negPow, 1)}% of the + power)`, font: { size: 10 } }, xaxis: { title: "k_u (1/deg), along k_v = 0", range: [0, 12] }, yaxis: { title: "P̂_k (mas²)", range: [-0.12 * peak, 0.35 * peak] }, legend: { x: 0.3, y: 0.98, font: { size: 10 } }, shapes: [{ type: "line", x0: 0, x1: 12, y0: 0, y1: 0, line: { color: "#000", width: 1 } }] }), CFG);
    };
    bind(["boch-u", "boch-v"], (e) => { comp = e.target.id.endsWith("u") ? "u" : "v"; toggle(["boch-u", "boch-v"], e.target.id); draw(); });
    draw();
  }

  // =============================================================== 10. matvec steps (toy / whole real visit)
  function matvec() {
    const n = 64, size = 3.5, gt = { x0: 0, y0: 0, h: size / n, nx: n, ny: n }, XI0 = 400, ell = 0.3, DS = 8;
    let seed = 9, step = 0, timer = null, mode = "toy", S = null;
    const compute = (src, ds) => {
      const t0 = performance.now();
      const img = NUM.spread(src.xs, src.ys, src.v, src.g), fld = src.cv.apply(img, true), Kv = NUM.readBack(fld, src.xs, src.ys, src.g), Cv = Kv.map((k, i) => k + src.sig[i] * src.sig[i] * src.v[i]);
      S = Object.assign(src, { img, fld, Kv, Cv, ds, dt: performance.now() - t0 });
      if (S.zf === null) S.zf = NUM.absmax(fld);
      // display copies (the real grid is reduced by DS x DS blocks for Plotly)
      S.dImg = ds > 1 ? NUM.blockReduce(img, src.g.nx, src.g.ny, ds, "sum") : { img, nx: src.g.nx, ny: src.g.ny };
      S.dFld = ds > 1 ? NUM.blockReduce(fld, src.g.nx, src.g.ny, ds, "mean") : { img: fld, nx: src.g.nx, ny: src.g.ny };
    };
    const setup = () => {
      S = null;
      if (mode === "toy") {
        const rr = NUM.rng(seed), ns = 120, xs = [], ys = [], v = [], sig = [];
        for (let i = 0; i < ns; i++) { xs.push(rr.uniform() * size * 0.97); ys.push(rr.uniform() * size * 0.97); v.push(rr.normal() * 10); sig.push(3 + 12 * rr.uniform()); }
        const cv = NUM.makeConvolver(gt, (lx, ly) => XI0 * NUM.vk(Math.hypot(lx, ly), ell), Math.round(2.5 * ell / gt.h), true);
        compute({ xs, ys, v, sig, g: gt, cv, zv: 25, zf: 1500, label: "toy: 120 stars, 64×64 grid" }, 1);
        draw();
      } else {
        const c = V.comp.u, t = trainOf("u"), g = fullGrid(c), useAlpha = mode === "alpha";
        $("mv-text").innerHTML = `computing the four steps on the whole visit (${t.xs.length} stars, ${g.nx}×${g.ny} grid, 2048² padded FFT)…`;
        Plotly.purge("mv-plot");
        setTimeout(() => { // let the message paint first
          const cv = fullConvolver("u");
          compute({ xs: t.xs, ys: t.ys, v: useAlpha ? t.alpha : t.y, y: t.y, sig: t.sig, g, cv, zv: useAlpha ? NUM.absmax(t.alpha) / 3 : 60, zf: useAlpha ? 60 : null, alpha: useAlpha, label: `real visit (v = ${useAlpha ? "α" : "y"}): ${t.xs.length} stars, ${g.nx}×${g.ny} grid, ${cv.L}² padded FFT` }, DS);
          draw();
        }, 40);
      }
    };
    const draw = () => {
      if (!S) return;
      const g = S.g, d1 = S.dImg, d2 = S.dFld, ds = S.ds, a1 = blockAxes(g, d1, ds), a2 = blockAxes(g, d2, ds);
      const xr = [g.x0, g.x0 + (g.nx - 1) * g.h], yr = [g.y0, g.y0 + (g.ny - 1) * g.h];
      toggle(["mv-0", "mv-1", "mv-2", "mv-3", "mv-4"], `mv-${step}`);
      [...$("mv-steps").children].forEach((li, i) => li.classList.toggle("active", i + 1 === step));
      const pts = (vals, title, zmax, unit) => Plotly.react("mv-plot", [{ type: mode === "toy" ? "scatter" : "scattergl", x: S.xs, y: S.ys, mode: "markers", marker: { size: mode === "toy" ? 9 : 3, color: vals, colorscale: SEISMIC, cmin: -zmax, cmax: zmax, colorbar: { thickness: 10, title: { text: unit, side: "right" } }, line: { width: mode === "toy" ? 0.5 : 0, color: "#333" } }, hovertemplate: "%{x:.3f}, %{y:.3f}<br>%{marker.color:.1f}<extra></extra>" }], Object.assign({}, BASE, { title: { text: title, font: { size: 13 } }, xaxis: { range: xr, title: "u (deg)", constrain: "domain", automargin: true }, yaxis: { range: yr, title: "v (deg)", scaleanchor: "x", constrain: "domain", automargin: true } }), CFG);
      const shown = (how) => (ds > 1 ? `, shown ${how} over ${ds}×${ds} nodes` : "");
      const uv = S.alpha ? "mas⁻¹" : "mas", uk = S.alpha ? "mas" : "mas³";
      if (step === 0) pts(S.v, S.alpha ? "the weights α from the solve of Eq. 3: one value per star (mas⁻¹)" : `a vector v: one value per star (mas)${mode === "toy" ? "" : ": here the real residuals y"}`, S.zv, `v (${uv})`);
      if (step === 1) heat("mv-plot", NUM.to2d(d1.img, d1.nx, d1.ny), a1.x, a1.y, { title: `step 1: Wᵀv (${uv}), the values deposited on the 4 nodes around each star${shown("summed")}`, zmax: S.zv, ctitle: `Wᵀv (${uv})` });
      if (step === 2) heat("mv-plot", NUM.to2d(d2.img, d2.nx, d2.ny), a2.x, a2.y, { title: `step 2: κ * (Wᵀv) (${uk}) by padded FFT (${S.cv.L}×${S.cv.L}): the field these weights create everywhere${shown("averaged")}`, zmax: S.zf, ctitle: `κ * Wᵀv (${uk})` });
      if (step === 3) pts(S.Kv, `step 3: read the field back at the stars (W): this is Kv (${uk})`, S.zf, `Kv (${uk})`);
      if (step === 4) pts(S.Cv, S.alpha ? `step 4: Cα = Kα + Dα = y, the data are recovered (rms |Cα − y| = ${fmt(Math.sqrt(S.Cv.reduce((t, c, i) => t + (c - S.y[i]) ** 2, 0) / S.Cv.length), 2)} mas, from the 4-digit rounding of the exported α; 1e-5 mas in treegp): Eq. 3 is solved` : `step 4: add σᵢ² vᵢ: Cv = Kv + Dv (${uk})`, S.zf, `Cv (${uk})`);
      $("mv-text").innerHTML = `Cost per application: \\(O(N + N_g \\log N_g)\\) time, \\(O(N + N_g)\\) memory (dense: \\(O(N^3)\\), \\(O(N^2)\\)); the only operation the solver needs. Eq. 3 is the question (find α with Cα = y); Eq. 12 is the tool: a black box returning Cv for any v with one value per star. The conjugate gradient (next slide) calls it once per iteration on its current search direction, the first one being y divided star by star by its total variance; fed with the final α, it returns y. ${S.label}: ${fmt(S.dt, 0)} ms here in your browser (~30 ms in treegp). Units: \\(\\mathbf{C}\\) is in mas², so with v in ${uv} the results are in ${uk}.`;
      renderMathInElement($("mv-text"), { delimiters: [{ left: "\\(", right: "\\)", display: false }] });
    };
    setup();
    ["mv-0", "mv-1", "mv-2", "mv-3", "mv-4"].forEach((id, i) => $(id).addEventListener("click", () => { step = i; draw(); }));
    $("mv-play").addEventListener("click", () => { if (timer) { clearInterval(timer); timer = null; return; } step = 0; draw(); timer = setInterval(() => { step = (step + 1) % 5; draw(); if (step === 4) { clearInterval(timer); timer = null; } }, 1300); });
    $("mv-new").addEventListener("click", () => { if (mode !== "toy") return; seed = Math.floor(Math.random() * 1e9); step = 0; setup(); });
    const setMode = (m) => { mode = m; toggle(["mv-toy", "mv-real", "mv-alpha"], `mv-${m}`); $("mv-tag").className = `tag ${m === "toy" ? "toy" : "real"}`; $("mv-tag").textContent = m === "toy" ? "toy" : "real visit"; $("mv-new").disabled = m !== "toy"; step = 0; setup(); };
    $("mv-toy").addEventListener("click", () => setMode("toy"));
    $("mv-real").addEventListener("click", () => { if (V) setMode("real"); });
    $("mv-alpha").addEventListener("click", () => { if (V) setMode("alpha"); });
    if (!V) { $("mv-real").disabled = true; $("mv-alpha").disabled = true; }
  }

  // =============================================================== 11. conjugate gradient (toy / real)
  function cg() {
    // (a) 2D bowl
    const A = [[8, 3], [3, 1.5]], b = [11, 4.5]; // solution (1, 1), condition number ~28
    const mv2 = (p) => Float64Array.from([A[0][0] * p[0] + A[0][1] * p[1], A[1][0] * p[0] + A[1][1] * p[1]]);
    const x0 = [0, 4];
    const sd = NUM.steepest(mv2, Float64Array.from(b), [1, 1], { maxit: 25, keepIterates: true, x0 });
    const cgp = NUM.pcg(mv2, Float64Array.from(b), [1, 1], { maxit: 2, keepIterates: true, rtol: 1e-12, x0 });
    const gx = NUM.linspace(-0.5, 2.5, 80), gy = NUM.linspace(-3, 5, 80);
    const fz = gy.map((y) => gx.map((x) => 0.5 * (A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y) - b[0] * x - b[1] * y));
    const fmin = -0.5 * (b[0] + b[1]); // value at the solution
    Plotly.react("cg-bowl", [
      { type: "contour", x: gx, y: gy, z: fz, colorscale: "Blues", reversescale: true, showscale: false, contours: { start: fmin + 0.05, end: fmin + 30, size: 0.75 }, line: { width: 1 }, hoverinfo: "skip" },
      { x: sd.iterates.map((p) => p[0]), y: sd.iterates.map((p) => p[1]), mode: "lines+markers", name: "steepest descent (25 steps)", line: { color: ORANGE }, marker: { size: 5 } },
      { x: cgp.iterates.map((p) => p[0]), y: cgp.iterates.map((p) => p[1]), mode: "lines+markers", name: "conjugate gradient (2 steps)", line: { color: BLUE, width: 3 }, marker: { size: 9 } },
    ], Object.assign({}, BASE, { title: { text: "TOY: why conjugate directions (2 unknowns, solution at (1, 1))", font: { size: 13 } }, xaxis: { title: "α₁", range: [-0.5, 2.5] }, yaxis: { title: "α₂", range: [-3, 5] }, legend: { x: 0.02, y: 0.98, font: { size: 10 } } }), CFG);
    // (b) toy GP solve / real history
    const n = 64, size = 3.5, g = { x0: 0, y0: 0, h: size / n, nx: n, ny: n }, XI0 = 400, ell = 0.3, ns = 300;
    let seed = 13, T = null, timer = null, mode = "toy";
    const setup = () => {
      const rr = NUM.rng(seed), xs = [], ys = [], sig = [];
      for (let i = 0; i < ns; i++) { xs.push(rr.uniform() * size * 0.97); ys.push(rr.uniform() * size * 0.97); sig.push(2 + 25 * rr.uniform() ** 2); }
      const kfun = (lx, ly) => XI0 * NUM.vk(NUM.anisoRadius(lx, ly, 0.3, 0.2), ell);
      const truthImg = NUM.grf(n, NUM.kernelImage(n, g.h, kfun), rr);
      const y = NUM.sampleImage(truthImg, n, g.h, xs, ys).map((t, i) => t + sig[i] * rr.normal());
      const cv = NUM.makeConvolver(g, kfun, Math.round(2.5 * ell / g.h), true);
      const matvec = (v) => { const Kv = NUM.readBack(cv.apply(NUM.spread(xs, ys, v, g), true), xs, ys, g); return Kv.map((k, i) => k + sig[i] * sig[i] * v[i]); };
      const solve = (pre) => NUM.pcg(matvec, Float64Array.from(y), sig.map((s) => (pre ? 1 / (XI0 + s * s) : 1)), { maxit: 60, keepIterates: true, rtol: 1e-9 });
      T = { xs, ys, sig, y, cv, truthImg, pre: solve(true), nopre: solve(false) };
    };
    const residLayout = (title) => Object.assign({}, BASE, { margin: { l: 50, r: 10, t: 22, b: 30 }, title: { text: title, font: { size: 12 } }, yaxis: { type: "log", range: [-8, 0.2], title: "‖r‖/‖y‖", automargin: true }, xaxis: { title: "iteration", automargin: true }, legend: { x: 0.5, y: 0.95, font: { size: 9 } } });
    const drawToy = () => {
      const pre = $("cg-precond").checked, k = Math.min(+$("cg-k").value, 60), res = pre ? T.pre : T.nopre;
      const kk = Math.min(k, res.iterates.length - 1);
      $("cg-k-v").textContent = kk;
      const alpha = res.iterates[kk], fld = T.cv.apply(NUM.spread(T.xs, T.ys, alpha, g), true), ax = NUM.linspace(0, size, n);
      heat("cg-field", NUM.to2d(fld, n, n), ax, ax, { title: `TOY: predicted field (mas) after k = ${kk} iterations (N = ${ns} toy stars)`, zmax: 50, showscale: false, traces: [starsTrace(T.xs, T.ys, 3)] });
      Plotly.react("cg-resid", [
        { x: T.pre.hist.map((_, i) => i), y: T.pre.hist, mode: "lines", name: "with Jacobi preconditioner", line: { color: BLUE } },
        { x: T.nopre.hist.map((_, i) => i), y: T.nopre.hist, mode: "lines", name: "without", line: { color: ORANGE } },
        { x: [kk], y: [res.hist[kk]], mode: "markers", marker: { size: 10, color: BLACK }, showlegend: false },
      ], residLayout("relative residual ‖r‖/‖y‖"), CFG);
    };
    const drawReal = () => {
      const c = V.comp.u, R = c.cg, snaps = R.snapshots, i = Math.min(+$("cg-k").value, snaps.length);
      const k = i === 0 ? 0 : snaps[i - 1].k;
      $("cg-k-v").textContent = k;
      const m = c.mean, x = NUM.linspace(m.x0, m.x0 + (R.nx - 1) * c.grid.h * R.ds, R.nx), y = NUM.linspace(m.y0, m.y0 + (R.ny - 1) * c.grid.h * R.ds, R.ny);
      const z = i === 0 ? Array.from({ length: R.ny }, () => Array(R.nx).fill(0)) : snaps[i - 1].img;
      heat("cg-field", z, x, y, { title: `REAL: mean field du (mas) after k = ${k} of ${R.n_pre} iterations (N = ${V.meta.n_train} stars)`, zmax: 40, showscale: false });
      Plotly.react("cg-resid", [
        { x: R.hist_pre.map((_, j) => j), y: R.hist_pre, mode: "lines", name: `with Jacobi preconditioner (${R.n_pre} it.)`, line: { color: BLUE } },
        { x: R.hist_nopre.map((_, j) => j), y: R.hist_nopre, mode: "lines", name: `without (${R.n_nopre} it.)`, line: { color: ORANGE } },
        { x: [k], y: [R.hist_pre[k]], mode: "markers", marker: { size: 10, color: BLACK }, showlegend: false },
      ], residLayout("REAL: relative residual ‖r‖/‖y‖ of the du solve"), CFG);
    };
    const draw = () => (mode === "toy" ? drawToy() : drawReal());
    const setMode = (mm) => {
      mode = mm; toggle(["cg-toy", "cg-real"], `cg-${mm}`);
      const sl = $("cg-k");
      if (mm === "toy") { sl.max = 60; sl.value = Math.min(+sl.value, 60); $("cg-precond").disabled = false; $("cg-new").disabled = false; }
      else { sl.max = V.comp.u.cg.snapshots.length; sl.value = Math.min(+sl.value, +sl.max); $("cg-precond").disabled = true; $("cg-new").disabled = true; }
      draw();
      $("cg-text").innerHTML = BASE_TEXT + (mm === "toy" ? "" : ` <b>Real visit:</b> the recorded history of the du solve: ${V.comp.u.cg.n_pre} iterations with the Jacobi preconditioner (${V.comp.v.cg.n_pre} for dv), ${V.comp.u.cg.n_nopre} without (${V.comp.v.cg.n_nopre}), ${fmt(V.comp.u.t_solve + V.comp.u.t_predict, 1)} s per component (solve and prediction) for ${V.meta.n_train} stars on a ${V.comp.u.grid.lx}×${V.comp.u.grid.ly} padded grid. The slider steps through the mean field at the recorded iterations (preconditioned run).`);
      renderMathInElement($("cg-text"), { delimiters: [{ left: "\\(", right: "\\)", display: false }] });
    };
    const BASE_TEXT = $("cg-text").innerHTML;
    setup(); setMode("toy");
    bind(["cg-k", "cg-precond"], draw);
    $("cg-new").addEventListener("click", () => { if (mode !== "toy") return; seed = Math.floor(Math.random() * 1e9); setup(); draw(); });
    $("cg-play").addEventListener("click", () => { if (timer) { clearInterval(timer); timer = null; return; } const sl = $("cg-k"); sl.value = 0; draw(); const dt = mode === "toy" ? 150 : 700; timer = setInterval(() => { const v = +sl.value + 1; sl.value = v; draw(); if (v >= +sl.max) { clearInterval(timer); timer = null; } }, dt); });
    $("cg-toy").addEventListener("click", () => setMode("toy"));
    $("cg-real").addEventListener("click", () => { if (V) setMode("real"); });
    if (!V) $("cg-real").disabled = true;
  }

  // =============================================================== 12. prediction as a field
  function field2() {
    let comp = "u";
    const draw = () => {
      const c = V.comp[comp], m = c.mean;
      const x = NUM.linspace(m.x0, m.x0 + (m.nx - 1) * m.h, m.nx), y = NUM.linspace(m.y0, m.y0 + (m.ny - 1) * m.h, m.ny);
      heat("f2-mean", m.img, x, y, { title: `mean field ŷ(X) for d${comp}: one convolution of Wᵀα, stored as an image (mas)`, zmax: 40, ctitle: "ŷ (mas)", hover: "%{x:.3f}, %{y:.3f}<br>%{z:.1f} mas<extra></extra>" });
      Plotly.react("f2-eb", ebTraces(V.eb, "all"), ebLayout("validation stars: before / after"), CFG);
      $("f2-text").innerHTML = `Eq. 13: \\(\\hat y(\\mathbf{X}) = \\sum_j \\alpha_j\\, \\xi(\\mathbf{X} - \\mathbf{X}_j)\\). Once α is known, this applied everywhere is a field: the convolution of the kernel with the weights placed at the stars. On the grid, that is one more FFT convolution of Wᵀα, computed once per component (${fmt(c.t_predict, 2)} s here for ${V.meta.n_val} stars) and stored. Each of the 189 CCDs then reads its 30×30 prediction grid off this image by bilinear interpolation, at a cost independent of N; the dense approach needed an M×N cross-covariance matrix per CCD. Far from every star the field is zero. E-mode ${fmt(e0(V.eb.E_before), 0)} → ${fmt(e0(V.eb.E_after), 0)} mas²; rms per star ${fmt(V.meta.rms_before, 1)} → ${fmt(V.meta.rms_after, 1)} mas (the remainder is mostly the ${fmt(c.noise_median, 0)} mas measurement noise of the sources).`;
      renderMathInElement($("f2-text"), { delimiters: [{ left: "\\(", right: "\\)", display: false }] });
    };
    bind(["f2-u", "f2-v"], (e) => { comp = e.target.id.endsWith("u") ? "u" : "v"; toggle(["f2-u", "f2-v"], e.target.id); draw(); });
    draw();
  }

  // =============================================================== 13. cost
  function cost() {
    const B = V ? V.bench : { dense: [{ n: 10000, t: 85, mem: 3.7 }, { n: 15000, t: 140, mem: 6.6 }], spectral: [{ n: 10000, t: 4.5, mem: 0.55 }, { n: 15000, t: 5, mem: 0.6 }] };
    const dT = (n) => 85 * Math.pow(n / 1e4, 3), dM = (n) => 3.7 * Math.pow(n / 1e4, 2), sT = (n) => 3.5 + 0.1 * n / 1e3, sM = (n) => 0.5 + 0.007 * n / 1e3;
    const ns = NUM.linspace(2000, 60000, 59);
    const draw = () => {
      const N = +$("cost-n").value; $("cost-n-v").textContent = N;
      Plotly.react("cost-time", [
        { x: ns, y: ns.map(dT), mode: "lines", name: "dense (Cholesky + M×N predict) ∝ N³", line: { color: ORANGE } },
        { x: ns, y: ns.map(sT), mode: "lines", name: "TURBO-GP grid (CG + FFT) ∝ N", line: { color: BLUE } },
        { x: B.dense.map((d) => d.n), y: B.dense.map((d) => d.t), mode: "markers", marker: { color: ORANGE, size: 10 }, name: "measured, dense" },
        { x: B.spectral.map((d) => d.n), y: B.spectral.map((d) => d.t), mode: "markers", marker: { color: BLUE, size: 10 }, name: "measured, grid" },
        { x: [N, N], y: [1, 2e4], mode: "lines", line: { color: "#000", dash: "dot", width: 1 }, showlegend: false },
      ], Object.assign({}, BASE, { title: { text: "time per component (s)", font: { size: 13 } }, xaxis: { title: "training stars N" }, yaxis: { type: "log", title: "seconds" }, legend: { x: 0.02, y: 0.98, font: { size: 10 } } }), CFG);
      Plotly.react("cost-mem", [
        { x: ns, y: ns.map(dM), mode: "lines", name: "dense ∝ N²", line: { color: ORANGE } },
        { x: ns, y: ns.map(sM), mode: "lines", name: "grid, ~flat", line: { color: BLUE } },
        { x: B.dense.map((d) => d.n), y: B.dense.map((d) => d.mem), mode: "markers", marker: { color: ORANGE, size: 10 }, name: "measured, dense" },
        { x: B.spectral.map((d) => d.n), y: B.spectral.map((d) => d.mem), mode: "markers", marker: { color: BLUE, size: 10 }, name: "measured, grid" },
        { x: [2000, 60000], y: [4, 4], mode: "lines", line: { color: "#000", dash: "dash" }, name: "DRP budget: 4 GB per core" },
        { x: [N, N], y: [0.1, 200], mode: "lines", line: { color: "#000", dash: "dot", width: 1 }, showlegend: false },
      ], Object.assign({}, BASE, { title: { text: "peak memory per job (GB)", font: { size: 13 } }, xaxis: { title: "training stars N" }, yaxis: { type: "log", title: "GB" }, legend: { x: 0.02, y: 0.98, font: { size: 10 } } }), CFG);
      $("cost-read").innerHTML = `N = ${N}: dense ≈ <b>${fmt(dT(N), 0)} s</b>, <b>${fmt(dM(N), 1)} GB</b> &nbsp;|&nbsp; grid ≈ <b>${fmt(sT(N), 0)} s</b>, <b>${fmt(sM(N), 1)} GB</b>`;
    };
    bind(["cost-n"], draw);
    draw();
  }

  const table = { field, grf, gp1d, kernel, clean, psd, psdreal, grid, gridreal, conv, convreal, bochner, matvec, cg, field2, cost };
  const needData = ["field", "kernel", "clean", "psdreal", "gridreal", "convreal", "bochner", "field2"];
  function initFor(slide) {
    const id = slide && slide.dataset.demo;
    if (!id || id === "none" || inited[id]) { if (id && inited[id]) resizeIn(slide); return; }
    if (!V && needData.includes(id)) { slide.insertAdjacentHTML("beforeend", '<p class="note">data/visit.js is missing: run make_data.py to build the real-data slides.</p>'); inited[id] = true; return; }
    inited[id] = true;
    try { table[id](); } catch (e) { console.error("demo", id, e); slide.insertAdjacentHTML("beforeend", `<p class="note">demo error: ${e.message}</p>`); }
  }
  function resizeIn(slide) { slide.querySelectorAll(".js-plotly-plot").forEach((p) => Plotly.Plots.resize(p)); }
  function start() {
    if (V) $("title-visit").textContent = `visit ${V.meta.visit}, ${V.meta.n_all} stars`;
    Reveal.on("ready", (e) => initFor(e.currentSlide));
    Reveal.on("slidechanged", (e) => initFor(e.currentSlide));
    if (Reveal.isReady && Reveal.isReady()) initFor(Reveal.getCurrentSlide());
  }
  return { start };
})();
