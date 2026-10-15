/* Numerics for the TURBO-GP slides. Plain JS, no dependencies.
   Arrays are Float64Array, images are row-major (ny, nx), index iy*nx+ix. */
"use strict";
const NUM = (() => {

  // ---------------------------------------------------------------- RNG
  function rng(seed) {
    let a = seed >>> 0;
    const u = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const n = () => {
      let v = 0, w = 0, s = 0;
      do { v = 2 * u() - 1; w = 2 * u() - 1; s = v * v + w * w; } while (s >= 1 || s === 0);
      return v * Math.sqrt(-2 * Math.log(s) / s);
    };
    return { uniform: u, normal: n };
  }

  // ---------------------------------------------------------------- FFT (radix 2)
  function fft1d(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = 2 * Math.PI / len * (inverse ? 1 : -1);
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let j = 0; j < len / 2; j++) {
          const a = i + j, b = i + j + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  // in-place 2D FFT on square or rectangular power-of-2 images (ny, nx)
  function fft2d(re, im, nx, ny, inverse) {
    const rr = new Float64Array(nx), ri = new Float64Array(nx);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) { rr[x] = re[y * nx + x]; ri[x] = im[y * nx + x]; }
      fft1d(rr, ri, inverse);
      for (let x = 0; x < nx; x++) { re[y * nx + x] = rr[x]; im[y * nx + x] = ri[x]; }
    }
    const cr = new Float64Array(ny), ci = new Float64Array(ny);
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) { cr[y] = re[y * nx + x]; ci[y] = im[y * nx + x]; }
      fft1d(cr, ci, inverse);
      for (let y = 0; y < ny; y++) { re[y * nx + x] = cr[y]; im[y * nx + x] = ci[y]; }
    }
  }
  // naive separable DFT for arbitrary (small) sizes, returns {re, im}
  function dft2d(re, im, n, inverse) {
    const s = inverse ? 1 : -1;
    const cs = new Float64Array(n * n), sn = new Float64Array(n * n);
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) {
      cs[k * n + j] = Math.cos(2 * Math.PI * k * j / n); sn[k * n + j] = s * Math.sin(2 * Math.PI * k * j / n);
    }
    const tr = new Float64Array(n * n), ti = new Float64Array(n * n);
    for (let y = 0; y < n; y++) for (let k = 0; k < n; k++) {
      let ar = 0, ai = 0;
      for (let x = 0; x < n; x++) {
        const c = cs[k * n + x], d = sn[k * n + x], a = re[y * n + x], b = im[y * n + x];
        ar += a * c - b * d; ai += a * d + b * c;
      }
      tr[y * n + k] = ar; ti[y * n + k] = ai;
    }
    const or = new Float64Array(n * n), oi = new Float64Array(n * n);
    for (let x = 0; x < n; x++) for (let k = 0; k < n; k++) {
      let ar = 0, ai = 0;
      for (let y = 0; y < n; y++) {
        const c = cs[k * n + y], d = sn[k * n + y], a = tr[y * n + x], b = ti[y * n + x];
        ar += a * c - b * d; ai += a * d + b * c;
      }
      or[k * n + x] = ar; oi[k * n + x] = ai;
    }
    if (inverse) for (let i = 0; i < n * n; i++) { or[i] /= n * n; oi[i] /= n * n; }
    return { re: or, im: oi };
  }
  const nextPow2 = (n) => 1 << Math.ceil(Math.log2(n));
  // fftshift for an (n,n) image: move origin from (0,0) to (n/2,n/2) (or back, same op for even n)
  function fftshift(img, nx, ny) {
    const out = new Float64Array(nx * ny);
    const sx = nx >> 1, sy = ny >> 1;
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++)
      out[((y + sy) % ny) * nx + ((x + sx) % nx)] = img[y * nx + x];
    return out;
  }

  // ---------------------------------------------------------------- kernels
  // Modified Bessel K_nu(x), nu = 5/6, by the integral representation.
  function besselK(nu, x) {
    if (x <= 0) return Infinity;
    const tmax = Math.max(2, Math.log(40 / x) + 3), nt = 400, dt = tmax / nt;
    let s = 0;
    for (let i = 0; i <= nt; i++) {
      const t = i * dt, w = (i === 0 || i === nt) ? 0.5 : 1;
      s += w * Math.exp(-x * Math.cosh(t)) * Math.cosh(nu * t);
    }
    return s * dt;
  }
  // von Karman-shaped correlation (L21 Eq. 6 form), normalized to 1 at r=0, tabulated once.
  const VK_TAB = (() => {
    const n = 4000, xmax = 40, tab = new Float64Array(n + 1);
    const norm = 1.0057; // 2^{-1/6} Gamma(5/6) = lim x^{5/6} K_{5/6}(x)
    tab[0] = 1;
    for (let i = 1; i <= n; i++) { const x = i * xmax / n; tab[i] = Math.pow(x, 5 / 6) * besselK(5 / 6, x) / norm; }
    return { tab, xmax, n };
  })();
  function vonKarman(r, ell) {
    const x = 2 * Math.PI * r / ell;
    if (x >= VK_TAB.xmax) return 0;
    const f = x / VK_TAB.xmax * VK_TAB.n, i = Math.floor(f), t = f - i;
    return VK_TAB.tab[i] * (1 - t) + VK_TAB.tab[Math.min(i + 1, VK_TAB.n)] * t;
  }
  // same profile parametrized by its half-width (xi = xi(0)/2 at r = hw): hw = 0.174 ell
  const vk = (r, hw) => vonKarman(r, hw / 0.174);
  const gaussian = (r, ell) => Math.exp(-0.5 * r * r / (ell * ell));
  // anisotropic radius with the L21 (g1, g2) shear parametrization (treegp get_correlation_length_matrix)
  function anisoRadius(dx, dy, g1, g2) {
    const e = Math.hypot(g1, g2), q = (1 - e) / (1 + e), phi = 0.5 * Math.atan2(g2, g1);
    const c = Math.cos(phi), s = Math.sin(phi);
    const a = c * dx + s * dy, b = -s * dx + c * dy; // rotate into the ellipse frame
    return Math.sqrt(a * a + (b * b) / (q * q));
  }
  // kernel image on an (n,n) grid in FFT order (lag 0 at index 0), spacing h
  function kernelImage(n, h, kfun) {
    const img = new Float64Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const lx = (x < n / 2 ? x : x - n) * h, ly = (y < n / 2 ? y : y - n) * h;
      img[y * n + x] = kfun(lx, ly);
    }
    return img;
  }
  // Gaussian random field with the given kernel image (FFT order), unit-free: variance = kfun(0,0)
  function grf(n, kimg, r) {
    if (n & (n - 1)) throw new Error(`grf needs a power-of-2 grid, got ${n}`);
    const re = Float64Array.from(kimg), im = new Float64Array(n * n);
    fft2d(re, im, n, n, false);
    const wr = new Float64Array(n * n), wi = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) wr[i] = r.normal();
    fft2d(wr, wi, n, n, false);
    for (let i = 0; i < n * n; i++) {
      const p = Math.max(re[i], 0), s = Math.sqrt(p / (n * n));
      wr[i] *= s; wi[i] *= s;
    }
    fft2d(wr, wi, n, n, true);
    for (let i = 0; i < n * n; i++) wr[i] *= n; // keep variance = kimg[0]
    return wr;
  }
  // 2D autocorrelation of an (n,n) image by FFT, returns image centered at (n/2,n/2)
  function autocorr(img, n) {
    const re = Float64Array.from(img), im = new Float64Array(n * n);
    fft2d(re, im, n, n, false);
    for (let i = 0; i < n * n; i++) { re[i] = (re[i] * re[i] + im[i] * im[i]) / (n * n); im[i] = 0; }
    fft2d(re, im, n, n, true);
    return fftshift(re, n, n);
  }
  function sampleImage(img, n, h, xs, ys) { // bilinear, periodic
    const out = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      const gx = xs[i] / h, gy = ys[i] / h;
      const ix = Math.floor(gx), iy = Math.floor(gy), tx = gx - ix, ty = gy - iy;
      const x0 = ((ix % n) + n) % n, x1 = (x0 + 1) % n, y0 = ((iy % n) + n) % n, y1 = (y0 + 1) % n;
      out[i] = (1 - tx) * (1 - ty) * img[y0 * n + x0] + tx * (1 - ty) * img[y0 * n + x1]
        + (1 - tx) * ty * img[y1 * n + x0] + tx * ty * img[y1 * n + x1];
    }
    return out;
  }

  // ---------------------------------------------------------------- 2-pcf by pair counting
  // returns {map (npix x npix, zero lag at npix/2), counts}
  function pairCount2pcf(xs, ys, f, npix, pix) {
    const map = new Float64Array(npix * npix), cnt = new Float64Array(npix * npix);
    const half = npix / 2, n = xs.length;
    let mean = 0; for (let i = 0; i < n; i++) mean += f[i]; mean /= n;
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
      const dx = xs[j] - xs[i], dy = ys[j] - ys[i];
      const bx = Math.floor(dx / pix + half), by = Math.floor(dy / pix + half);
      if (bx < 0 || bx >= npix || by < 0 || by >= npix) continue;
      const v = (f[i] - mean) * (f[j] - mean);
      map[by * npix + bx] += v; cnt[by * npix + bx] += 1;
      const bx2 = Math.floor(-dx / pix + half), by2 = Math.floor(-dy / pix + half);
      if (i !== j && bx2 >= 0 && bx2 < npix && by2 >= 0 && by2 < npix) { map[by2 * npix + bx2] += v; cnt[by2 * npix + bx2] += 1; }
    }
    for (let i = 0; i < map.length; i++) map[i] = cnt[i] > 0 ? map[i] / cnt[i] : 0;
    return { map, cnt };
  }

  // ---------------------------------------------------------------- apodization + threshold (treegp rules)
  function hann(r) { return r >= 1 ? 0 : 0.5 * (1 + Math.cos(Math.PI * r)); }
  function blackmanHarris(r) {
    if (r >= 1) return 0;
    const a = [0.35875, 0.48829, 0.14128, 0.01168], t = Math.PI * (1 + r); // r in [0,1] -> second half of the window
    return a[0] - a[1] * Math.cos(t) + a[2] * Math.cos(2 * t) - a[3] * Math.cos(3 * t);
  }
  // window on an (n,n) map with zero lag at (n/2,n/2), reaching zero at rmax pixels (elliptical with g1,g2)
  function apodWindow(n, rmax, type, g1, g2) {
    const w = new Float64Array(n * n), c = n >> 1, fn = type === "hann" ? hann : blackmanHarris;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const r = anisoRadius(x - c, y - c, g1 || 0, g2 || 0) / rmax;
      w[y * n + x] = fn(r);
    }
    return w;
  }
  // power spectrum of a centered (n,n) correlation map: real part of the DFT (symmetric map), centered
  function corr2power(map, n) {
    const sh = fftshift(map, n, n), im = new Float64Array(n * n);
    let out;
    if ((n & (n - 1)) === 0) { const re = Float64Array.from(sh); fft2d(re, im, n, n, false); out = re; }
    else out = dft2d(sh, im, n, false).re;
    return fftshift(out, n, n);
  }
  function power2corr(pk, n) {
    const sh = fftshift(pk, n, n), im = new Float64Array(n * n);
    let out;
    if ((n & (n - 1)) === 0) { const re = Float64Array.from(sh); fft2d(re, im, n, n, true); out = re; }
    else out = dft2d(sh, im, n, true).re;
    return fftshift(out, n, n);
  }
  function percentile(arr, p) {
    const a = Float64Array.from(arr).sort(); const k = (a.length - 1) * p / 100, i = Math.floor(k), t = k - i;
    return a[i] * (1 - t) + a[Math.min(i + 1, a.length - 1)] * t;
  }
  // treegp _threshold: noise from the 16-50-84 percentiles of the non-zero modes
  function thresholdSpectrum(pk, nSigma) {
    const nz = Array.from(pk).filter((v) => v !== 0);
    if (nz.length === 0) return { pk: Float64Array.from(pk), thresh: 0, kept: 0 };
    const p16 = percentile(nz, 16), p50 = percentile(nz, 50), p84 = percentile(nz, 84);
    const thresh = p50 + nSigma * 0.5 * (p84 - p16);
    const out = new Float64Array(pk.length); let kept = 0;
    for (let i = 0; i < pk.length; i++) if (pk[i] > thresh) { out[i] = pk[i]; kept++; }
    return { pk: out, thresh, kept };
  }

  // ---------------------------------------------------------------- grid machinery (K = W G W^T)
  // bilinear spread of values at points (x,y) onto grid (nx,ny) with origin (x0,y0), spacing h
  function spread(xs, ys, vals, g) {
    const img = new Float64Array(g.nx * g.ny);
    for (let i = 0; i < xs.length; i++) {
      const gx = (xs[i] - g.x0) / g.h, gy = (ys[i] - g.y0) / g.h;
      const ix = Math.floor(gx), iy = Math.floor(gy), tx = gx - ix, ty = gy - iy;
      if (ix < 0 || iy < 0 || ix + 1 >= g.nx || iy + 1 >= g.ny) continue;
      img[iy * g.nx + ix] += (1 - tx) * (1 - ty) * vals[i];
      img[iy * g.nx + ix + 1] += tx * (1 - ty) * vals[i];
      img[(iy + 1) * g.nx + ix] += (1 - tx) * ty * vals[i];
      img[(iy + 1) * g.nx + ix + 1] += tx * ty * vals[i];
    }
    return img;
  }
  function readBack(img, xs, ys, g) {
    const out = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      const gx = (xs[i] - g.x0) / g.h, gy = (ys[i] - g.y0) / g.h;
      const ix = Math.floor(gx), iy = Math.floor(gy), tx = gx - ix, ty = gy - iy;
      if (ix < 0 || iy < 0 || ix + 1 >= g.nx || iy + 1 >= g.ny) continue;
      out[i] = (1 - tx) * (1 - ty) * img[iy * g.nx + ix] + tx * (1 - ty) * img[iy * g.nx + ix + 1]
        + (1 - tx) * ty * img[(iy + 1) * g.nx + ix] + tx * ty * img[(iy + 1) * g.nx + ix + 1];
    }
    return out;
  }
  // Reduce a (ny, nx) image by ds x ds blocks (sum or mean); returns {img, nx, ny}.
  function blockReduce(img, nx, ny, ds, how = "sum") {
    const mx = Math.ceil(nx / ds), my = Math.ceil(ny / ds), out = new Float64Array(mx * my), cnt = new Float64Array(mx * my);
    for (let y = 0; y < ny; y++) { const by = Math.floor(y / ds) * mx; for (let x = 0; x < nx; x++) { const b = by + Math.floor(x / ds); out[b] += img[y * nx + x]; cnt[b]++; } }
    if (how === "mean") for (let i = 0; i < out.length; i++) out[i] /= cnt[i];
    return { img: out, nx: mx, ny: my };
  }
  // Sub-image [iy0, iy0+ny) x [ix0, ix0+nx) of a (NY, NX) image, as a 2d array.
  function crop2d(img, NX, ix0, iy0, nx, ny) {
    const out = []; for (let y = iy0; y < iy0 + ny; y++) { const row = new Array(nx); for (let x = 0; x < nx; x++) row[x] = img[y * NX + ix0 + x]; out.push(row); }
    return out;
  }
  function bilinearWeights(x, y, g) {
    const gx = (x - g.x0) / g.h, gy = (y - g.y0) / g.h;
    const ix = Math.floor(gx), iy = Math.floor(gy), tx = gx - ix, ty = gy - iy;
    return [
      { ix, iy, w: (1 - tx) * (1 - ty) }, { ix: ix + 1, iy, w: tx * (1 - ty) },
      { ix, iy: iy + 1, w: (1 - tx) * ty }, { ix: ix + 1, iy: iy + 1, w: tx * ty },
    ];
  }
  // Convolution operator G: kernel image in FFT order on an (L,L) padded torus.
  // kfun(lx,ly) is the (continuous) kernel; ks = half support in grid units.
  function makeConvolver(g, kfun, ks, padded) {
    const need = padded ? Math.max(g.nx, g.ny) + 2 * ks + 1 : Math.max(g.nx, g.ny);
    const L = nextPow2(need);
    const kimg = kernelImage(L, g.h, (lx, ly) => (Math.abs(lx) > ks * g.h || Math.abs(ly) > ks * g.h) ? 0 : kfun(lx, ly));
    const re = Float64Array.from(kimg), im = new Float64Array(L * L);
    fft2d(re, im, L, L, false);
    const spectrum = re; // real (symmetric image)
    let smin = Infinity, nneg = 0;
    for (let i = 0; i < L * L; i++) { if (spectrum[i] < smin) smin = spectrum[i]; if (spectrum[i] < 0) nneg++; }
    const clipped = spectrum.map((v) => Math.max(v, 0));
    return {
      L, kimg, spectrum, clipped, smin, nneg,
      apply(field, useClipped = true) {
        const fr = new Float64Array(L * L), fi = new Float64Array(L * L);
        for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) fr[y * L + x] = field[y * g.nx + x];
        fft2d(fr, fi, L, L, false);
        const P = useClipped ? clipped : spectrum;
        for (let i = 0; i < L * L; i++) { fr[i] *= P[i]; fi[i] *= P[i]; }
        fft2d(fr, fi, L, L, true);
        const out = new Float64Array(g.nx * g.ny);
        for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) out[y * g.nx + x] = fr[y * L + x];
        return out;
      },
      padded() { // full padded torus content for display
        const fr = new Float64Array(L * L); return fr;
      },
    };
  }

  // Same operator from a TABULATED kernel image (side x side, odd side, zero lag at
  // the center), e.g. the kernel image treegp builds from the cleaned map.
  function makeConvolverFromImage(g, kimgCentered, side, padded) {
    const ks = side >> 1;
    const need = padded ? Math.max(g.nx, g.ny) + side : Math.max(g.nx, g.ny);
    const L = nextPow2(need);
    const kimg = new Float64Array(L * L);
    for (let dy = -ks; dy <= ks; dy++) for (let dx = -ks; dx <= ks; dx++)
      kimg[(((dy % L) + L) % L) * L + (((dx % L) + L) % L)] += kimgCentered[(dy + ks) * side + (dx + ks)];
    const re = Float64Array.from(kimg), im = new Float64Array(L * L);
    fft2d(re, im, L, L, false);
    const spectrum = re;
    let smin = Infinity, nneg = 0;
    for (let i = 0; i < L * L; i++) { if (spectrum[i] < smin) smin = spectrum[i]; if (spectrum[i] < 0) nneg++; }
    const clipped = spectrum.map((v) => Math.max(v, 0));
    return {
      L, kimg, spectrum, clipped, smin, nneg,
      apply(field, useClipped = true) {
        const fr = new Float64Array(L * L), fi = new Float64Array(L * L);
        for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) fr[y * L + x] = field[y * g.nx + x];
        fft2d(fr, fi, L, L, false);
        const P = useClipped ? clipped : spectrum;
        for (let i = 0; i < L * L; i++) { fr[i] *= P[i]; fi[i] *= P[i]; }
        fft2d(fr, fi, L, L, true);
        const out = new Float64Array(g.nx * g.ny);
        for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++) out[y * g.nx + x] = fr[y * L + x];
        return out;
      },
    };
  }

  // ---------------------------------------------------------------- linear algebra (small dense)
  function cholesky(A, n) {
    const Lm = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= Lm[i * n + k] * Lm[j * n + k];
      if (i === j) { if (s <= 0) return null; Lm[i * n + i] = Math.sqrt(s); }
      else Lm[i * n + j] = s / Lm[j * n + j];
    }
    return Lm;
  }
  function cholSolve(Lm, b, n) {
    const y = new Float64Array(n), x = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= Lm[i * n + k] * y[k]; y[i] = s / Lm[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= Lm[k * n + i] * x[k]; x[i] = s / Lm[i * n + i]; }
    return x;
  }
  // eigenvalues of a symmetric matrix by cyclic Jacobi rotations (n <= ~100)
  function jacobiEigenvalues(Ain, n) {
    const A = Float64Array.from(Ain);
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q]; if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
        }
      }
    }
    const ev = []; for (let i = 0; i < n; i++) ev.push(A[i * n + i]);
    return ev.sort((a, b) => a - b);
  }
  // preconditioned conjugate gradient; records residual norms and (optionally) iterates
  function pcg(matvec, b, minv, opts) {
    const n = b.length, maxit = opts.maxit || 200, rtol = opts.rtol || 1e-7, keep = !!opts.keepIterates;
    const x = opts.x0 ? Float64Array.from(opts.x0) : new Float64Array(n);
    const Ax0 = opts.x0 ? matvec(x) : new Float64Array(n);
    const r = Float64Array.from(b, (v, i) => v - Ax0[i]), z = new Float64Array(n), p = new Float64Array(n);
    const bn = Math.sqrt(b.reduce((s, v) => s + v * v, 0)) || 1;
    for (let i = 0; i < n; i++) { z[i] = r[i] * minv[i]; p[i] = z[i]; }
    let rz = 0; for (let i = 0; i < n; i++) rz += r[i] * z[i];
    const hist = [1], iterates = keep ? [Float64Array.from(x)] : null;
    let it = 0;
    for (it = 0; it < maxit; it++) {
      const Ap = matvec(p);
      let pAp = 0; for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
      const a = rz / pAp;
      for (let i = 0; i < n; i++) { x[i] += a * p[i]; r[i] -= a * Ap[i]; }
      const rn = Math.sqrt(r.reduce((s, v) => s + v * v, 0)) / bn;
      hist.push(rn); if (keep) iterates.push(Float64Array.from(x));
      if (rn < rtol) { it++; break; }
      for (let i = 0; i < n; i++) z[i] = r[i] * minv[i];
      let rz2 = 0; for (let i = 0; i < n; i++) rz2 += r[i] * z[i];
      const beta = rz2 / rz; rz = rz2;
      for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    }
    return { x, hist, iterates, iterations: it };
  }
  // steepest descent for comparison (same interface)
  function steepest(matvec, b, minv, opts) {
    const n = b.length, maxit = opts.maxit || 200, keep = !!opts.keepIterates;
    const x = opts.x0 ? Float64Array.from(opts.x0) : new Float64Array(n);
    const Ax0 = opts.x0 ? matvec(x) : new Float64Array(n);
    const r = Float64Array.from(b, (v, i) => v - Ax0[i]);
    const bn = Math.sqrt(b.reduce((s, v) => s + v * v, 0)) || 1;
    const hist = [1], iterates = keep ? [Float64Array.from(x)] : null;
    for (let it = 0; it < maxit; it++) {
      const z = r.map((v, i) => v * minv[i]);
      const Az = matvec(z);
      let rz = 0, zAz = 0; for (let i = 0; i < n; i++) { rz += r[i] * z[i]; zAz += z[i] * Az[i]; }
      const a = rz / zAz;
      for (let i = 0; i < n; i++) { x[i] += a * z[i]; r[i] -= a * Az[i]; }
      hist.push(Math.sqrt(r.reduce((s, v) => s + v * v, 0)) / bn); if (keep) iterates.push(Float64Array.from(x));
    }
    return { x, hist, iterates };
  }

  // ---------------------------------------------------------------- 1D GP (dense)
  function gp1d(xs, ys, sig, kfun, xgrid) {
    const n = xs.length, C = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) C[i * n + j] = kfun(Math.abs(xs[i] - xs[j])) + (i === j ? sig[i] * sig[i] : 0);
    const Lm = cholesky(C, n);
    const alpha = cholSolve(Lm, ys, n);
    const mean = new Float64Array(xgrid.length), sd = new Float64Array(xgrid.length);
    for (let m = 0; m < xgrid.length; m++) {
      const kx = new Float64Array(n); let mu = 0;
      for (let j = 0; j < n; j++) { kx[j] = kfun(Math.abs(xgrid[m] - xs[j])); mu += alpha[j] * kx[j]; }
      const v = cholSolve(Lm, kx, n); let q = 0; for (let j = 0; j < n; j++) q += kx[j] * v[j];
      mean[m] = mu; sd[m] = Math.sqrt(Math.max(kfun(0) - q, 0));
    }
    return { alpha, mean, sd };
  }

  // ---------------------------------------------------------------- helpers
  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
  const to2d = (arr, nx, ny) => { const out = []; for (let y = 0; y < ny; y++) out.push(Array.from(arr.subarray(y * nx, (y + 1) * nx))); return out; };
  const minmax = (arr) => { let lo = Infinity, hi = -Infinity; for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; } return [lo, hi]; };
  const absmax = (arr) => { let m = 0; for (const v of arr) { const a = Math.abs(v); if (a > m) m = a; } return m; };

  return {
    rng, fft1d, fft2d, dft2d, nextPow2, fftshift, vonKarman, vk, gaussian, anisoRadius, kernelImage, grf, autocorr, sampleImage,
    pairCount2pcf, hann, blackmanHarris, apodWindow, corr2power, power2corr, thresholdSpectrum, percentile,
    spread, readBack, bilinearWeights, blockReduce, crop2d, makeConvolver, makeConvolverFromImage, cholesky, cholSolve, jacobiEigenvalues, pcg, steepest, gp1d,
    linspace, to2d, minmax, absmax,
  };
})();
