"""Build data/visit.js for the TURBO-GP slides from a simulated Rubin visit
(by default data/rubin_visit_2025052000348.npz, see --data) with treegp.

Runs the TURBO-GP solver (empirical 2-pcf kernel + spectral solve) with the
RTN-129 configuration on 10k training sources per component and exports the
intermediate products the slides show: the measured / cleaned correlation
maps, the kernel image and its padded spectrum, a 0.5 deg patch of the real
grid with its stars and weights, the lowest eigenvalues of the dense and
grid kernels on real stars, the conjugate-gradient history with snapshots of
the mean field, the mean field itself, predictions on the validation sample
and E/B-mode curves. Also fits the L21 anisotropic von Karman kernel to the
same measured map for the comparison slide (slow: ~25 min per component
because of the bootstrap; the result is cached in data/cache_l21.pkl, delete
that file to refit).

Usage: python3 make_data.py [--data PATH.npz|PATH.pkl] [--no-l21] [--write-npz]
"""

import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
import scipy.fft as sfft
import treegp
from scipy.sparse.linalg import LinearOperator, cg
from treegp.utils import comp_eb_treecorr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "visit.js")
CACHE_L21 = os.path.join(HERE, "data", "cache_l21.pkl")
NPZ = os.path.join(HERE, "data", "rubin_visit_2025052000348.npz")
PKL = os.path.join(HERE, "..", "example", "2025052000348.pkl")
DATA_KEYS = ("u", "v", "du", "dv", "du_err", "dv_err")

N_TRAIN = 10000
N_DISPLAY = 6000
SEED = 1234
PIXEL = 20.0 / 3600.0  # deg
MAX_SEP = 0.3
APOD_RADIUS = 0.278
WINDOW = "hann"
THRESHOLD = 2.5
MEAN_DS = 6  # downsampling of the mean-field image for display
CG_DS = 10  # downsampling of the mean-field snapshots along the CG iterations
CG_SNAPSHOTS = [1, 3, 10, 30]  # plus the final iterate
PATCH_DEG = 0.5  # side of the real-grid patch shown on the grid / matvec slides
N_EIG = 2000  # stars used for the eigenvalue check (the closest to the field centre: real density)
N_EIG_KEEP = 40  # lowest eigenvalues exported
SPEC_CUT = 96  # columns of the raw padded spectrum exported along k_v = 0


def r4(x):
    """Round floats to 4 significant digits for a compact JS file."""
    if isinstance(x, (list, tuple)):
        return [r4(v) for v in x]
    if isinstance(x, np.ndarray):
        return r4(x.tolist())
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        return float(f"{x:.4g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def fit_l21(X, y, yerr, train, val, name):
    """L21 anisotropic von Karman fit on the same training sample."""
    print(f"component {name}: L21 anisotropic fit (slow, bootstrap)")
    gpl = treegp.GPInterpolation(
        kernel="15**2 * AnisotropicVonKarman(invLam=array([[1./0.8**2,0],[0,1./0.8**2]]))",
        optimizer="anisotropic",
        normalize=True,
        nbins=21,
        min_sep=None,
        max_sep=MAX_SEP,
        p0=[0.3, 0.0, 0.0],
    )
    gpl.initialize(X[train], y[train], y_err=yerr[train])
    t0 = time.time()
    gpl.solve()
    t_fit = time.time() - t0
    o2 = gpl._optimizer
    n2 = int(np.sqrt(len(o2._2pcf)))
    vk = dict(
        n=n2,
        lag=r4((np.arange(n2) - n2 // 2) * (2 * MAX_SEP / n2)),
        measured=r4(np.asarray(o2._2pcf).reshape(n2, n2)),
        fit=r4(np.asarray(o2._2pcf_fit).reshape(n2, n2)),
        kernel=str(gpl.kernel),
        t_fit=r4(t_fit),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        pred = gpl.predict(X[val])
        t_pred = time.time() - t0
    return dict(vk=vk, vk_info=dict(t_predict=r4(t_pred)), pred_val=pred)


def grid_kernel_matrix(eng, X):
    """Dense K = W G W^T of the TURBO-GP grid kernel between the points X,
    from the clipped kernel image on the padded grid (exact node lags)."""
    w = eng._spread_matrix(X)
    w = w.tocsr()
    n = len(X)
    cols = w.indices.reshape(n, 4)
    vals = w.data.reshape(n, 4)
    ix = cols % eng._nx
    iy = cols // eng._nx
    img = sfft.irfft2(eng._spectrum, s=(eng._ly, eng._lx))
    K = np.zeros((n, n))
    for a in range(4):
        for b in range(4):
            dx = (ix[:, a][:, None] - ix[:, b][None, :]) % eng._lx
            dy = (iy[:, a][:, None] - iy[:, b][None, :]) % eng._ly
            K += vals[:, a][:, None] * vals[:, b][None, :] * img[dy, dx]
    return K


def cg_history(eng, X, y, yerr, precond=True):
    """Re-run the engine's conjugate gradient, recording the relative
    residual at every iteration and the iterates at a few of them."""
    w = eng._w_train
    noise = yerr**2
    n = len(y)

    def matvec(v):
        field = (w.T @ v).reshape(eng._ny, eng._nx)
        return w @ eng._convolve(field).ravel() + noise * v

    op = LinearOperator((n, n), matvec=matvec, dtype=float)
    M = None
    if precond:
        d = 1.0 / (eng.xi0_eff + noise)
        M = LinearOperator((n, n), matvec=lambda v: d * v, dtype=float)
    yn = np.linalg.norm(y)
    hist = [1.0]
    snaps = {}

    def callback(xk):
        k = len(hist)
        hist.append(float(np.linalg.norm(y - matvec(xk)) / yn))
        if k in CG_SNAPSHOTS:
            snaps[k] = xk.copy()

    t0 = time.time()
    alpha, info = cg(
        op,
        y,
        rtol=eng.cg_rtol,
        atol=0.0,
        maxiter=eng.cg_maxiter,
        M=M,
        callback=callback,
    )
    t = time.time() - t0
    snaps[len(hist) - 1] = alpha.copy()
    return hist, snaps, t


def load_visit(path):
    """Read the simulated visit from the original pickle or from the compact
    npz shipped with the slides (float32: u, v in deg; du, dv, du_err,
    dv_err in mas; visit, band)."""
    if path.endswith(".npz"):
        with np.load(path) as z:
            D = {k: z[k] for k in z.files}
    else:
        with open(path, "rb") as f:
            D = pickle.load(f)
    out = {k: np.asarray(D[k], float) for k in DATA_KEYS}
    out["visit"] = str(D["visit"])
    out["band"] = str(D["band"])
    return out


def write_npz(D, path):
    """Write the compact float32 version of the visit used by the slides and
    by the treegp documentation example."""
    np.savez_compressed(
        path,
        visit=D["visit"],
        band=D["band"],
        **{k: np.float32(D[k]) for k in DATA_KEYS},
    )
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--data",
        default=NPZ if os.path.exists(NPZ) else PKL,
        help="simulated visit: the compact .npz or the original .pkl "
        "(u, v in deg; du, dv, du_err, dv_err in mas)",
    )
    ap.add_argument("--no-l21", action="store_true", help="skip the L21 comparison fit")
    ap.add_argument(
        "--write-npz",
        action="store_true",
        help=f"write the compact float32 copy of the visit to {NPZ} and exit",
    )
    args = ap.parse_args()

    D = load_visit(args.data)
    if args.write_npz:
        write_npz(D, NPZ)
        return
    u, v, du, dv, du_err, dv_err = (D[k] for k in DATA_KEYS)
    n_all = len(u)
    X = np.vstack([u, v]).T

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_all)
    train = perm[:N_TRAIN]
    val = perm[N_TRAIN:]
    disp = rng.choice(val, N_DISPLAY, replace=False)
    # the N_EIG training stars closest to the field centre (same density as the real solve)
    dc = np.hypot(*(X[train] - X[train].mean(axis=0)).T)
    sub_eig = np.argsort(dc)[:N_EIG]

    out = {
        "meta": dict(
            visit=D["visit"],
            band=D["band"],
            n_all=n_all,
            n_train=N_TRAIN,
            n_val=len(val),
            pixel_deg=PIXEL,
            max_sep=MAX_SEP,
            apod_radius=APOD_RADIUS,
            window=WINDOW,
            threshold=THRESHOLD,
            patch_deg=PATCH_DEG,
            n_eig=N_EIG,
        ),
        "comp": {},
    }

    cache = {}
    if os.path.exists(CACHE_L21):
        with open(CACHE_L21, "rb") as f:
            cache = pickle.load(f)
        print("L21 fit results read from", CACHE_L21)

    gp_pred = {}
    for name, y, yerr in (("u", du, du_err), ("v", dv, dv_err)):
        print(f"component {name}: TURBO-GP solve on {N_TRAIN} sources")
        gp = treegp.GPInterpolation(
            optimizer="empirical-2pcf",
            normalize=True,
            max_sep=MAX_SEP,
            pixel_size=PIXEL,
            power_threshold=THRESHOLD,
            apodize=True,
            apod_window=WINDOW,
            apod_radius=APOD_RADIUS,
            apod_anisotropy="auto",
            apod_g_scale=1.0,
            white_noise=0.0,
        )
        gp.initialize(X[train], y[train], y_err=yerr[train])
        t0 = time.time()
        gp.solve()
        t_solve = time.time() - t0
        t0 = time.time()
        pred_val = gp.predict(X[val])
        t_pred = time.time() - t0
        gp_pred[name] = pred_val
        opt = gp._optimizer
        eng = gp._engine
        npix = opt.npix
        lag = (np.arange(npix) - npix // 2) * opt.pixel_size
        # the centered values the engine actually solves for
        y_eng = gp._y - gp._mean - gp._spatial_average
        y_err_eng = np.asarray(gp._y_err, float)

        # raw (unclipped) padded spectrum of the kernel image
        ks = eng._kernel_image.shape[0]
        buf = np.zeros((eng._ly, eng._lx))
        buf[:ks, :ks] = eng._kernel_image
        buf = np.roll(buf, (-(ks // 2), -(ks // 2)), axis=(0, 1))
        spec_raw = sfft.rfft2(buf).real
        nr, nc = 64, 64
        rows = np.r_[np.arange(0, nr), np.arange(eng._ly - nr, eng._ly)]
        crop = spec_raw[np.ix_(rows, np.arange(nc))]
        crop = np.roll(crop, nr, axis=0)  # rows now -64..63
        neg_frac = float(np.mean(spec_raw < 0))
        neg_power = float(-spec_raw[spec_raw < 0].sum() / spec_raw[spec_raw > 0].sum())

        # mean field, downsampled
        mf = eng._mean_field
        mf_ds = mf[::MEAN_DS, ::MEAN_DS]

        # a PATCH_DEG square of the real grid around the field centre, with
        # a margin of one kernel half support so that every star contributing
        # to the field inside the window is included (the slides convolve the
        # full box and display the central window)
        ksh = ks // 2
        cx = 0.5 * (u[train].min() + u[train].max())
        cy = 0.5 * (v[train].min() + v[train].max())
        pn = int(round(PATCH_DEG / eng.h))
        nb = pn + 2 * ksh
        ix0 = int(round((cx - eng._x0) / eng.h)) - nb // 2
        iy0 = int(round((cy - eng._y0) / eng.h)) - nb // 2
        px0 = eng._x0 + ix0 * eng.h
        py0 = eng._y0 + iy0 * eng.h
        inp = (
            (X[train, 0] >= px0)
            & (X[train, 0] < px0 + (nb - 1) * eng.h)
            & (X[train, 1] >= py0)
            & (X[train, 1] < py0 + (nb - 1) * eng.h)
        )
        patch = dict(
            x0=r4(px0),
            y0=r4(py0),
            h=r4(eng.h),
            nx=nb,
            ny=nb,
            win=pn,
            off=ksh,
            n_stars=int(inp.sum()),
            mean=r4(mf[iy0 + ksh : iy0 + ksh + pn, ix0 + ksh : ix0 + ksh + pn]),
        )
        print(
            f"  patch: {inp.sum()} training stars in the {nb}x{nb} box, "
            f"{PATCH_DEG} deg window of {pn}x{pn} nodes"
        )

        # lowest eigenvalues on real stars: tabulated kernel, + noise, grid kernel
        print(f"  eigenvalues on {N_EIG} stars")
        Xs = X[train][sub_eig]
        kern = treegp.EmpiricalCorrelationKernel(
            lag, lag, opt._xi_clean, clip_eigenvalues=False
        )
        K_tab = kern(Xs)
        ev_tab = np.linalg.eigvalsh(K_tab)
        ev_tabD = np.linalg.eigvalsh(K_tab + np.diag(y_err_eng[sub_eig] ** 2))
        K_grid = grid_kernel_matrix(eng, Xs)
        ev_grid = np.linalg.eigvalsh(K_grid)
        eig = dict(
            tab=r4(ev_tab[:N_EIG_KEEP]),
            tabD=r4(ev_tabD[:N_EIG_KEEP]),
            grid=r4(ev_grid[:N_EIG_KEEP]),
            n_neg_tab=int(np.sum(ev_tab < 0)),
            n_neg_tabD=int(np.sum(ev_tabD < 0)),
            n_neg_grid=int(np.sum(ev_grid < -1e-6 * ev_grid.max())),
            max_tab=r4(float(ev_tab.max())),
            min_sig=r4(float(y_err_eng[sub_eig].min())),
            radius=r4(float(dc[sub_eig].max())),
        )
        print(
            f"  tabulated kernel: {eig['n_neg_tab']} negative eigenvalues (min {ev_tab[0]:.1f}), "
            f"with noise {eig['n_neg_tabD']} (min {ev_tabD[0]:.1f}); grid kernel min {ev_grid[0]:.2e}"
        )

        # conjugate gradient history, with and without the preconditioner
        print("  CG history")
        hist_pre, snaps, t_cg = cg_history(
            eng, X[train], y_eng, y_err_eng, precond=True
        )
        hist_nopre, _, _ = cg_history(eng, X[train], y_eng, y_err_eng, precond=False)
        snap_out = []
        for k in sorted(snaps):
            field = (eng._w_train.T @ snaps[k]).reshape(eng._ny, eng._nx)
            fk = eng._convolve(field)[::CG_DS, ::CG_DS]
            snap_out.append(dict(k=int(k), img=r4(fk)))
        cg_out = dict(
            hist_pre=r4(hist_pre),
            hist_nopre=r4(hist_nopre),
            n_pre=len(hist_pre) - 1,
            n_nopre=len(hist_nopre) - 1,
            snapshots=snap_out,
            ds=CG_DS,
            nx=int(fk.shape[1]),
            ny=int(fk.shape[0]),
            t=r4(t_cg),
        )
        print(
            f"  CG: {cg_out['n_pre']} iterations with Jacobi, {cg_out['n_nopre']} without"
        )

        # L21 comparison (cached)
        l21 = cache.get(name)
        if l21 is None and not args.no_l21:
            try:
                l21 = fit_l21(X, y, yerr, train, val, name)
                cache[name] = l21
                with open(CACHE_L21, "wb") as f:
                    pickle.dump(cache, f)
            except Exception as exc:  # the comparison is optional
                print("  L21 fit failed:", exc)
        if l21 is not None and "pred_val" in l21:
            gp_pred[name + "_l21"] = np.asarray(l21["pred_val"])

        out["comp"][name] = dict(
            npix=int(npix),
            lag=r4(lag),
            raw=r4(opt._xi),
            clean=r4(opt._xi_clean),
            pass1=r4(opt._xi_clean_pass1),
            g_meas=r4(list(opt._apod_g_measured)),
            g_app=r4(list(opt._apod_g_applied)),
            xi0_raw=r4(float(opt._xi[npix // 2, npix // 2])),
            xi0_clean=r4(float(opt._xi_clean[npix // 2, npix // 2])),
            xi0_eff=r4(float(eng.xi0_eff)),
            spectrum_min=r4(float(eng.spectrum_min)),
            neg_frac=r4(neg_frac),
            neg_power=r4(neg_power),
            n_iter=int(eng.n_iterations),
            rtol=eng.cg_rtol,
            maxiter=eng.cg_maxiter,
            grid=dict(
                h=r4(eng.h),
                x0=r4(eng._x0),
                y0=r4(eng._y0),
                nx=int(eng._nx),
                ny=int(eng._ny),
                lx=int(eng._lx),
                ly=int(eng._ly),
                kernel_side=int(ks),
            ),
            kimg=r4(eng._kernel_image),
            spec_crop=dict(
                rows=int(2 * nr),
                cols=int(nc),
                vals=r4(np.sign(crop) * np.log10(1 + np.abs(crop))),
            ),
            spec_cut=r4(spec_raw[0, :SPEC_CUT]),
            spec_peak=r4(float(spec_raw.max())),
            mean=dict(
                nx=int(mf_ds.shape[1]),
                ny=int(mf_ds.shape[0]),
                x0=r4(eng._x0),
                y0=r4(eng._y0),
                h=r4(eng.h * MEAN_DS),
                img=r4(mf_ds),
            ),
            patch=patch,
            train=dict(y=r4(y_eng), sig=r4(y_err_eng), alpha=r4(eng._alpha)),
            eig=eig,
            cg=cg_out,
            t_solve=r4(t_solve),
            t_predict=r4(t_pred),
            noise_median=r4(float(np.median(yerr[train]))),
            vk=l21["vk"] if l21 else None,
            vk_info=l21["vk_info"] if l21 else None,
        )
        print(
            f"  xi0 raw {opt._xi[npix // 2, npix // 2]:.1f}, clean {opt._xi_clean[npix // 2, npix // 2]:.1f}, "
            f"eff {eng.xi0_eff:.1f}; spectrum min {eng.spectrum_min:.3g}, neg frac {neg_frac:.3f}; "
            f"CG iterations {eng.n_iterations}; solve {t_solve:.1f} s, predict {t_pred:.2f} s; "
            f"g_meas {opt._apod_g_measured}"
        )

    # all training stars (positions shared by both components)
    out["train"] = dict(u=r4(u[train]), v=r4(v[train]))
    # display subsample: positions, residuals and the GP predictions
    out["disp"] = dict(u=r4(u[disp]), v=r4(v[disp]), du=r4(du[disp]), dv=r4(dv[disp]))
    idx_in_val = {int(i): k for k, i in enumerate(val)}
    sel = np.array([idx_in_val[int(i)] for i in disp])
    out["disp"]["gpu"] = r4(gp_pred["u"][sel])
    out["disp"]["gpv"] = r4(gp_pred["v"][sel])
    if "u_l21" in gp_pred and "v_l21" in gp_pred:
        cache["disp_l21"] = dict(
            gpu=r4(gp_pred["u_l21"][sel]), gpv=r4(gp_pred["v_l21"][sel])
        )
    if "disp_l21" in cache:
        out["disp"]["gpu_l21"] = cache["disp_l21"]["gpu"]
        out["disp"]["gpv_l21"] = cache["disp_l21"]["gpv"]

    # E/B curves on the validation sample, before and after correction
    print("E/B modes on the validation sample")
    kw = dict(rmin=5.0 / 3600.0, rmax=1.5, dlogr=0.1)
    xie0, xib0, logr = comp_eb_treecorr(u[val], v[val], du[val], dv[val], **kw)
    xie1, xib1, _ = comp_eb_treecorr(
        u[val], v[val], du[val] - gp_pred["u"], dv[val] - gp_pred["v"], **kw
    )
    eb = dict(
        r=r4(np.exp(logr)),
        E_before=r4(xie0),
        B_before=r4(xib0),
        E_after=r4(xie1),
        B_after=r4(xib1),
    )
    if "u_l21" in gp_pred and "v_l21" in gp_pred:
        xie2, xib2, _ = comp_eb_treecorr(
            u[val], v[val], du[val] - gp_pred["u_l21"], dv[val] - gp_pred["v_l21"], **kw
        )
        cache["eb_l21"] = dict(E=r4(xie2), B=r4(xib2))
        # the derived L21 products are all the slides need: drop the big arrays
        for name in ("u", "v"):
            cache[name].pop("pred_val", None)
        with open(CACHE_L21, "wb") as f:
            pickle.dump(cache, f)
    if "eb_l21" in cache:
        eb["E_after_l21"] = cache["eb_l21"]["E"]
        eb["B_after_l21"] = cache["eb_l21"]["B"]
    out["eb"] = eb
    rms0 = np.sqrt(np.mean(du[val] ** 2 + dv[val] ** 2))
    rms1 = np.sqrt(
        np.mean((du[val] - gp_pred["u"]) ** 2 + (dv[val] - gp_pred["v"]) ** 2)
    )
    out["meta"]["rms_before"] = r4(rms0)
    out["meta"]["rms_after"] = r4(rms1)
    print(
        f"  E(0) before {xie0[0]:.0f} -> after {xie1[0]:.0f} mas^2; rms {rms0:.1f} -> {rms1:.1f} mas"
    )

    # benchmark numbers measured on this visit (treegp/CLAUDE.md, per component, one core)
    out["bench"] = dict(
        dense=[dict(n=10000, t=85, mem=3.7), dict(n=15000, t=140, mem=6.6)],
        spectral=[dict(n=10000, t=4.5, mem=0.55), dict(n=15000, t=5, mem=0.6)],
        dp2=dict(
            err_l21=5,
            err_turbo=1,
            wall_l21_days=979,
            wall_turbo_days=15,
            mem_l21=7,
            mem_turbo=1.5,
        ),
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("window.VISIT = ")
        json.dump(out, f, separators=(",", ":"))
        f.write(";\n")
    print("wrote", OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
