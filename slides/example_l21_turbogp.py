"""L21 and TURBO-GP on a real LSSTCam visit: the worked example of the
treegp documentation (docs/treegp_gp_interp.rst) and of the TURBO-GP slides.

The visit (163k stars, astrometric residuals du, dv in mas with their
errors, positions u, v in degrees on the tangent plane) is read from the
compact npz shipped with the slides, or downloaded from the slides website
when the file is not found. 10 000 random stars are used to train each
Gaussian process; the others validate it through the E/B-mode correlation
functions of the residuals before and after correction.

Usage: python3 example_l21_turbogp.py [--data PATH] [--n-train N] [--no-l21]
                                      [--out FIGURE.png]
"""

import argparse
import os
import time
import urllib.request

import matplotlib
import numpy as np
import treegp
from treegp.utils import comp_eb_treecorr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data", "rubin_visit_2025052000348.npz")
URL = "https://pfleget.github.io/treegp/data/rubin_visit_2025052000348.npz"


def load_visit(path):
    if not os.path.exists(path):
        print("downloading", URL)
        urllib.request.urlretrieve(URL, path)
    with np.load(path) as z:
        d = {
            k: np.asarray(z[k], float)
            for k in ("u", "v", "du", "dv", "du_err", "dv_err")
        }
        d["visit"] = str(z["visit"])
    return d


def run_l21(X, y, y_err):
    """Leget et al. 2021: anisotropic von Karman kernel, hyperparameters
    fitted on the measured 2-point correlation function."""
    gp = treegp.GPInterpolation(
        kernel="15**2 * AnisotropicVonKarman(invLam=array([[1./0.8**2,0],[0,1./0.8**2]]))",
        optimizer="anisotropic",
        normalize=True,
        nbins=21,
        min_sep=None,
        max_sep=0.3,
        p0=[0.3, 0.0, 0.0],
    )
    gp.initialize(X, y, y_err=y_err)
    gp.solve()  # measures the 2-pcf (with a bootstrap for its errors) and fits the kernel
    return gp


def run_turbogp(X, y, y_err):
    """TURBO-GP: the measured 2D 2-point correlation function, cleaned, is
    the kernel; solved on a grid with the FFT and conjugate gradient."""
    gp = treegp.GPInterpolation(
        optimizer="empirical-2pcf",
        normalize=True,
        max_sep=0.3,  # half width of the correlation map (deg)
        pixel_size=20.0 / 3600.0,  # its pixel (deg)
        apodize=True,
        apod_window="hann",
        apod_radius=0.278,
        apod_anisotropy="auto",
        power_threshold=2.5,
        # solve_method="spectral" is the default for this optimizer
    )
    gp.initialize(X, y, y_err=y_err)
    gp.solve()  # measure, clean, tabulate the kernel, solve C alpha = y by CG
    return gp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=NPZ)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--no-l21", action="store_true", help="skip the (slow) L21 fit")
    ap.add_argument("--out", default="rubin_visit_l21_turbogp.png")
    args = ap.parse_args()

    D = load_visit(args.data)
    X = np.vstack([D["u"], D["v"]]).T
    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(X))
    train, val = perm[: args.n_train], perm[args.n_train :]

    pred = {}
    timing = {}
    for comp in ("du", "dv"):
        y, y_err = D[comp], D[comp + "_err"]
        t0 = time.time()
        gp = run_turbogp(X[train], y[train], y_err[train])
        pred["turbo", comp] = gp.predict(X[val])
        timing["turbo", comp] = time.time() - t0
        xi, xi_clean, lags, pixel = gp.return_empirical_2pcf()
        print(
            f"TURBO-GP {comp}: {timing['turbo', comp]:.1f} s, kernel {xi_clean.shape} at {pixel * 3600:.0f}\""
        )
        if not args.no_l21:
            t0 = time.time()
            gp = run_l21(X[train], y[train], y_err[train])
            pred["l21", comp] = gp.predict(X[val])
            timing["l21", comp] = time.time() - t0
            print(f"L21 {comp}: {timing['l21', comp]:.0f} s, kernel {gp.kernel}")

    # E/B-mode correlation functions of the validation stars
    kw = dict(rmin=5.0 / 3600.0, rmax=1.5, dlogr=0.1)
    u, v, du, dv = D["u"][val], D["v"][val], D["du"][val], D["dv"][val]
    eb = {"raw": comp_eb_treecorr(u, v, du, dv, **kw)}
    for m in ("turbo", "l21"):
        if (m, "du") in pred:
            eb[m] = comp_eb_treecorr(u, v, du - pred[m, "du"], dv - pred[m, "dv"], **kw)
    for m, (xie, xib, logr) in eb.items():
        print(
            f"{m:6s}: E-mode at the smallest separations {np.mean(xie[:4]):6.0f} mas^2, B-mode {np.mean(xib[:4]):5.0f}"
        )

    # figure: du over the field before / after each correction, and the E/B curves
    panels = [("raw", "du, raw", du)] + [
        (m, f"du after {lab}", du - pred[m, "du"])
        for m, lab in (("turbo", "TURBO-GP"), ("l21", "L21"))
        if (m, "du") in pred
    ]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.2 * (len(panels) + 1), 4.2))
    show = rng.choice(len(val), 6000, replace=False)
    for i, (ax, (m, title, vals)) in enumerate(zip(axes, panels)):
        sc = ax.scatter(
            u[show], v[show], c=vals[show], s=3, cmap=plt.cm.seismic, vmin=-60, vmax=60
        )
        ax.set_aspect("equal")
        ax.set_title(f"{title} (validation stars)", fontsize=10)
        ax.set_xlabel("u (deg)")
        if i == 0:
            ax.set_ylabel("v (deg)")
    fig.colorbar(sc, ax=list(axes[:-1]), label="mas", shrink=0.8)
    ax = axes[-1]
    for m, lab, c in (
        ("raw", "before", "k"),
        ("turbo", "TURBO-GP", "tab:blue"),
        ("l21", "L21", "tab:orange"),
    ):
        if m in eb:
            xie, xib, logr = eb[m]
            ax.plot(np.exp(logr), xie, "-", color=c, label=f"E-mode, {lab}")
            ax.plot(np.exp(logr), xib, ":", color=c, label=f"B-mode, {lab}")
    ax.set_xscale("log")
    ax.set_xlabel("separation (deg)")
    ax.set_ylabel(r"$\xi$ (mas$^2$)")
    ax.set_title(f"visit {D['visit']}, {len(val)} validation stars", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
