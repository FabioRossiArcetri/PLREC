"""Score the 4x256 per-wavelength MLPs: wrapped error per wavelength and the fused error."""
import os

import numpy as np

import ml_reconstruct as ml

DATA = os.path.join("data", "ml", "pl19_1200")
CACHE = os.path.join(DATA, "cache_big")

ds = ml.LanternDataset.load(os.path.join(DATA, "spectral_ao120_21wl.npz"))
wls = np.asarray(ds.wavelengths_nm, float)
S = np.asarray(ds.signals)                      # (n, W, 19)
P = np.concatenate([np.asarray(ds.train[1]), np.asarray(ds.val[1])], 0)
n_tr = len(ds.train[0])
n_wl, M = len(wls), P.shape[1]
SPAN = float(np.abs(P).max())
print(f"{n_wl} wavelengths, {n_tr} fit / {len(P) - n_tr} held out, span +-{SPAN:.0f} nm")

S_va, P_va = S[n_tr:], P[n_tr:]
ghat = np.zeros((len(P_va), M, n_wl))
sig = np.zeros((n_wl, M))
for w, lam in enumerate(wls):
    xs = ml.Standardizer().fit(S[:n_tr, w, :])
    ys = ml.PhasorCodec(lam)
    params, layers, extra = ml.load_mlp(os.path.join(CACHE, f"w{w}-mlp-256x4.npz"))
    g = ys.inverse(ml.mlp_forward(params, xs.transform(S_va[:, w, :])))
    ghat[:, :, w] = g
    sig[w] = np.asarray(ml.rmse_nm(g, P_va, lam))
    print(f"  w{w:2d} {lam:6.1f} nm: wrapped {sig[w].mean():6.2f} nm, "
          f"best epoch {int(extra['best_epoch'])}")

FITS = dict(ghat_va=ghat, conf_va=np.ones_like(ghat), sig_va=sig, sig_tr=sig,
            periods=wls, wl_use=np.arange(n_wl))
P_hat, Q = ml.unwrap_all(FITS, SPAN)
err = np.asarray(ml.rmse_nm(P_hat, P_va, 2 * SPAN))
bad = np.abs(np.asarray(ml.wrap(P_hat - P_va, 2 * SPAN))) > 0.25 * 1200.0
print(f"\nfused: mean {err.mean():.2f} nm, median {np.median(err):.2f}, "
      f"worst {err.max():.1f}; wrong fringes {100 * bad.mean():.1f}%")
print(f"per-wavelength wrapped mean: {sig.mean():.2f} nm")
np.savez(os.path.join(DATA, "mlp_big_spectral_scores.npz"), wls=wls, sig=sig,
         err=err, bad=bad, span=SPAN)
