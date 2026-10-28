"""
vergleich_ansys.py - Nachrechnung der ANSYS-Batches (Batch*_all_data.csv) mit dem Balkenmodell.

Aufruf:  python vergleich_ansys.py <Ordner mit Batch*_all_data.csv> [--esz]
Ausgabe: vergleich_ansys.csv (Kennwerte je Simulation), vergleich_ansys.png (Beispielkurven)

Erwartete Spalten: folder, displacement [mm], force [N], stress [MPa], stress_node_center [MPa],
                   length, width, thick, l2, l1, roll_radius, fric, emodul
Querkontraktion fest 0,2 (wie im ANSYS-Template).
"""
import sys
import glob
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from vierpunkt_fe import Aufbau, Vierpunkt

warnings.filterwarnings("ignore")
EDZ = "--esz" not in sys.argv


def rechne(arg):
    batch, folder, g = arg
    p = g.iloc[0]
    a = Aufbau(E=float(p.emodul), nu=0.2, ebener_dehnungszustand=EDZ, b=float(p.width),
               h=float(p.thick), Ls=float(p.l2), Li=float(p.l1), Rs=float(p.roll_radius),
               Rl=float(p.roll_radius), mu=float(p.fric), Lp=float(p.length))
    d = g["displacement [mm]"].to_numpy()
    e = Vierpunkt(a, 20, 20).rechne(d_max=float(d.max()), speichere_zustaende=False)
    F = g["force [N]"].to_numpy()
    i = int(np.argmax(F))
    F_mod = np.interp(d, e.d, e.F, right=np.nan)
    sel = np.isfinite(F_mod) & (d > 0)
    sel[i + 1:] = False
    sm = np.interp(d, e.d, e.sigma_mitte, right=np.nan)
    sx = np.interp(d, e.d, e.sigma, right=np.nan)
    return dict(
        batch=batch, folder=folder, h=a.h, b=a.b, Ls=a.Ls, Li=a.Li, R=a.Rs, mu=a.mu, E=a.E,
        w_Ls_Fmax=d[i] / a.Ls,
        dF_max_rel=np.nanmax(np.abs(F_mod[sel] - F[sel])) / F.max(),
        dF_Fmax_rel=np.nanmax(e.F) / F.max() - 1,
        d_sigma_mitte_rel=np.nanmax(np.abs(sm[sel] - g["stress_node_center [MPa]"].to_numpy()[sel]))
        / g["stress_node_center [MPa]"].max(),
        d_sigma_max_rel=np.nanmax(np.abs(sx[sel] - g["stress [MPa]"].to_numpy()[sel]))
        / g["stress [MPa]"].max(),
        hinweis=e.warnung,
        _kurven=(d, F, F_mod, g["stress [MPa]"].to_numpy(), sx),
    )


def main():
    ordner = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path(".")
    jobs = []
    for f in sorted(glob.glob(str(ordner / "Batch*_all_data.csv"))):
        batch = Path(f).stem.split("_")[0]
        for folder, g in pd.read_csv(f).groupby("folder", sort=False):
            jobs.append((batch, str(folder), g.sort_values("displacement [mm]")))
    if not jobs:
        sys.exit(f"Keine Batch*_all_data.csv in {ordner}")
    with Pool() as pool:
        res = pool.map(rechne, jobs)
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in res])
    df.to_csv("vergleich_ansys.csv", sep=";", decimal=",", index=False)
    print(f"{len(df)} Simulationen, {'ebener Dehnungszustand' if EDZ else 'ebener Spannungszustand'}")
    for c in ("dF_max_rel", "d_sigma_mitte_rel", "d_sigma_max_rel"):
        print(f"  {c:20s} Median {100 * df[c].median():5.1f} %   90%-Quantil {100 * df[c].quantile(0.9):5.1f} %")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    beisp = res[:: max(1, len(res) // 6)][:6]
    fig, ax = plt.subplots(len(beisp), 2, figsize=(10, 2.6 * len(beisp)))
    for j, r in enumerate(beisp):
        d, F, Fm, s, sm = r["_kurven"]
        ax[j, 0].plot(d, F, "k.", ms=3, label="ANSYS")
        ax[j, 0].plot(d, Fm, c="#2a78d6", label="Balken")
        ax[j, 1].plot(F, s, "k.", ms=3)
        ax[j, 1].plot(Fm, sm, c="#2a78d6")
        ax[j, 0].set_title(f"{r['batch']}/{r['folder']}: h={r['h']} b={r['b']} Ls={r['Ls']} R={r['R']} mu={r['mu']}",
                           fontsize=8)
        ax[j, 0].set_ylabel("F [N]")
        ax[j, 1].set_ylabel("σ max [MPa]")
    ax[0, 0].legend(fontsize=7)
    ax[-1, 0].set_xlabel("Traversenweg [mm]")
    ax[-1, 1].set_xlabel("F [N]")
    plt.tight_layout()
    plt.savefig("vergleich_ansys.png", dpi=90)


if __name__ == "__main__":
    main()
