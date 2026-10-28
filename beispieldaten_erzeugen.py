"""
Erzeugt synthetische Beispieldaten (dünnes Glas, 4-Punkt-Biegung nach
EN 843-1-Geometrie B mit dünnen Proben) zum Ausprobieren der App:

* 20 Kraft-Weg-Kurven (Traversenweg) im testXpert-ähnlichen Format
  (Kopfzeilen, Semikolon, Dezimalkomma), inkl. Maschinennachgiebigkeit,
  Anlaufbereich und Rauschen
* probentabelle.csv mit gemessenen b, h je Probe

Wahre Festigkeit (nichtlinear) ~ Weibull(m = 8, sigma_0 = 520 MPa).
"""
from pathlib import Path
import numpy as np
from vierpunkt_fe import Aufbau, berechne

ZIEL = Path(__file__).parent / "beispieldaten"
E, NU = 72000.0, 0.22
LS, LI, R = 40.0, 20.0, 2.5
C_MASCHINE = 0.004      # mm/N
rng = np.random.default_rng(2026)


def main():
    ZIEL.mkdir(exist_ok=True)
    zeilen = ["ID;b [mm];h [mm]"]
    for i in range(1, 21):
        pid = f"P{i:02d}"
        b = 4.0 + rng.normal(0, 0.01)
        h = 0.40 + rng.normal(0, 0.006)
        a = Aufbau(E=E, nu=NU, ebener_dehnungszustand=True, b=b, h=h, Ls=LS, Li=LI, Rs=R, Rl=R, mu=0.2, Lp=45.0)
        sig_f = 520.0 * rng.weibull(8.0)
        e = berechne(a, F_ziel=1e4)                        # bis Kraftmaximum
        sig_f = min(sig_f, 0.97 * e.sigma[-1])
        F_f = float(np.interp(sig_f, e.sigma, e.F))
        e = berechne(a, F_ziel=F_f)
        F = np.linspace(0, F_f, 400)
        d = np.interp(F, e.F, e.d) + C_MASCHINE * F
        # Anlaufbereich: 0,03 mm Weg ohne Kraft
        d_anl = np.linspace(0, 0.03, 15)
        d = np.concatenate([d_anl, d[1:] + 0.03])
        F = np.concatenate([np.zeros(15), F[1:]])
        F = F + rng.normal(0, 0.002 * F_f, F.size)
        F[:15] = np.abs(F[:15]) * 0.1
        # nach Bruch: Kraftabfall
        d = np.concatenate([d, d[-1] + [0.002, 0.004]])
        F = np.concatenate([F, [0.3, 0.05]])
        with open(ZIEL / f"{pid}.csv", "w", encoding="cp1252") as f:
            f.write(f"Prüfung;4-Punkt-Biegung\nProbe;{pid}\nPrüfgeschwindigkeit;0,5 mm/min\n\n")
            f.write("Standardweg;Standardkraft\nmm;N\n")
            for dd, FF in zip(d, F):
                f.write(f"{dd:.5f};{FF:.4f}\n".replace(".", ","))
        zeilen.append(f"{pid};{b:.3f};{h:.4f}".replace(".", ","))
        print(pid, f"h={h:.4f} F={F_f:.3f} N sigma_wahr={sig_f:.1f} lin={a.sigma_linear(F_f):.1f}")
    (ZIEL / "probentabelle.csv").write_text("\n".join(zeilen) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
