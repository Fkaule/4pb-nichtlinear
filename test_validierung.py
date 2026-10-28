"""
test_validierung.py - Verifikation des FE-Modells.

1. Konsistente Tangente (Vergleich mit zentralen Differenzen)
2. Linearer Grenzfall: Spannung und Durchbiegungen gegen Balkentheorie
3. Netzkonvergenz bei großer Durchbiegung
4. Unabhängige Referenzlösung: Elastica-Randwertproblem (dehnbarer
   Kirchhoff-Stab, gleiche Kontaktgeometrie, Shooting mit DOP853),
   ohne und mit Reibung

Aufruf:  python test_validierung.py
"""
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from vierpunkt_fe import Aufbau, Vierpunkt, berechne

OK = True


def pruefe(name, wert, grenze):
    global OK
    status = "OK " if wert <= grenze else "FEHLER"
    OK &= wert <= grenze
    print(f"  [{status}] {name}: {wert:.3e} (Grenze {grenze:.0e})")


# ---------------------------------------------------------------------------
def test_tangente():
    print("1) Tangente")
    a = Aufbau(E=70000, b=4, h=0.4, Ls=40, Li=20, Rs=2, Rl=3, mu=0.25)
    m = Vierpunkt(a, 8, 8)
    q0, s1, s2 = m.startzustand()
    rng = np.random.default_rng(0)
    z = np.concatenate([q0[m.free], [s1 + 0.1, s2 + 0.2, 1.3, 1.1, 0.2, -0.1]])
    ref = m._schlupf_ref(z)
    z[:m.nf] += rng.normal(0, 0.05, m.nf)
    for modus in ((1, -1), (0, 0), (0, 1)):
        R, J = m._system(z, 0.7, modus, ref)
        Jn = np.zeros_like(J)
        eps = 1e-7
        for i in range(z.size):
            zp, zm = z.copy(), z.copy()
            zp[i] += eps
            zm[i] -= eps
            Jn[:, i] = (m._system(zp, 0.7, modus, ref)[0] - m._system(zm, 0.7, modus, ref)[0]) / (2 * eps)
        pruefe(f"max |J - J_FD| / max |J|, Reibmodus {modus}", np.abs(J - Jn).max() / np.abs(J).max(), 1e-6)


def test_linear():
    print("2) Linearer Grenzfall (sehr kleine Last)")
    a = Aufbau(E=200000, b=5, h=2.0, Ls=40, Li=20, Rs=0.5, Rl=0.5)
    F = 1e-3 * a.F_aus_sigma_linear(100.0)
    e = berechne(a, F_ziel=float(F), n_innen=32, n_aussen=32)
    cl, cm = a.nachgiebigkeit_linear()
    pruefe("Spannung  rel. Abw.", abs(e.sigma[-1] / a.sigma_linear(F) - 1), 1e-4)
    pruefe("Lastrollenweg rel. Abw.", abs(e.d[-1] / (cl * F) - 1), 1e-3)
    pruefe("Mittendurchbiegung rel. Abw.", abs(e.w0[-1] / (cm * F) - 1), 1e-3)


# ---------------------------------------------------------------------------
class Elastica:
    """Unabhängige Referenz: Randwertproblem des dehnbaren Kirchhoff-Stabes."""

    def __init__(self, a: Aufbau, rho=(0.0, 0.0)):
        """rho = ft/Fn je Kontakt (Gleiten: +-mu, Haften: aus FE übernommen)."""
        self.a = a
        self.EA, self.EI = a.E_eff * a.A, a.E_eff * a.I
        self.rho = rho

    def _ode(self, s, Y, H, V):
        x, y, th, m = Y
        c, sn = np.cos(th), np.sin(th)
        eps = (H * c + V * sn) / self.EA
        return [(1 + eps) * c, (1 + eps) * sn, m / self.EI, -(1 + eps) * (c * V - sn * H)]

    def _int(self, Y0, s0, s1, H, V, dense=False):
        sol = solve_ivp(self._ode, (s0, s1), Y0, args=(H, V), method="DOP853",
                        rtol=1e-12, atol=1e-13, dense_output=dense)
        return sol

    def residuum(self, u, d, voll=False):
        a = self.a
        y0, M0, H0, s1, Fn1, s2, Fn2 = u
        h2 = a.h / 2
        solA = self._int([0.0, y0, 0.0, M0], 0.0, s1, H0, 0.0, dense=voll)
        x1, y1, th1, m1 = solA.y[:, -1]
        n1 = np.array([-np.sin(th1), np.cos(th1)]); t1 = np.array([np.cos(th1), np.sin(th1)])
        P1 = -Fn1 * n1 + self.rho[0] * Fn1 * t1
        C1 = -h2 * self.rho[0] * Fn1
        HB, VB = np.array([H0, 0.0]) - P1
        solB = self._int([x1, y1, th1, m1 - C1], s1, s2, HB, VB, dense=voll)
        x2, y2, th2, m2 = solB.y[:, -1]
        n2 = np.array([-np.sin(th2), np.cos(th2)]); t2 = np.array([np.cos(th2), np.sin(th2)])
        P2 = Fn2 * n2 + self.rho[1] * Fn2 * t2
        C2 = h2 * self.rho[1] * Fn2
        Cl = np.array([a.Li / 2, h2 + a.Rl - d])
        Cs = np.array([a.Ls / 2, -h2 - a.Rs])
        Fs = max(abs(Fn1), 1e-12)
        res = np.concatenate([
            (np.array([x1, y1]) + (h2 + a.Rl) * n1 - Cl) / a.h,
            (np.array([x2, y2]) - (h2 + a.Rs) * n2 - Cs) / a.h,
            (np.array([HB, VB]) - P2) / Fs,
            [(m2 - C2) / (Fs * a.h)],
        ])
        if not voll:
            return res
        F = 2 * (Fn1 * np.cos(th1) - self.rho[0] * Fn1 * np.sin(th1))
        smax = 0.0
        for sol, (H, V) in ((solA, (H0, 0.0)), (solB, (HB, VB))):
            ss = np.linspace(sol.t[0], sol.t[-1], 4001)
            Y = sol.sol(ss)
            N = H * np.cos(Y[2]) + V * np.sin(Y[2])
            smax = max(smax, np.max(N / a.A + np.abs(Y[3]) / a.W))
        return F, smax


def fe_startwert(m: Vierpunkt, z):
    q = m._q_aus_z(z)
    s1, s2, Fn1, Fn2, ft1, ft2 = z[m.nf:]
    _, dat = m.intern(q, s1, s2, tangente=False)
    M0 = m.EI * (q[5] - q[2]) / dat["L0"][0]
    return np.array([q[1], M0, dat["N"][0], s1, Fn1, s2, Fn2])


def test_netz_und_elastica():
    for mu in (0.0, 0.3):
        print(f"3/4) Große Durchbiegung, mu = {mu}: Netzkonvergenz und Elastica-Referenz")
        a = Aufbau(E=70000, b=4, h=0.4, Ls=40, Li=20, Rs=2.5, Rl=2.5, mu=mu)
        d = 4.5  # Lastrollenweg [mm]
        werte = []
        for n in (8, 16, 32, 64):
            m = Vierpunkt(a, n, n)
            t = time.time()
            e = m.rechne(d_max=d)
            dt = time.time() - t
            werte.append((n, e.F[-1], e.sigma[-1], e.w0[-1], dt, m, e.zustaende[-1]))
            print(f"     n={n:3d}+{n:<3d}  F={e.F[-1]:.6f} N  sigma={e.sigma[-1]:.4f} MPa  "
                  f"w0={e.w0[-1]:.4f} mm  ({dt:.2f} s, {len(e.F)} Schritte)")
        zl = werte[-1][6]
        Fn1, Fn2, ft1, ft2 = zl[werte[-1][5].nf + 2:]
        print(f"     Reibzustand Ende: {e.modus[-1]}, ft/Fn = {ft1 / Fn1:+.3f}, {ft2 / Fn2:+.3f}")
        el = Elastica(a, rho=(ft1 / Fn1, ft2 / Fn2))
        n, F_fe, s_fe, _, _, m, z = werte[-1]
        u = fsolve(el.residuum, fe_startwert(m, z), args=(d,), xtol=1e-13)
        pruefe("Elastica-Residuum", np.abs(el.residuum(u, d)).max(), 1e-8)
        F_el, s_el = el.residuum(u, d, voll=True)
        print(f"     Elastica:   F={F_el:.6f} N  sigma={s_el:.4f} MPa")
        for n, F_fe, s_fe, *_ in werte:
            print(f"     n={n:3d}: Abw. F {abs(F_fe / F_el - 1):.2e}, sigma {abs(s_fe / s_el - 1):.2e}")
        n, F_fe, s_fe, *_ = werte[2]   # Standardnetz 24..32 Elemente je Abschnitt
        pruefe("rel. Abw. F (32+32 El.)", abs(F_fe / F_el - 1), 1e-3)
        pruefe("rel. Abw. sigma (32+32 El.)", abs(s_fe / s_el - 1), 1e-3)


if __name__ == "__main__":
    test_tangente()
    test_linear()
    test_netz_und_elastica()
    print("\nALLE TESTS BESTANDEN" if OK else "\nTESTS FEHLGESCHLAGEN")
