"""
vierpunkt_fe.py
===============
Geometrisch nichtlineares FE-Modell der 4-Punkt-Biegung mit Rollenkontakt.

Modell
------
* Halbmodell (Symmetrie in Probenmitte: x = 0, Rotation = 0, vertikal frei).
* Korotationale 2D-Balkenelemente (Euler-Bernoulli lokal, Crisfield/Battini),
  beliebig große Verschiebungen und Rotationen, kleine Dehnungen,
  linear-elastisches Material.
* Kontakt Probe/Rolle exakt geometrisch: die Rolle (Radius R) berührt die
  Probenoberfläche tangential, d. h. der Rollenmittelpunkt liegt im Abstand
  R + h/2 auf der Normalen der Balkenachse im Kontaktpunkt.
* Die Kontaktpunkte wandern über die Probe. Ihre materiellen Koordinaten
  s1 (Lastrolle) und s2 (Stützrolle) sind zusätzliche Unbekannte
  (ALE-Ansatz: die Elementlängen der Abschnitte [0, s1] und [s1, s2] skalieren
  mit s1, s2). Dadurch liegt jeder Kontaktpunkt immer exakt auf einem Knoten,
  ohne Sprünge beim Wandern - glatte Kraft-Weg-Kurve.
* Coulomb-Reibung Rolle/Probe optional (mu = 0: frei rollende Rollen),
  fest stehende Rollen mit Haften/Gleiten (Schlupf inkl. Abrollanteil R*dtheta).
  Reibkraft wirkt tangential an der Oberfläche -> Zusatzmoment ft*h/2.
* Wegsteuerung über die Lastrollenposition, Newton-Raphson mit konsistenter
  Tangente, adaptive Schrittweite.

Einheiten: mm, N, MPa (= N/mm^2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# --------------------------------------------------------------------------
# Eingabedaten
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Aufbau:
    E: float                 # Elastizitätsmodul [MPa]
    b: float                 # Probenbreite [mm]
    h: float                 # Probendicke [mm]
    Ls: float                # Stützweite (Abstand Stützrollen) [mm]
    Li: float                # Lastweite (Abstand Lastrollen) [mm]
    Rs: float = 2.0          # Radius Stützrollen [mm]
    Rl: float = 2.0          # Radius Lastrollen [mm]
    mu: float = 0.0          # Reibkoeffizient Rolle/Probe [-]
    nu: float = 0.22         # Querkontraktionszahl [-]
    ebener_dehnungszustand: bool = False  # True: E' = E/(1-nu^2) (breite Proben/Platten)
    Lp: float | None = None  # Probenlänge [mm] (nur für Abrutsch-Prüfung)

    @property
    def E_eff(self) -> float:
        return self.E / (1.0 - self.nu**2) if self.ebener_dehnungszustand else self.E

    @property
    def A(self) -> float:
        return self.b * self.h

    @property
    def I(self) -> float:
        return self.b * self.h**3 / 12.0

    @property
    def W(self) -> float:
        return self.b * self.h**2 / 6.0

    @property
    def a(self) -> float:
        """Hebelarm Stützrolle - Lastrolle (linear) [mm]."""
        return 0.5 * (self.Ls - self.Li)

    # --- lineare Balkentheorie (Referenz) ---------------------------------
    def sigma_linear(self, F):
        """Randspannung nach linearer Theorie: 3 F (Ls - Li) / (2 b h^2)."""
        return 3.0 * np.asarray(F) * (self.Ls - self.Li) / (2.0 * self.b * self.h**2)

    def F_aus_sigma_linear(self, sigma):
        return np.asarray(sigma) * 2.0 * self.b * self.h**2 / (3.0 * (self.Ls - self.Li))

    def nachgiebigkeit_linear(self):
        """Lineare Nachgiebigkeiten d/F (Lastrolle) und w0/F (Mitte) [mm/N]."""
        EI = self.E_eff * self.I
        a, L = self.a, self.Ls
        c_last = a**2 * (3 * L - 4 * a) / (12.0 * EI)          # P = F/2: P a^2(3L-4a)/(6EI)
        c_mitte = a * (3 * L**2 - 4 * a**2) / (48.0 * EI)      # P a(3L^2-4a^2)/(24EI)
        return c_last, c_mitte


@dataclass
class Ergebnis:
    aufbau: Aufbau
    d: np.ndarray            # Lastrollenweg (Traverse, relativ zu Stützrollen) [mm]
    w0: np.ndarray           # Durchbiegung Probenmitte [mm]
    F: np.ndarray            # Gesamtkraft (beide Lastrollen) [N]
    sigma: np.ndarray        # max. Zugspannung (nichtlinear) [MPa]
    x_sigma: np.ndarray      # x-Position der max. Zugspannung (aktuelle Lage) [mm]
    s1: np.ndarray           # materielle Koordinate Kontakt Lastrolle [mm]
    s2: np.ndarray           # materielle Koordinate Kontakt Stützrolle [mm]
    Fn1: np.ndarray          # Normalkraft Lastrolle (Halbmodell) [N]
    Fn2: np.ndarray          # Normalkraft Stützrolle (Halbmodell) [N]
    sigma_mitte: np.ndarray = None  # Zugspannung in Probenmitte [MPa]
    ft1: np.ndarray = None   # Tangentialkraft Lastrolle auf Probe (Halbmodell) [N]
    ft2: np.ndarray = None   # Tangentialkraft Stützrolle auf Probe (Halbmodell) [N]
    modus: list = None       # Reibzustand je Schritt (0 Haften, +-1 Gleiten)
    zustaende: list = field(default_factory=list)  # (q, s1, s2) je Schritt
    abgerutscht: bool = False
    warnung: str = ""

    @property
    def sigma_lin(self):
        return self.aufbau.sigma_linear(self.F)

    def sigma_bei(self, F):
        """Nichtlineare max. Zugspannung bei Kraft F (Interpolation)."""
        return np.interp(F, self.F, self.sigma, right=np.nan)

    def weg_bei(self, F, art="traverse"):
        y = self.d if art == "traverse" else self.w0
        return np.interp(F, self.F, y, right=np.nan)

    def kraft_bei_weg(self, weg, art="traverse"):
        y = self.d if art == "traverse" else self.w0
        return np.interp(weg, y, self.F, right=np.nan)


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
class Vierpunkt:
    """Korotationales Balken-FE-Halbmodell mit wandernden Rollenkontakten."""

    def __init__(self, aufbau: Aufbau, n_innen: int = 24, n_aussen: int = 24):
        self.a = aufbau
        self.n1, self.n2 = int(n_innen), int(n_aussen)
        self.ne = self.n1 + self.n2
        self.nn = self.ne + 1
        self.ndof = 3 * self.nn
        self.k1 = self.n1            # Knoten Kontakt Lastrolle
        self.k2 = self.ne            # Knoten Kontakt Stützrolle (Ende Modell)
        fixed = (0, 2)               # x0 = 0, theta0 = 0 (Symmetrie)
        self.free = np.array([i for i in range(self.ndof) if i not in fixed])
        self.nf = self.free.size
        e = np.arange(self.ne)
        self.edofs = np.stack([3*e, 3*e+1, 3*e+2, 3*e+3, 3*e+4, 3*e+5], axis=1)
        # dL0/ds1, dL0/ds2 je Element
        self.dL0ds = np.zeros((self.ne, 2))
        self.dL0ds[:self.n1, 0] = 1.0 / self.n1
        self.dL0ds[self.n1:, 0] = -1.0 / self.n2
        self.dL0ds[self.n1:, 1] = 1.0 / self.n2
        self.EA = aufbau.E_eff * aufbau.A
        self.EI = aufbau.E_eff * aufbau.I
        self._K = np.zeros((self.ndof, self.ndof))
        self._ii = np.broadcast_to(self.edofs[:, :, None], (self.ne, 6, 6))
        self._jj = np.broadcast_to(self.edofs[:, None, :], (self.ne, 6, 6))

    # --- Hilfen -----------------------------------------------------------
    def L0(self, s1, s2):
        L0 = np.empty(self.ne)
        L0[:self.n1] = s1 / self.n1
        L0[self.n1:] = (s2 - s1) / self.n2
        return L0

    def startzustand(self):
        a = self.a
        s1, s2 = 0.5 * a.Li, 0.5 * a.Ls
        X = np.concatenate([np.linspace(0, s1, self.n1 + 1),
                            np.linspace(s1, s2, self.n2 + 1)[1:]])
        q = np.zeros(self.ndof)
        q[0::3] = X
        return q, s1, s2

    # --- Elementebene (vektorisiert) ----------------------------------------
    def intern(self, q, s1, s2, tangente=True):
        x, y, th = q[0::3], q[1::3], q[2::3]
        dx, dy = np.diff(x), np.diff(y)
        Ln = np.hypot(dx, dy)
        c, s = dx / Ln, dy / Ln
        beta = np.arctan2(dy, dx)
        t1 = np.angle(np.exp(1j * (th[:-1] - beta)))
        t2 = np.angle(np.exp(1j * (th[1:] - beta)))
        L0 = self.L0(s1, s2)
        EA, EI = self.EA, self.EI
        N = EA * (Ln - L0) / L0
        M1 = EI / L0 * (4 * t1 + 2 * t2)
        M2 = EI / L0 * (2 * t1 + 4 * t2)

        o = np.zeros_like(c)
        r = np.stack([-c, -s, o, c, s, o], axis=1)
        z = np.stack([s, -c, o, -s, c, o], axis=1)
        B = np.zeros((self.ne, 3, 6))
        B[:, 0] = r
        B[:, 1] = -z / Ln[:, None]
        B[:, 2] = -z / Ln[:, None]
        B[:, 1, 2] += 1.0
        B[:, 2, 5] += 1.0
        fl = np.stack([N, M1, M2], axis=1)
        fe = np.einsum('eij,ei->ej', B, fl)
        f = np.zeros(self.ndof)
        np.add.at(f, self.edofs, fe)
        daten = dict(N=N, M1=M1, M2=M2, L0=L0, Ln=Ln)
        if not tangente:
            return f, daten

        Kl = np.zeros((self.ne, 3, 3))
        Kl[:, 0, 0] = EA / L0
        Kl[:, 1, 1] = Kl[:, 2, 2] = 4 * EI / L0
        Kl[:, 1, 2] = Kl[:, 2, 1] = 2 * EI / L0
        Ke = np.einsum('eki,ekl,elj->eij', B, Kl, B)
        Ke += (N / Ln)[:, None, None] * z[:, :, None] * z[:, None, :]
        Ke += ((M1 + M2) / Ln**2)[:, None, None] * (r[:, :, None] * z[:, None, :]
                                                    + z[:, :, None] * r[:, None, :])
        K = self._K
        K[:] = 0.0
        np.add.at(K, (self._ii, self._jj), Ke)

        dfl = np.stack([-EA * Ln / L0**2, -M1 / L0, -M2 / L0], axis=1)
        dfe = np.einsum('eij,ei->ej', B, dfl)
        dfds = np.zeros((self.ndof, 2))
        for k in range(2):
            np.add.at(dfds[:, k], self.edofs, dfe * self.dL0ds[:, k:k+1])
        return f, K.copy(), dfds, daten

    # --- Kontakt ------------------------------------------------------------
    # Unbekannte z = [q_frei, s1, s2, Fn1, Fn2, ft1, ft2]
    #   Fn: Normalkraft (>0, Druck), ft: Tangentialkraft auf die Probe in +t-Richtung
    # Reibmodus je Kontakt: 0 = Haften, +1/-1 = Gleiten (ft = +-mu*Fn)
    #
    # Schlupf (Relativverschiebung Probe/Rolle, fest stehende Rolle) je Schritt:
    #   Lastrolle  (Oberseite):  dg1 = c1*ds1 - Rl*dth1,  c1 = (1+eps) - kappa*h/2
    #   Stützrolle (Unterseite): dg2 = c2*ds2 + Rs*dth2,  c2 = (1+eps) + kappa*h/2
    # Haften: dg = 0 und |ft| <= mu*Fn.  Gleiten: ft = mu*Fn*sign(dg).
    # (frei drehende Rollen: mu = 0)

    def kontakt(self, q, Fn1, Fn2, ft1, ft2, d):
        """Kontaktkräfte auf die Probe und geometrische Kontaktbedingungen.

        Rückgabe: fext, dfext/dq, dfext/d(Fn1, Fn2, ft1, ft2), g (4), dg/dq (4 x ndof)
        """
        a = self.a
        h2 = 0.5 * a.h
        i1, i2 = 3 * self.k1, 3 * self.k2
        th1, th2 = q[i1 + 2], q[i2 + 2]
        n1 = np.array([-np.sin(th1), np.cos(th1)]); t1 = np.array([np.cos(th1), np.sin(th1)])
        n2 = np.array([-np.sin(th2), np.cos(th2)]); t2 = np.array([np.cos(th2), np.sin(th2)])

        f = np.zeros(self.ndof)
        dfdq = np.zeros((self.ndof, self.ndof))
        dfdF = np.zeros((self.ndof, 4))
        # Lastrolle (Oberseite): F = -Fn1 n + ft1 t, Moment um Achse = -(h/2) ft1
        f[i1:i1+2] = -Fn1 * n1 + ft1 * t1
        f[i1 + 2] = -h2 * ft1
        dfdq[i1:i1+2, i1 + 2] = Fn1 * t1 + ft1 * n1
        dfdF[i1:i1+2, 0] = -n1
        dfdF[i1:i1+2, 2] = t1
        dfdF[i1 + 2, 2] = -h2
        # Stützrolle (Unterseite): F = Fn2 n + ft2 t, Moment = +(h/2) ft2
        f[i2:i2+2] = Fn2 * n2 + ft2 * t2
        f[i2 + 2] = h2 * ft2
        dfdq[i2:i2+2, i2 + 2] = -Fn2 * t2 + ft2 * n2
        dfdF[i2:i2+2, 1] = n2
        dfdF[i2:i2+2, 3] = t2
        dfdF[i2 + 2, 3] = h2

        Cl = np.array([0.5 * a.Li, h2 + a.Rl - d])
        Cs = np.array([0.5 * a.Ls, -h2 - a.Rs])
        rl, rs = h2 + a.Rl, h2 + a.Rs
        g = np.concatenate([q[i1:i1+2] + rl * n1 - Cl,
                            q[i2:i2+2] - rs * n2 - Cs])
        dg = np.zeros((4, self.ndof))
        dg[0, i1] = dg[1, i1 + 1] = 1.0
        dg[0:2, i1 + 2] = -rl * t1
        dg[2, i2] = dg[3, i2 + 1] = 1.0
        dg[2:4, i2 + 2] = rs * t2
        return f, dfdq, dfdF, g, dg

    def _schlupf_ref(self, z):
        """Referenz für den Schlupf: (s1, s2, th1, th2, c1, c2) im Zustand z."""
        q = self._q_aus_z(z)
        s1, s2 = z[self.nf:self.nf + 2]
        _, dat = self.intern(q, s1, s2, tangente=False)
        th = q[2::3]
        k1, k2 = self.k1, self.k2
        L0 = dat['L0']
        kap1 = (th[k1 + 1] - th[k1 - 1]) / (L0[k1 - 1] + L0[k1])
        kap2 = (th[k2] - th[k2 - 1]) / L0[k2 - 1]
        eps1 = 0.5 * (dat['N'][k1 - 1] + dat['N'][k1]) / self.EA
        eps2 = dat['N'][k2 - 1] / self.EA
        h2 = 0.5 * self.a.h
        return np.array([s1, s2, th[k1], th[k2], 1 + eps1 - kap1 * h2, 1 + eps2 + kap2 * h2])

    def schlupf(self, z, ref):
        q = self._q_aus_z(z)
        s1, s2 = z[self.nf:self.nf + 2]
        th1, th2 = q[3 * self.k1 + 2], q[3 * self.k2 + 2]
        dg1 = ref[4] * (s1 - ref[0]) - self.a.Rl * (th1 - ref[2])
        dg2 = ref[5] * (s2 - ref[1]) + self.a.Rs * (th2 - ref[3])
        return np.array([dg1, dg2])

    # --- Gleichungssystem ---------------------------------------------------
    def _system(self, z, d, modus, ref):
        n = self.nf
        q = self._q_aus_z(z)
        s1, s2, Fn1, Fn2, ft1, ft2 = z[n:]
        fi, K, dfds, _ = self.intern(q, s1, s2)
        fe, dfedq, dfedF, g, dg = self.kontakt(q, Fn1, Fn2, ft1, ft2, d)
        fr = self.free
        mu = self.a.mu
        R = np.zeros(n + 6)
        J = np.zeros((n + 6, n + 6))
        R[:n] = (fi - fe)[fr]
        R[n:n + 4] = g
        J[:n, :n] = (K - dfedq)[np.ix_(fr, fr)]
        J[:n, n:n+2] = dfds[fr]
        J[:n, n+2:n+6] = -dfedF[fr]
        J[n:n+4, :n] = dg[:, fr]
        # Reibgleichungen
        dgl = self.schlupf(z, ref)
        ith = (3 * self.k1 + 2, 3 * self.k2 + 2)
        Rk = (self.a.Rl, self.a.Rs)
        vz = (-1.0, 1.0)
        for k in range(2):
            r = n + 4 + k
            if modus[k] == 0:          # Haften: kein Schlupf
                R[r] = dgl[k]
                J[r, n + k] = ref[4 + k]
                col = np.searchsorted(fr, ith[k])
                J[r, col] = vz[k] * Rk[k]
            else:                      # Gleiten: ft = mu*sign*Fn
                R[r] = z[n + 4 + k] - mu * modus[k] * z[n + 2 + k]
                J[r, n + 4 + k] = 1.0
                J[r, n + 2 + k] = -mu * modus[k]
        return R, J

    def _q_aus_z(self, z):
        q = np.zeros(self.ndof)
        q[self.free] = z[:self.nf]
        return q

    def newton(self, z0, d, modus, ref, tol=1e-10, maxit=25):
        z = z0.copy()
        Fref = self.EI * self.a.h / self.a.Ls**3
        F_rausch = 200 * np.finfo(float).eps * self.EA   # Rundungsgrenze der Normalkraft
        n = self.nf
        for it in range(1, maxit + 1):
            R, J = self._system(z, d, modus, ref)
            Fsk = max(Fref, abs(z[n + 2]), abs(z[n + 3]))
            rf = max(np.max(np.abs(R[:n])) - F_rausch, 0.0) / Fsk
            rg = np.max(np.abs(R[n:n + 4])) / self.a.h
            rt = max(abs(R[n + 4 + k]) / (self.a.h if modus[k] == 0 else Fsk) for k in range(2))
            if it > 1 and rf < tol and rg < tol and rt < tol:
                return z, it, True
            try:
                dz = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                return z, it, False
            if not np.all(np.isfinite(dz)):
                return z, it, False
            z = z + dz
            if it > 1 and np.max(np.abs(dz[:n])) < 1e-13 * self.a.Ls and rf < 1e-7:
                return z, it, True
        return z, maxit, False

    def schritt(self, zp, d, modus, ref):
        """Ein Lastschritt mit Haft-/Gleit-Iteration (Coulomb)."""
        mu = self.a.mu
        n = self.nf
        if mu == 0:
            return (*self.newton(zp, d, (1, 1), ref), (1, 1))
        modus = list(modus)
        gesehen = set()
        for _ in range(8):
            zn, it, ok = self.newton(zp, d, modus, ref)
            if not ok:
                return zn, it, ok, tuple(modus)
            dgl = self.schlupf(zn, ref)
            neu = list(modus)
            for k in range(2):
                Fn, ft = zn[n + 2 + k], zn[n + 4 + k]
                if modus[k] == 0 and abs(ft) > mu * Fn * (1 + 1e-9):
                    neu[k] = 1 if ft > 0 else -1
                elif modus[k] != 0 and modus[k] * dgl[k] < -1e-12 * self.a.h:
                    neu[k] = 0
            if neu == modus:
                return zn, it, True, tuple(modus)
            gesehen.add(tuple(modus))
            if tuple(neu) in gesehen:          # Umschaltpunkt: Gleiten akzeptieren
                neu = [m if m != 0 else (1 if zn[n + 4 + k] >= 0 else -1) for k, m in enumerate(neu)]
                zn, it, ok = self.newton(zp, d, neu, ref)
                return zn, it, ok, tuple(neu)
            modus = neu
        return zn, it, ok, tuple(modus)

    # --- Auswertung eines Zustands -----------------------------------------
    def kenngroessen(self, z):
        q = self._q_aus_z(z)
        s1, s2, Fn1, Fn2, ft1, ft2 = z[self.nf:]
        _, dat = self.intern(q, s1, s2, tangente=False)
        a = self.a
        th1 = q[3 * self.k1 + 2]
        F_half = Fn1 * np.cos(th1) - ft1 * np.sin(th1)
        sig_e = dat['N'] / a.A + np.maximum(np.abs(dat['M1']), np.abs(dat['M2'])) / a.W
        ie = int(np.argmax(sig_e))
        xm = 0.5 * (q[3 * ie] + q[3 * ie + 3])
        s_mitte = dat['N'][0] / a.A + abs(dat['M1'][0]) / a.W
        return dict(F=2.0 * F_half, sigma=float(sig_e[ie]), x_sigma=float(xm),
                    sigma_mitte=float(s_mitte), w0=-q[1], s1=s1, s2=s2, Fn1=Fn1, Fn2=Fn2,
                    ft1=ft1, ft2=ft2, q=q)

    def randspannung(self, z):
        """Zugseitige Randspannung je Knoten (für Verlaufsplot)."""
        q = self._q_aus_z(z)
        s1, s2 = z[self.nf:self.nf + 2]
        _, dat = self.intern(q, s1, s2, tangente=False)
        a = self.a
        sl = dat['N'] / a.A + np.abs(dat['M1']) / a.W
        sr = dat['N'] / a.A + np.abs(dat['M2']) / a.W
        sk = np.empty(self.nn)
        sk[0], sk[-1] = sl[0], sr[-1]
        sk[1:-1] = 0.5 * (sr[:-1] + sl[1:])
        return q[0::3], q[1::3], sk

    # --- Lastpfad -------------------------------------------------------------
    def rechne(self, F_ziel=None, d_max=None, n_schritte=40, speichere_zustaende=True):
        """Wegesteuerte Rechnung bis Kraft F_ziel (exakt getroffen) oder Weg d_max."""
        a = self.a
        c_last, _ = a.nachgiebigkeit_linear()
        if F_ziel is None and d_max is None:
            raise ValueError("F_ziel oder d_max angeben")
        d_ref = F_ziel * c_last if F_ziel is not None else d_max
        if d_max is None:
            d_max = 50.0 * d_ref
        dd_grenz = a.Ls / 300.0                      # max. Wegschritt (Geometrie)
        dd = min(d_ref / n_schritte, dd_grenz)
        dd_min = min(d_ref, dd_grenz) * 1e-7
        n = self.nf

        q0, s1, s2 = self.startzustand()
        z = np.concatenate([q0[self.free], [s1, s2, 0.0, 0.0, 0.0, 0.0]])
        modus = (0, 0) if a.mu > 0 else (1, 1)
        k0 = self.kenngroessen(z)
        k0.update(d=0.0, z=z.copy(), modus=modus)
        hist = [k0]
        z_alt, d_alt = None, None
        d = 0.0
        abgerutscht, warnung = False, ""
        ziel_erreicht = False
        while True:
            d_neu = min(d + dd, d_max)
            # Prädiktor: lineare Extrapolation
            if z_alt is not None and d > d_alt:
                zp = z + (z - z_alt) * (d_neu - d) / (d - d_alt)
            else:
                zp = z.copy()
            ref = self._schlupf_ref(z)
            zn, it, ok, modus_n = self.schritt(zp, d_neu, modus, ref)
            if ok and not (0 < zn[n] < zn[n + 1]):
                ok = False
            if ok and (zn[n + 2] < 0 or zn[n + 3] < 0):
                ok = False
                warnung = "Kontakt abgehoben (negative Normalkraft)."
            if not ok:
                dd *= 0.5
                if dd < dd_min:
                    warnung = warnung or "Newton-Verfahren konvergiert nicht."
                    break
                continue
            z_alt, d_alt = z, d
            z, d, modus = zn, d_neu, modus_n
            k = self.kenngroessen(z)
            k.update(d=d, z=z.copy(), modus=modus)
            hist.append(k)
            if a.Lp is not None and k['s2'] > 0.5 * a.Lp:
                abgerutscht = True
                warnung = "Probe rutscht über die Stützrollen (s2 > Lp/2)."
                break
            if F_ziel is not None and k['F'] >= F_ziel:
                ziel_erreicht = True
                break
            if d >= d_max - 1e-15:
                break
            dd = dd * (1.4 if it <= 3 else (1.0 if it <= 5 else 0.6))
            dd = min(dd, d_ref / 8, dd_grenz)

        # Letzten Punkt exakt auf F_ziel legen (Sekantenverfahren auf d)
        if ziel_erreicht and len(hist) >= 2:
            h0, h1 = hist[-2], hist[-1]
            ref = self._schlupf_ref(h0['z'])
            modus = h0['modus']
            za, da, Fa = h0['z'], h0['d'], h0['F']
            zb, db, Fb = h1['z'], h1['d'], h1['F']
            for _ in range(10):
                if abs(Fb - F_ziel) < 1e-9 * F_ziel:
                    break
                dc = db + (F_ziel - Fb) * (db - da) / (Fb - Fa)
                zc = zb + (zb - za) * (dc - db) / (db - da)
                zc, it, ok, modus_c = self.schritt(zc, dc, modus, ref)
                if not ok:
                    break
                kc = self.kenngroessen(zc)
                za, da, Fa = zb, db, Fb
                zb, db, Fb = zc, dc, kc['F']
            kc = self.kenngroessen(zb)
            kc.update(d=db, z=zb.copy(), modus=h1['modus'])
            hist[-1] = kc
            if len(hist) > 2 and hist[-1]['F'] < hist[-2]['F']:
                hist.pop(-2)

        nicht_erreichbar = False
        if F_ziel is not None and not ziel_erreicht:
            # Nur den ansteigenden Ast behalten (jenseits des Kraftmaximums ist
            # sigma(F) nicht mehr eindeutig: Probe wird durch die Stützen gezogen)
            Fh = np.array([hh['F'] for hh in hist])
            imax = int(np.argmax(Fh))
            if imax < len(hist) - 1:
                hist = hist[:imax + 1]
            nicht_erreichbar = True
            warnung = (f"Zielkraft {F_ziel:.4g} N nicht erreicht; max. Kraft im Modell "
                       f"{hist[-1]['F']:.4g} N. " + warnung).strip()

        arr = lambda key: np.array([hh[key] for hh in hist])
        return Ergebnis(
            aufbau=a, d=arr('d'), w0=arr('w0'), F=arr('F'), sigma=arr('sigma'),
            x_sigma=arr('x_sigma'), sigma_mitte=arr('sigma_mitte'), s1=arr('s1'), s2=arr('s2'),
            Fn1=arr('Fn1'), Fn2=arr('Fn2'), ft1=arr('ft1'), ft2=arr('ft2'),
            modus=[hh['modus'] for hh in hist],
            zustaende=[hh['z'] for hh in hist] if speichere_zustaende else [hist[-1]['z']],
            abgerutscht=abgerutscht or nicht_erreichbar, warnung=warnung)


def berechne(aufbau: Aufbau, F_ziel=None, d_max=None, n_innen=24, n_aussen=24):
    """Bequemer Einstieg: Lastpfad bis F_ziel (oder d_max)."""
    return Vierpunkt(aufbau, n_innen, n_aussen).rechne(F_ziel=F_ziel, d_max=d_max)


def verformung(aufbau: Aufbau, z, n_innen=24, n_aussen=24):
    """Knotenlagen (x, y) und Randspannung für einen gespeicherten Zustand."""
    return Vierpunkt(aufbau, n_innen, n_aussen).randspannung(z)
