"""
weibull.py - Zweiparametrige Weibull-Auswertung von Festigkeitsdaten.

* Maximum-Likelihood-Schätzung nach DIN EN 843-5 / ASTM C1239
* Erwartungstreue Korrektur von m und 90-%-Konfidenzintervalle für m und
  sigma_0 über pivotale Größen (m_hat/m und m_hat*ln(s0_hat/s0) sind
  unabhängig von den wahren Parametern) -> Monte-Carlo-Tabellen für jedes N,
  identisch zum Vorgehen, mit dem die Normtabellen erzeugt wurden.
* Versagenswahrscheinlichkeit der Messwerte: F_i = (i - 0.5) / N
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np


def _ml_m(x: np.ndarray, m0=None, maxit=100) -> np.ndarray:
    """ML-Schätzer für m, vektorisiert über Zeilen von x (Proben x N).

    Löst  sum(x^m ln x)/sum(x^m) - 1/m - mean(ln x) = 0  mit Newton.
    """
    x = np.atleast_2d(x)
    lx = np.log(x)
    lx = lx - lx.mean(axis=1, keepdims=True)       # Skaleninvarianz -> stabil
    mbar = 0.0
    sd = lx.std(axis=1)
    m = 1.2 / np.maximum(sd, 1e-300) if m0 is None else np.full(x.shape[0], m0, float)
    for _ in range(maxit):
        w = np.exp(m[:, None] * (lx - lx.max(axis=1, keepdims=True)))
        S0 = w.sum(1)
        S1 = (w * lx).sum(1)
        S2 = (w * lx**2).sum(1)
        g = S1 / S0 - 1.0 / m - mbar
        dg = S2 / S0 - (S1 / S0) ** 2 + 1.0 / m**2
        dm = -g / dg
        m_neu = np.where(m + dm > 0, m + dm, 0.5 * m)
        if np.all(np.abs(m_neu - m) <= 1e-12 * m_neu):
            m = m_neu
            break
        m = m_neu
    return m


def ml_schaetzung(sigma) -> tuple[float, float]:
    """ML-Schätzwerte (m, sigma_0)."""
    s = np.asarray(sigma, float)
    m = float(_ml_m(s[None, :])[0])
    s0 = float(np.mean(s**m) ** (1.0 / m))
    return m, s0


@lru_cache(maxsize=64)
def _pivot_tabelle(N: int, n_mc: int = 20000, seed: int = 4711):
    """Verteilungen der pivotalen Größen für Stichprobenumfang N."""
    rng = np.random.default_rng(seed)
    x = rng.exponential(size=(n_mc, N))          # Weibull mit m = 1, sigma_0 = 1
    m = _ml_m(x)
    s0 = np.mean(x ** m[:, None], axis=1) ** (1.0 / m)
    p_m = m                                      # m_hat / m   (m = 1)
    p_s = m * np.log(s0)                         # m_hat * ln(s0_hat / s0)
    return p_m, p_s


@dataclass
class WeibullErgebnis:
    N: int
    m: float                 # ML-Schätzwert
    m_korr: float            # erwartungstreu korrigiert
    s0: float                # ML-Schätzwert sigma_0 [MPa]
    m_ki: tuple              # Konfidenzintervall m
    s0_ki: tuple             # Konfidenzintervall sigma_0
    korrekturfaktor: float   # b_N
    niveau: float
    sigma_sortiert: np.ndarray
    F_i: np.ndarray          # Versagenswahrscheinlichkeiten (i-0.5)/N

    def F_modell(self, sigma):
        return 1.0 - np.exp(-(np.asarray(sigma) / self.s0) ** self.m)

    def mittelwert(self):
        from math import gamma
        return self.s0 * gamma(1 + 1 / self.m)


def auswerten(sigma, niveau: float = 0.90) -> WeibullErgebnis:
    s = np.sort(np.asarray(sigma, float))
    s = s[np.isfinite(s) & (s > 0)]
    N = s.size
    if N < 3:
        raise ValueError("Mindestens 3 gültige Festigkeitswerte nötig.")
    m, s0 = ml_schaetzung(s)
    p_m, p_s = _pivot_tabelle(N)
    b_N = 1.0 / p_m.mean()
    al = 0.5 * (1.0 - niveau)
    qm_lo, qm_hi = np.quantile(p_m, [al, 1 - al])
    qs_lo, qs_hi = np.quantile(p_s, [al, 1 - al])
    m_ki = (m / qm_hi, m / qm_lo)
    s0_ki = (s0 * np.exp(-qs_hi / m), s0 * np.exp(-qs_lo / m))
    F_i = (np.arange(1, N + 1) - 0.5) / N
    return WeibullErgebnis(N=N, m=m, m_korr=m * b_N, s0=s0, m_ki=m_ki, s0_ki=s0_ki,
                           korrekturfaktor=b_N, niveau=niveau, sigma_sortiert=s, F_i=F_i)


if __name__ == "__main__":
    # Selbsttest: Korrekturfaktoren gegen Tabellenwerte (ASTM C1239 / EN 843-5)
    for N, soll in [(5, 0.700), (10, 0.859), (20, 0.934), (30, 0.955)]:
        p_m, _ = _pivot_tabelle(N)
        print(f"N={N:3d}: b_N = {1 / p_m.mean():.3f} (Tabelle {soll:.3f})")
    rng = np.random.default_rng(1)
    x = 500 * rng.weibull(12, size=30)
    r = auswerten(x)
    print(f"m={r.m:.2f} (korr {r.m_korr:.2f}, KI {r.m_ki[0]:.2f}-{r.m_ki[1]:.2f}), "
          f"s0={r.s0:.1f} (KI {r.s0_ki[0]:.1f}-{r.s0_ki[1]:.1f})")
