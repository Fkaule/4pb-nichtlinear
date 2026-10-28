"""
datenimport.py - robustes Einlesen von Kraft-Weg-Kurven (CSV/TXT aus
Prüfmaschinen, z. B. testXpert-Export mit Kopfzeilen, Semikolon, Dezimalkomma).
"""
from __future__ import annotations

import io
import re
import numpy as np
import pandas as pd

_TRENNER = [";", "\t", ",", " "]


def _dekodiere(roh: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return roh.decode(enc)
        except UnicodeDecodeError:
            continue
    return roh.decode("latin-1", errors="replace")


def _zahl(feld: str, dezimalkomma: bool):
    f = feld.strip().strip('"')
    if dezimalkomma:
        f = f.replace(".", "").replace(",", ".") if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+,\d*", f) else f.replace(",", ".")
    try:
        return float(f)
    except ValueError:
        return None


def _zerlege(zeile: str, trenner: str):
    if trenner == " ":
        return zeile.split()
    return zeile.split(trenner)


def lies_kurve(roh: bytes) -> pd.DataFrame:
    """Liest eine Messdatei und gibt den numerischen Datenblock als DataFrame
    zurück (Spaltennamen aus der Kopfzeile, falls vorhanden)."""
    text = _dekodiere(roh)
    zeilen = [z for z in text.splitlines() if z.strip()]
    beste = None
    for tr in _TRENNER:
        for dk in (tr != ",", False) if tr != "," else (False,):
            # numerische Zeilen finden
            num = []
            for i, z in enumerate(zeilen):
                felder = _zerlege(z, tr)
                felder = [f for f in felder if f.strip()]
                werte = [_zahl(f, dk) for f in felder]
                if len(werte) >= 2 and all(w is not None for w in werte):
                    num.append(i)
            if not num:
                continue
            # längster zusammenhängender Block
            bloecke, start = [], num[0]
            for a, b in zip(num, num[1:]):
                if b != a + 1:
                    bloecke.append((start, a))
                    start = b
            bloecke.append((start, num[-1]))
            s, e = max(bloecke, key=lambda be: be[1] - be[0])
            n_zeilen = e - s + 1
            n_sp = len(_zerlege(zeilen[s], tr))
            bewertung = n_zeilen * 10 + n_sp
            if beste is None or bewertung > beste[0]:
                beste = (bewertung, tr, dk, s, e, n_sp)
    if beste is None:
        raise ValueError("Kein numerischer Datenblock gefunden.")
    _, tr, dk, s, e, n_sp = beste

    daten = []
    for z in zeilen[s:e + 1]:
        werte = [_zahl(f, dk) for f in _zerlege(z, tr)]
        werte = (werte + [None] * n_sp)[:n_sp]
        daten.append([np.nan if w is None else w for w in werte])

    # Kopfzeile: nächste Textzeile davor mit passender Spaltenzahl
    namen = None
    for j in range(s - 1, max(s - 6, -1), -1):
        felder = [f.strip().strip('"') for f in _zerlege(zeilen[j], tr)]
        if len(felder) == n_sp and sum(_zahl(f, dk) is None for f in felder) == n_sp:
            # Einheitenzeile (z. B. "mm;N") überspringen, Namenszeile davor bevorzugen
            if j - 1 >= 0:
                vorher = [f.strip().strip('"') for f in _zerlege(zeilen[j - 1], tr)]
                if len(vorher) == n_sp and all(_zahl(f, dk) is None for f in vorher) \
                        and all(len(f) <= 6 for f in felder):
                    namen = [f"{a} [{b}]" if b else a for a, b in zip(vorher, felder)]
                    break
            namen = felder
            break
    if namen is None or len(set(namen)) != len(namen):
        namen = [f"Spalte {i + 1}" for i in range(n_sp)]
    df = pd.DataFrame(daten, columns=namen).dropna(axis=1, how="all").dropna()
    return df


def spalten_raten(spalten) -> tuple[str, str]:
    """Rät (Weg-Spalte, Kraft-Spalte) anhand der Namen."""
    sp = list(spalten)
    low = [s.lower() for s in sp]
    kraft = next((sp[i] for i, s in enumerate(low)
                  if any(k in s for k in ("kraft", "force", "last", "load", "[n]", "[kn]", "standard force"))), None)
    weg = next((sp[i] for i, s in enumerate(low)
                if any(k in s for k in ("weg", "disp", "deflect", "durchb", "traverse", "dehnung", "[mm]", "position"))
                and sp[i] != kraft), None)
    if kraft is None:
        kraft = sp[1] if len(sp) > 1 else sp[0]
    if weg is None:
        weg = next((s for s in sp if s != kraft), sp[0])
    return weg, kraft


def lies_probentabelle(roh: bytes) -> pd.DataFrame:
    """Probentabelle mit Spalten ID, b, h, optional F_max. Trenner/Dezimal automatisch."""
    text = _dekodiere(roh)
    erste = text.splitlines()[0] if text else ""
    tr = ";" if ";" in erste else ("\t" if "\t" in erste else ",")
    dez = "," if tr != "," else "."
    df = pd.read_csv(io.StringIO(text), sep=tr, decimal=dez)
    df.columns = [c.strip() for c in df.columns]
    abbild = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("id", "probe", "name", "proben-id", "probennr", "nr"):
            abbild[c] = "ID"
        elif cl.startswith("b") and ("breite" in cl or cl.split()[0] in ("b", "b[mm]") or cl == "b"):
            abbild[c] = "b [mm]"
        elif cl.startswith("h") or "dicke" in cl:
            abbild[c] = "h [mm]"
        elif "f" in cl and ("max" in cl or "bruch" in cl):
            abbild[c] = "F_Bruch [N]"
    df = df.rename(columns=abbild)
    if "ID" not in df.columns:
        df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip()
    return df
