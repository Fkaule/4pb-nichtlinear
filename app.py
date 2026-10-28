"""
4-Punkt-Biegung mit großer Durchbiegung - nichtlineare Spannungsauswertung
und Weibull-Statistik.   Start:  streamlit run app.py
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import datenimport as di
import weibull
from vierpunkt_fe import Aufbau, berechne, verformung

st.set_page_config(page_title="4PB nichtlinear", layout="wide")

FARBE = {"fe": "#2a78d6", "lin": "#eb6834", "mess": "#1baf7a", "rolle": "#8a8985"}
BEISPIEL = Path(__file__).parent / "beispieldaten"


# ---------------------------------------------------------------------------
# Rechenkern (gecacht)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=500)
def fe_rechnung(aufbau: Aufbau, F_ziel: float, n_el: int):
    return berechne(aufbau, F_ziel=F_ziel, n_innen=n_el, n_aussen=n_el)


@st.cache_data(show_spinner=False)
def lies_datei(name: str, roh: bytes):
    return di.lies_kurve(roh)


def de_csv(df: pd.DataFrame) -> bytes:
    return df.round(4).to_csv(sep=";", decimal=",", index=False).encode("utf-8-sig")


def layout(fig, xt, yt, hoehe=380):
    fig.update_layout(height=hoehe, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title=xt, yaxis_title=yt, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Seitenleiste: Material, Geometrie, Aufbau
# ---------------------------------------------------------------------------
sb = st.sidebar
sb.header("Material")
E = sb.number_input("E-Modul [GPa]", 1.0, 1000.0, 72.0, 1.0)
nu = sb.number_input("Querkontraktionszahl ν [-]", 0.0, 0.49, 0.22, 0.01)
edz = sb.checkbox("Ebener Dehnungszustand (E' = E/(1-ν²))", True,
                  help="Für breite, plattenartige Proben (Wafer, b ≫ h): Querdehnung behindert. "
                       "Gegen ANSYS-Schalenmodelle (b/L_s = 0,7-2,9) genauer als E. "
                       "Für schmale Balken (b ≈ h) ausschalten.")
sb.header("Probe (Nennmaße)")
b_nenn = sb.number_input("Breite b [mm]", 0.01, 500.0, 4.0, 0.1, format="%.3f")
h_nenn = sb.number_input("Dicke h [mm]", 0.001, 100.0, 0.40, 0.01, format="%.4f")
Lp = sb.number_input("Probenlänge L_p [mm]", 0.0, 2000.0, 45.0, 1.0,
                     help="Nur für die Prüfung, ob die Probe über die Stützrollen rutscht. 0 = ignorieren.")
sb.header("Prüfaufbau")
Ls = sb.number_input("Stützweite L_s [mm]", 0.1, 2000.0, 40.0, 1.0)
Li = sb.number_input("Lastweite L_i [mm]", 0.0, 2000.0, 20.0, 1.0)
Rs = sb.number_input("Radius Stützrollen R_s [mm]", 0.0, 100.0, 2.5, 0.1)
Rl = sb.number_input("Radius Lastrollen R_l [mm]", 0.0, 100.0, 2.5, 0.1)
mu = sb.number_input("Reibkoeffizient μ Rolle/Probe [-]", 0.0, 1.0, 0.20, 0.05,
                     help="0 = frei rollende Rollen (EN 843-1 / ASTM C1161). "
                          "> 0: fest stehende Rollen, Coulomb mit Haften/Gleiten "
                          "(typ. 0,1-0,3; an Versuch/Simulation abgleichen).")
sb.header("Numerik")
n_el = sb.slider("Elemente je Abschnitt", 8, 64, 20, 4,
                 help="Je Abschnitt Mitte-Lastrolle und Lastrolle-Stützrolle. "
                      "20 → Diskretisierungsfehler ≈ 3e-5 (gegen Elastica verifiziert).")

if Li >= Ls:
    st.error("Lastweite muss kleiner als die Stützweite sein.")
    st.stop()

basis = Aufbau(E=E * 1000.0, nu=nu, ebener_dehnungszustand=edz, b=b_nenn, h=h_nenn,
               Ls=Ls, Li=Li, Rs=Rs, Rl=Rl, mu=mu, Lp=Lp if Lp > 0 else None)

st.title("4-Punkt-Biegung: nichtlineare Festigkeitsauswertung")
tab_daten, tab_probe, tab_weibull, tab_methode = st.tabs(
    ["1 · Daten", "2 · Ergebnisse je Probe", "3 · Weibull", "Methode"])

# ---------------------------------------------------------------------------
# Tab 1: Daten
# ---------------------------------------------------------------------------
with tab_daten:
    c1, c2 = st.columns([3, 2])
    with c1:
        dateien = st.file_uploader("Kraft-Weg-Kurven (eine Datei je Probe, CSV/TXT)",
                                   accept_multiple_files=True, type=["csv", "txt", "dat", "asc"])
    with c2:
        ptab_datei = st.file_uploader("Probentabelle (optional: ID; b; h; F_Bruch)",
                                      type=["csv", "txt"])
        if st.button("Beispieldaten laden (20 Glasproben, h ≈ 0,4 mm)"):
            st.session_state["beispiel"] = True
        if st.session_state.get("beispiel") and not dateien:
            st.caption("Beispieldaten aktiv. Passende Eingaben: E = 72 GPa, ν = 0,22, "
                       "L_s = 40, L_i = 20, R = 2,5 mm, μ = 0,2, L_p = 45 mm, ebener Dehnungszustand an.")

    kurven_roh: dict[str, bytes] = {}
    ptab_roh = None
    if dateien:
        kurven_roh = {Path(f.name).stem: f.getvalue() for f in dateien}
        ptab_roh = ptab_datei.getvalue() if ptab_datei else None
    elif st.session_state.get("beispiel"):
        kurven_roh = {p.stem: p.read_bytes() for p in sorted(BEISPIEL.glob("P*.csv"))}
        ptab_roh = (ptab_datei.getvalue() if ptab_datei
                    else (BEISPIEL / "probentabelle.csv").read_bytes())
    elif ptab_datei:
        ptab_roh = ptab_datei.getvalue()

    kurven: dict[str, pd.DataFrame] = {}
    fehler = []
    for name, roh in kurven_roh.items():
        try:
            kurven[name] = lies_datei(name, roh)
        except Exception as ex:  # noqa: BLE001
            fehler.append(f"{name}: {ex}")
    if fehler:
        st.warning("Nicht lesbar:\n\n" + "\n\n".join(fehler))

    ptab = None
    if ptab_roh:
        try:
            ptab = di.lies_probentabelle(ptab_roh)
        except Exception as ex:  # noqa: BLE001
            st.warning(f"Probentabelle nicht lesbar: {ex}")

    if not kurven and ptab is None:
        st.info("Kurven hochladen, eine Probentabelle mit Bruchkräften hochladen "
                "oder Beispieldaten laden.")
        st.stop()

    # --- Spaltenzuordnung ---------------------------------------------------
    weg_sp = kraft_sp = None
    faktor_F, faktor_d = 1.0, 1.0
    weg_art, C_m, nullpunkt = "traverse", 0.0, True
    if kurven:
        erste = next(iter(kurven.values()))
        spalten = list(erste.columns)
        w_g, k_g = di.spalten_raten(spalten)
        st.subheader("Kurvenformat")
        k1, k2, k3, k4 = st.columns(4)
        weg_sp = k1.selectbox("Weg-Spalte", spalten, spalten.index(w_g))
        kraft_sp = k2.selectbox("Kraft-Spalte", spalten, spalten.index(k_g))
        faktor_F = {"N": 1.0, "kN": 1000.0}[k3.radio("Krafteinheit", ["N", "kN"], horizontal=True,
                                                     index=1 if "kn" in kraft_sp.lower() else 0)]
        faktor_d = {"mm": 1.0, "µm": 1e-3}[k4.radio("Wegeinheit", ["mm", "µm"], horizontal=True)]
        k5, k6, k7 = st.columns(3)
        weg_art = k5.radio("Gemessener Weg", ["traverse", "mitte"], horizontal=True,
                           format_func=lambda s: "Traverse / Lastrollen" if s == "traverse"
                           else "Durchbiegung Probenmitte")
        C_m = k6.number_input("Maschinennachgiebigkeit [mm/N]", 0.0, 1.0, 0.0, 0.0005,
                              format="%.4f", disabled=weg_art != "traverse",
                              help="Wird vom Traversenweg abgezogen: d = d_mess − C·F")
        nullpunkt = k7.checkbox("Nullpunkt bei 5 % F_Bruch an Modell angleichen", True,
                                help="Entfernt Anlauf/Setzen am Kurvenanfang.")

    # --- Probentabelle --------------------------------------------------------
    ids = list(kurven.keys()) or ([] if ptab is None else ptab["ID"].tolist())
    if ptab is not None:
        ids += [i for i in ptab["ID"] if i not in ids]
    zeilen = []
    for pid in ids:
        b, h, Fb = b_nenn, h_nenn, np.nan
        if pid in kurven:
            F = kurven[pid][kraft_sp].to_numpy(float) * faktor_F
            Fb = float(np.nanmax(F))
        if ptab is not None and pid in set(ptab["ID"]):
            r = ptab[ptab["ID"] == pid].iloc[0]
            b = float(r.get("b [mm]", b)) if pd.notna(r.get("b [mm]", np.nan)) else b
            h = float(r.get("h [mm]", h)) if pd.notna(r.get("h [mm]", np.nan)) else h
            if pid not in kurven and pd.notna(r.get("F_Bruch [N]", np.nan)):
                Fb = float(r["F_Bruch [N]"])
        zeilen.append({"verwenden": True, "ID": pid, "b [mm]": b, "h [mm]": h,
                       "F_Bruch [N]": Fb, "Kurve": pid in kurven})
    st.subheader(f"Proben ({len(zeilen)})")
    st.caption("b, h und F_Bruch sind editierbar. F_Bruch = Maximum der Kraft-Weg-Kurve.")
    tabelle = st.data_editor(
        pd.DataFrame(zeilen), hide_index=True, key="probentabelle",
        disabled=["ID", "Kurve"],
        column_config={"b [mm]": st.column_config.NumberColumn(format="%.3f"),
                       "h [mm]": st.column_config.NumberColumn(format="%.4f"),
                       "F_Bruch [N]": st.column_config.NumberColumn(format="%.4f")})

    aktiv = tabelle[tabelle["verwenden"] & tabelle["F_Bruch [N]"].notna()
                    & (tabelle["F_Bruch [N]"] > 0)]
    signatur = (basis, n_el, weg_art, C_m, nullpunkt, faktor_F, faktor_d, weg_sp, kraft_sp,
                tuple(map(tuple, aktiv[["ID", "b [mm]", "h [mm]", "F_Bruch [N]"]].values)))

    if st.button(f"Berechnen ({len(aktiv)} Proben)", type="primary", disabled=aktiv.empty):
        erg = []
        bar = st.progress(0.0, "FE-Rechnung …")
        for k, (_, r) in enumerate(aktiv.iterrows()):
            a = replace(basis, b=float(r["b [mm]"]), h=float(r["h [mm]"]))
            e = fe_rechnung(a, float(r["F_Bruch [N]"]), n_el)
            erg.append({"ID": r["ID"], "aufbau": a, "fe": e, "F_Bruch": float(r["F_Bruch [N]"])})
            bar.progress((k + 1) / len(aktiv), f"FE-Rechnung {r['ID']}")
        bar.empty()
        st.session_state["ergebnisse"] = erg
        st.session_state["signatur"] = signatur
        st.success("Fertig. Ergebnisse in den Tabs 2 und 3.")

ergebnisse = st.session_state.get("ergebnisse")
veraltet = ergebnisse is not None and st.session_state.get("signatur") != signatur


def messkurve(pid, aufbau, e):
    """Gemessene Kurve (bis Bruch), korrigiert; Rückgabe d, F, rel. RMS-Abweichung."""
    if pid not in kurven:
        return None
    df = kurven[pid]
    d = df[weg_sp].to_numpy(float) * faktor_d
    F = df[kraft_sp].to_numpy(float) * faktor_F
    i = int(np.nanargmax(F))
    d, F = d[:i + 1], F[:i + 1]
    if weg_art == "traverse" and C_m > 0:
        d = d - C_m * F
    F_b = F[-1]
    if nullpunkt:
        F_v = 0.05 * F_b
        j = int(np.argmax(F >= F_v))
        d_mess_v = np.interp(F_v, F[max(j - 1, 0):j + 1], d[max(j - 1, 0):j + 1]) if j > 0 else d[j]
        d = d - (d_mess_v - e.weg_bei(F_v, weg_art))
    sel = F > 0.05 * F_b
    F_mod = e.kraft_bei_weg(d[sel], weg_art)
    ok = np.isfinite(F_mod)
    rms = np.sqrt(np.mean((F[sel][ok] - F_mod[ok]) ** 2)) / F_b if ok.any() else np.nan
    return d, F, rms


def ergebnis_tabelle():
    zeilen = []
    for r in ergebnisse:
        a, e, Fb = r["aufbau"], r["fe"], r["F_Bruch"]
        s_lin = float(a.sigma_linear(Fb))
        erreicht = e.F[-1] >= Fb * (1 - 1e-6)
        s_fe = float(e.sigma[-1]) if erreicht else np.nan
        mk = messkurve(r["ID"], a, e)
        zeilen.append({
            "ID": r["ID"], "b [mm]": a.b, "h [mm]": a.h, "F_Bruch [N]": Fb,
            "d_Last [mm]": e.d[-1], "w0 [mm]": e.w0[-1], "w0/L_s [%]": 100 * e.w0[-1] / a.Ls,
            "σ_linear [MPa]": s_lin, "σ_FE [MPa]": s_fe,
            "Δ [%]": 100 * (s_fe / s_lin - 1),
            "Kurvenabw. [%]": 100 * mk[2] if mk else np.nan,
            "Hinweis": e.warnung,
        })
    return pd.DataFrame(zeilen)


# ---------------------------------------------------------------------------
# Tab 2: Ergebnisse je Probe
# ---------------------------------------------------------------------------
with tab_probe:
    if ergebnisse is None:
        st.info("Zuerst in Tab 1 berechnen.")
    else:
        if veraltet:
            st.warning("Eingaben geändert - in Tab 1 neu berechnen.")
        df_e = ergebnis_tabelle()
        st.dataframe(df_e, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.3f" if c in ("b [mm]", "F_Bruch [N]", "d_Last [mm]", "w0 [mm]")
                                             else ("%.4f" if c == "h [mm]" else "%.1f"))
            for c in df_e.columns if c not in ("ID", "Hinweis")})
        st.download_button("Ergebnistabelle (CSV)", de_csv(df_e), "ergebnisse_4pb.csv", "text/csv")
        if df_e["Hinweis"].astype(bool).any():
            st.warning("Bei einigen Proben gibt es Hinweise (Spalte 'Hinweis').")

        pid = st.selectbox("Probe", [r["ID"] for r in ergebnisse])
        r = next(x for x in ergebnisse if x["ID"] == pid)
        a, e, Fb = r["aufbau"], r["fe"], r["F_Bruch"]
        cl, cm = a.nachgiebigkeit_linear()
        c_lin = cl if weg_art == "traverse" else cm
        y_fe = e.d if weg_art == "traverse" else e.w0
        wlab = "Lastrollenweg d [mm]" if weg_art == "traverse" else "Mittendurchbiegung w₀ [mm]"

        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure()
            mk = messkurve(pid, a, e)
            if mk:
                fig.add_scatter(x=mk[0], y=mk[1], name="Messung (korrigiert)", mode="lines",
                                line=dict(color=FARBE["mess"], width=2))
            fig.add_scatter(x=y_fe, y=e.F, name="FE nichtlinear", mode="lines",
                            line=dict(color=FARBE["fe"], width=2))
            fig.add_scatter(x=[0, c_lin * Fb], y=[0, Fb], name="lineare Theorie", mode="lines",
                            line=dict(color=FARBE["lin"], width=2, dash="dash"))
            st.plotly_chart(layout(fig, wlab, "Kraft F [N]"))
            if mk:
                st.caption(f"Rel. RMS-Abweichung Messung/FE: {100 * mk[2]:.2f} % von F_Bruch. "
                           "Größere Werte: E, h oder Maschinennachgiebigkeit prüfen.")
        with g2:
            fig = go.Figure()
            fig.add_scatter(x=e.F, y=e.sigma, name="σ FE nichtlinear", mode="lines",
                            line=dict(color=FARBE["fe"], width=2))
            fig.add_scatter(x=e.F, y=e.sigma_lin, name="σ linear", mode="lines",
                            line=dict(color=FARBE["lin"], width=2, dash="dash"))
            if e.F[-1] >= Fb * (1 - 1e-6):
                fig.add_scatter(x=[Fb, Fb], y=[e.sigma[-1], a.sigma_linear(Fb)], mode="markers",
                                name="Bruch", marker=dict(size=10, color=[FARBE["fe"], FARBE["lin"]],
                                                          line=dict(width=2, color="white")))
            st.plotly_chart(layout(fig, "Kraft F [N]", "max. Zugspannung σ [MPa]"))

        # Verformte Probe in wahrem Maßstab
        x, y, sk = verformung(a, e.zustaende[-1], n_el, n_el)
        tx, ty = np.gradient(x), np.gradient(y)
        nrm = np.hypot(tx, ty)
        nx, ny = -ty / nrm, tx / nrm
        if a.Lp and a.Lp / 2 > e.s2[-1]:
            ue = a.Lp / 2 - e.s2[-1]
            x = np.append(x, x[-1] + ue * tx[-1] / nrm[-1]); y = np.append(y, y[-1] + ue * ty[-1] / nrm[-1])
            nx = np.append(nx, nx[-1]); ny = np.append(ny, ny[-1]); sk = np.append(sk, 0.0)
        fig = go.Figure()
        for sgn in (1, -1):
            xo = np.concatenate([x + a.h / 2 * nx, (x - a.h / 2 * nx)[::-1]])
            yo = np.concatenate([y + a.h / 2 * ny, (y - a.h / 2 * ny)[::-1]])
            fig.add_scatter(x=sgn * xo, y=yo, fill="toself", mode="lines", showlegend=sgn == 1,
                            name="Probe", line=dict(color=FARBE["fe"], width=1),
                            fillcolor="rgba(42,120,214,0.35)", hoverinfo="skip")
        phi = np.linspace(0, 2 * np.pi, 80)
        for (cx, cy, R, nm) in [(a.Li / 2, a.h / 2 + a.Rl - e.d[-1], a.Rl, "Lastrolle"),
                                (a.Ls / 2, -a.h / 2 - a.Rs, a.Rs, "Stützrolle")]:
            for sgn in (1, -1):
                fig.add_scatter(x=sgn * cx + max(R, 1e-3) * np.cos(phi), y=cy + max(R, 1e-3) * np.sin(phi),
                                fill="toself", mode="lines", name=nm, showlegend=sgn == 1,
                                line=dict(color=FARBE["rolle"], width=1),
                                fillcolor="rgba(138,137,133,0.3)", hoverinfo="skip")
        layout(fig, "x [mm]", "y [mm]", 330)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(hovermode=False)
        st.markdown(f"**Verformung bei Bruch (maßstäblich)**, w₀ = {e.w0[-1]:.2f} mm")
        st.plotly_chart(fig)

        fig = go.Figure()
        xs = np.concatenate([-x[::-1], x])
        fig.add_scatter(x=xs, y=np.concatenate([sk[::-1], sk]), mode="lines", name="FE",
                        line=dict(color=FARBE["fe"], width=2))
        st.plotly_chart(layout(fig, "x [mm] (verformte Lage)", "Zugseitige Randspannung [MPa]", 260))
        st.caption(f"Kontaktpunkte (materiell, ab Mitte): Lastrolle s₁ = {e.s1[-1]:.3f} mm "
                   f"(nominell {a.Li / 2:.3f}), Stützrolle s₂ = {e.s2[-1]:.3f} mm (nominell {a.Ls / 2:.3f}). "
                   f"Max. Spannung bei x = {e.x_sigma[-1]:.2f} mm.")

# ---------------------------------------------------------------------------
# Tab 3: Weibull
# ---------------------------------------------------------------------------
with tab_weibull:
    if ergebnisse is None:
        st.info("Zuerst in Tab 1 berechnen.")
    else:
        if veraltet:
            st.warning("Eingaben geändert - in Tab 1 neu berechnen.")
        df_e = ergebnis_tabelle()
        niveau = st.select_slider("Konfidenzniveau", [0.80, 0.90, 0.95, 0.99], 0.90,
                                  format_func=lambda v: f"{100 * v:.0f} %")
        ok = df_e["σ_FE [MPa]"].notna()
        if (~ok).any():
            st.warning(f"{(~ok).sum()} Probe(n) ohne gültige FE-Spannung ausgeschlossen: "
                       + ", ".join(df_e.loc[~ok, "ID"]))
        try:
            w_lin = weibull.auswerten(df_e.loc[ok, "σ_linear [MPa]"], niveau)
            w_fe = weibull.auswerten(df_e.loc[ok, "σ_FE [MPa]"], niveau)
        except ValueError as ex:
            st.error(str(ex))
            st.stop()

        p = int(round(100 * niveau))
        tab_p = pd.DataFrame({
            "Kenngröße": ["N", "m (ML)", "m erwartungstreu", f"m {p}%-KI", "σ₀ (ML) [MPa]",
                          f"σ₀ {p}%-KI [MPa]", "Mittelwert [MPa]"],
            "linear": [str(w_lin.N), f"{w_lin.m:.2f}", f"{w_lin.m_korr:.2f}",
                       f"{w_lin.m_ki[0]:.2f} … {w_lin.m_ki[1]:.2f}", f"{w_lin.s0:.1f}",
                       f"{w_lin.s0_ki[0]:.1f} … {w_lin.s0_ki[1]:.1f}", f"{w_lin.mittelwert():.1f}"],
            "FE nichtlinear": [str(w_fe.N), f"{w_fe.m:.2f}", f"{w_fe.m_korr:.2f}",
                               f"{w_fe.m_ki[0]:.2f} … {w_fe.m_ki[1]:.2f}", f"{w_fe.s0:.1f}",
                               f"{w_fe.s0_ki[0]:.1f} … {w_fe.s0_ki[1]:.1f}", f"{w_fe.mittelwert():.1f}"],
        })
        c1, c2 = st.columns([2, 3])
        with c1:
            st.dataframe(tab_p, hide_index=True)
            st.caption("ML-Schätzung nach DIN EN 843-5 / ASTM C1239; Korrekturfaktor und "
                       "Konfidenzintervalle über pivotale Größen (Monte-Carlo, 20 000 Stichproben).")
        with c2:
            fig = go.Figure()
            yt = lambda F: np.log(-np.log(1 - np.asarray(F)))
            for w, nm, farbe, sym in ((w_fe, "FE nichtlinear", FARBE["fe"], "circle"),
                                      (w_lin, "linear", FARBE["lin"], "diamond")):
                fig.add_scatter(x=w.sigma_sortiert, y=yt(w.F_i), mode="markers", name=nm,
                                marker=dict(color=farbe, size=9, symbol=sym,
                                            line=dict(width=1.5, color="white")),
                                customdata=100 * w.F_i,
                                hovertemplate="σ = %{x:.1f} MPa<br>P_f = %{customdata:.1f} %<extra>" + nm + "</extra>")
                ss = np.geomspace(w.sigma_sortiert[0] * 0.9, w.sigma_sortiert[-1] * 1.1, 50)
                fig.add_scatter(x=ss, y=yt(w.F_modell(ss)), mode="lines", showlegend=False,
                                line=dict(color=farbe, width=2), hoverinfo="skip")
            pv = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.632, 0.9, 0.99])
            layout(fig, "Bruchspannung σ [MPa]", "Versagenswahrscheinlichkeit P_f", 460)
            fig.update_xaxes(type="log")
            fig.update_yaxes(tickvals=yt(pv), ticktext=[f"{100 * v:g} %" for v in pv],
                             range=[yt(min(0.005, 0.5 / w_fe.N)), yt(0.995)])
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig)

        exp = df_e.loc[ok, ["ID", "h [mm]", "F_Bruch [N]", "σ_linear [MPa]", "σ_FE [MPa]", "Δ [%]"]]
        st.download_button("Festigkeitswerte (CSV)", de_csv(exp), "festigkeiten_4pb.csv", "text/csv")

# ---------------------------------------------------------------------------
# Tab 4: Methode
# ---------------------------------------------------------------------------
with tab_methode:
    st.markdown(r"""
**Modell.** Halbmodell der Probe (Symmetrie in der Mitte) aus korotationalen 2D-Balkenelementen
(Euler-Bernoulli lokal, beliebig große Verschiebungen und Rotationen, kleine Dehnungen,
linear-elastisch). Gelöst wird wegegesteuert (Lastrollenposition) mit Newton-Raphson und
konsistenter Tangente.

**Kontakt.** Die Rollen sind starre Zylinder (Radius $R$). Im Kontaktpunkt berührt die Rolle die
Probenoberfläche tangential: der Rollenmittelpunkt liegt auf der Normalen der Balkenachse im Abstand
$R + h/2$. Die Kontaktpunkte wandern mit der Verformung über die Probe; ihre materiellen Koordinaten
$s_1$ (Lastrolle) und $s_2$ (Stützrolle) sind Unbekannte des Gleichungssystems (die Elementlängen der
beiden Abschnitte skalieren mit $s_1$, $s_2$). Damit liegt der Kontakt immer exakt auf einem Knoten -
die Kraft-Weg-Kurve ist glatt, ohne Knotensprünge. Die Normalkraft wirkt senkrecht zur Oberfläche, also
mit zunehmender Rotation zunehmend horizontal.

**Reibung** (fest stehende Rollen, $\mu > 0$): Coulomb mit Haften und Gleiten. Der Schlupf zwischen Probe und
Rolle enthält den Abrollanteil: Stützrolle $\Delta g = c\,\Delta s_2 + R_s\Delta\theta_2$, Lastrolle
$\Delta g = c\,\Delta s_1 - R_l\Delta\theta_1$ ($c = 1+\varepsilon \pm \kappa h/2$, Oberflächen-Bogenlänge).
Haften: $\Delta g = 0$, $|f_t| \le \mu F_n$; Gleiten: $f_t = \mu F_n\,\mathrm{sign}(\Delta g)$. Die
Tangentialkraft greift an der Oberfläche an (Versatzmoment $f_t h/2$). Bei großen Rollenradien kehrt der
Abrollanteil die Gleitrichtung gegenüber einer reinen Betrachtung von $\Delta s$ um.

**Spannung.** Maximale zugseitige Randspannung $\sigma = N/A + |M|/W$ über die ganze Probe,
$W = b h^2/6$. Referenz: lineare Theorie $\sigma_\mathrm{lin} = 3F(L_s-L_i)/(2bh^2)$.

**Typisches Verhalten.** Bei mäßiger Durchbiegung ist $\sigma_\mathrm{FE} < \sigma_\mathrm{lin}$ (Kontaktpunkte
wandern nach innen bzw. außen → kürzerer Hebelarm). Nahe dem Kraftmaximum des Aufbaus (Probe wird durch
die Stützen gezogen) steigt die Spannung bei kaum steigender Kraft → $\sigma_\mathrm{FE} > \sigma_\mathrm{lin}$.

**Verifikation** (`test_validierung.py`): konsistente Tangente, linearer Grenzfall (Spannung, Lastrollenweg,
Mittendurchbiegung < 1e-5), Netzkonvergenz $O(h_e^2)$, unabhängige Lösung des Elastica-Randwertproblems
(Shooting, DOP853) mit und ohne Reibung: Abweichung < 3e-5 bei 20 Elementen je Abschnitt.

**Vergleich mit ANSYS** (221 Schalenmodelle SHELL93, 3D-Viertelmodell, Batches 8-12): Kraft-Weg bis zum
Kraftmaximum Median 2 % (90 %-Quantil 5 %) mit ebenem Dehnungszustand. Spannung in Probenmitte bei gleicher Kraft
< 1 % (kleine) bis 3 % (sehr große Durchbiegung). Die globale Maximalspannung im 3D-Modell liegt am freien
Längsrand 3-8 % über dem Balkenwert (Randeffekt der Platte, im Balkenmodell nicht abbildbar).

**Nicht enthalten.** Randüberhöhung am freien Längsrand breiter Platten, Hertzsche Kontaktspannungen und Eindrückung an den Rollen, Schubverformung
(bei $L/h > 20$ vernachlässigbar), Querkrümmung breiter Platten (antiklastisch; nur grob über
ebenen Dehnungszustand), Materialnichtlinearität. Bei dicken Proben ($L_s/h < 10$) ist ein Balkenmodell
generell ungenau.

**Weibull.** ML-Schätzung (DIN EN 843-5, ASTM C1239), $P_{f,i} = (i-0.5)/N$. Erwartungstreuer Faktor
$b_N$ und Konfidenzintervalle über die pivotalen Größen $\hat m/m$ und $\hat m \ln(\hat\sigma_0/\sigma_0)$,
per Monte-Carlo für das jeweilige $N$ (gleiches Prinzip wie die Normtabellen).
""")
