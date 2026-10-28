# 4PB nichtlinear

Festigkeitsauswertung der 4-Punkt-Biegung bei großer Durchbiegung (dünne Proben):
geometrisch nichtlineares Balken-FE-Modell mit wandernden Rollenkontakten, Abgleich mit
gemessenen Kraft-Weg-Kurven, Weibull-Auswertung (linear vs. nichtlinear).

## Start

```
pip install -r requirements.txt
streamlit run app.py
```

Browser öffnet sich unter http://localhost:8501. „Beispieldaten laden“ zeigt den Ablauf
mit 20 synthetischen Glasproben (h ≈ 0,4 mm, wahre Weibull-Parameter m = 8, σ₀ = 520 MPa).

## Web-Version (ohne Installation)

`index.html` startet dieselbe App im Browser (stlite: Streamlit auf Pyodide/WebAssembly).
Python läuft auf dem Rechner des Nutzers; hochgeladene Messdaten verlassen den Browser nicht.
Ein Server ist nicht nötig, jedes statische Hosting reicht (GitHub Pages, Institutsserver).

GitHub Pages:
1. Ordnerinhalt in ein Repository legen (Root: `index.html`, `*.py`, `beispieldaten/`, `.nojekyll`).
2. Settings → Pages → Deploy from branch → `main` / `(root)`.
3. Aufruf `https://<user>.github.io/<repo>/`. Erster Start ca. 20-60 s (Pyodide-Download, danach im Cache).

Nach Änderungen an den `.py`-Dateien in `index.html` den Wert `?v=` hochzählen.
Lokal testen: `python -m http.server` im Ordner, dann http://localhost:8000.

## Eingabe

* **Seitenleiste:** E, ν, Nennmaße b, h, Probenlänge, Stützweite, Lastweite, Rollenradien,
  Reibkoeffizient (0 = frei rollende Rollen), Elementanzahl.
* **Kurven:** eine Datei je Probe (CSV/TXT). Trenner (`;` Tab `,` Leerzeichen),
  Dezimalkomma, Kopf- und Einheitenzeilen werden erkannt. Dateiname = Proben-ID.
  Weg als Traverse/Lastrollenweg oder als Mittendurchbiegung.
  Bruchkraft = Kraftmaximum der Kurve.
* **Probentabelle (optional):** `ID; b; h; F_Bruch` – gemessene Maße je Probe. Ohne Kurven
  genügt eine Tabelle mit Bruchkräften.

## Dateien

| Datei | Inhalt |
|---|---|
| `vierpunkt_fe.py` | FE-Solver (ohne GUI nutzbar: `berechne(Aufbau(...), F_ziel=...)`) |
| `weibull.py` | ML-Schätzung, Korrekturfaktor, Konfidenzintervalle |
| `datenimport.py` | Einlesen von Maschinen-Exporten |
| `app.py` | Streamlit-Oberfläche |
| `test_validierung.py` | Verifikation (Tangente, linearer Grenzfall, Netzkonvergenz, Elastica-Referenz) |
| `beispieldaten_erzeugen.py` | erzeugt `beispieldaten/` |
| `vergleich_ansys.py` | Nachrechnung der ANSYS-Batches: `python vergleich_ansys.py <Ordner mit Batch*_all_data.csv>` |

## Skript-Nutzung

```python
from vierpunkt_fe import Aufbau, berechne
a = Aufbau(E=72000, b=4.0, h=0.4, Ls=40, Li=20, Rs=2.5, Rl=2.5, mu=0.0)
e = berechne(a, F_ziel=10.0)          # N
print(e.sigma[-1], a.sigma_linear(10.0), e.w0[-1])
```

Einheiten: mm, N, MPa. Modellannahmen und Grenzen: Tab „Methode“ in der App.
