# Nexia AI NSFW Server

Dieser Dienst stellt die mehragentige Bildprüfung für die Nexia-App
bereit.  Er basiert auf Flask und dem bestehenden PyTorch-Modell und
bietet parallele Verarbeitung mit bis zu vier gleichzeitigen Agenten.

## Voraussetzungen

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Starten

```bash
python app.py
```

Der Server lauscht standardmäßig auf Port `7535` und ist über alle
Netzwerkinterfaces erreichbar (`0.0.0.0`).

## Endpunkte

- `GET /status` – Gibt eine menschenlesbare Statusmeldung mit aktiven
  Agenten, Warteschlange und Fehleranzahl zurück.
- `GET /models` – Liefert alle gefundenen `.pth`-Modelle im
  `models/`-Ordner.
- `GET/POST /agents` – Abfragen oder setzen Sie die aktive Agentenzahl
  (1 bis 4).
- `POST /scan` – Fügt ein neues Bild zur Warteschlange hinzu.  Das Bild
  wird Base64-kodiert übertragen (`{"image": "..."}`), optional mit
  `model`-Slug und eigener `id`.
- `GET /results/<job_id>` – Fragt das Ergebnis eines einzelnen Jobs ab.

Ergebnisse werden zusätzlich im Ordner `processed/` abgelegt, so dass
bestehende Integrationen weiterhin funktionieren.  Fehlerhafte Dateien
landen im Ordner `errors/`.
