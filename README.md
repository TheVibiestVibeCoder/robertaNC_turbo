# 🧠 Narrative Intelligence Platform - Enhanced Version

Ein fortschrittliches System zur Analyse von Nachrichtendaten mit semantischem Clustering, Multi-dimensionaler Sentiment-Analyse und intelligenter Narrative-Erkennung.

## 🚀 Features

- **Dynamic Semantic Clustering**: Intelligente Gruppierung ähnlicher Artikel
- **Multi-Dimensional Embeddings**: Verschiedene Embedding-Strategien für optimale Analyse
- **Advanced Sentiment & Emotion Analysis**: Sentiment, Emotion und Financial Sentiment
- **Context-Aware Keyword Extraction**: Dynamische und temporale Schlüsselwort-Erkennung
- **Interactive Dashboard**: Plotly-basierte Visualisierungen
- **Comprehensive Intelligence Reports**: Detaillierte Analyseberichte
- **Actor Network Analysis**: Journalisten- und Akteurs-Netzwerk-Analyse

## 📋 Voraussetzungen

### Python Pakete
```bash
pip install pandas numpy matplotlib seaborn
pip install transformers sentence-transformers torch
pip install sklearn scikit-learn
pip install spacy plotly
pip install networkx hdbscan
```

### Optionale Pakete (für erweiterte Features)
```bash
pip install umap-learn textstat yake keybert textblob
pip install python-louvain python-igraph
```

### Spacy Modell
```bash
python -m spacy download en_core_web_sm
```

## 🎯 Schnellstart

### Basis-Analyse
```bash
python robertaNC_turbo.py --csv "pfad/zu/deinen/daten.csv"
```

### Kleine Test-Analyse (schnell)
```bash
python robertaNC_turbo.py --csv "C:\Users\schwi\OneDrive\Desktop\bloomberg_news_1000.csv" --sample-size 50
```

### Analyse ohne Dashboard (für Entwicklung)
```bash
python robertaNC_turbo.py --csv "C:\Users\schwi\OneDrive\Desktop\bloomberg_news_1000.csv" --sample-size 50 --skip-dashboard
```

## 🔧 Command Line Parameter

### Erforderlich
- `--csv`: Pfad zur CSV-Datei mit den Nachrichtendaten

### Optional
- `--sample-size`: Maximale Anzahl Artikel für Analyse (Standard: 5000)
- `--min-cluster-size`: Minimale Cluster-Größe (Standard: 5)
- `--temporal-window-days`: Zeitfenster für temporales Clustering (Standard: 7)
- `--topic`: Keyword-Filter für Artikel (kann mehrfach verwendet werden)
- `--skip-dashboard`: Dashboard-Erstellung überspringen (schneller für Tests)

## 📊 Beispiel-Commands

### Schnelle Entwicklungs-Tests
```bash
# Sehr kleine Analyse für schnelle Tests
python robertaNC_turbo.py --csv "bloomberg_news_1000.csv" --sample-size 20 --skip-dashboard

# Kleine Analyse mit Dashboard
python robertaNC_turbo.py --csv "bloomberg_news_1000.csv" --sample-size 100

# Mittelgroße Analyse
python robertaNC_turbo.py --csv "bloomberg_news_1000.csv" --sample-size 500
```

### Produktive Analysen
```bash
# Vollständige Analyse mit allen Features
python robertaNC_turbo.py --csv "bloomberg_news_1000.csv"

# Große Analyse mit benutzerdefinierten Parametern
python robertaNC_turbo.py --csv "bloomberg_news_1000.csv" --sample-size 2000 --min-cluster-size 3

# Temporale Analyse mit kürzerem Zeitfenster
python robertaNC_turbo.py --csv "bloomberg_news_1000.csv" --temporal-window-days 3
```

### Topic-spezifische Analysen
```bash
# Nur Artikel zu einem bestimmten Thema
python robertaNC_turbo.py --csv "bloomberg_news_1000.csv" --topic "Tesla"

# Mehrere Keywords
python robertaNC_turbo.py --csv "bloomberg_news_1000.csv" --topic "Tesla" "Stock" "Market"

# Themen-Analyse mit kleiner Stichprobe
python robertaNC_turbo.py --csv "bloomberg_news_1000.csv" --topic "AI" --sample-size 200
```

## 📁 CSV-Format Anforderungen

Deine CSV-Datei sollte mindestens diese Spalten enthalten:
- `Date`: Datum der Artikel (YYYY-MM-DD Format)
- `Full_Text`: Volltext der Artikel
- `Headline`: Schlagzeilen (optional)
- `Journalists_List`: Liste der Journalisten (optional)

## 📈 Output-Dateien

Nach der Analyse werden folgende Dateien erstellt:

### Visualisierungen
- `enhanced_narrative_dashboard.html`: Interaktives Dashboard

### Daten-Exports
- `enhanced_analysis_export.csv`: Hauptanalysedaten
- `cluster_analysis.csv`: Cluster-Informationen
- `actor_influence_export.csv`: Akteurs-Einfluss-Daten
- `narrative_summaries.csv`: Narrative Zusammenfassungen
- `keyword_analysis.csv`: Schlüsselwort-Analyse

## 🏃‍♂️ Analyse-Phasen

Das System durchläuft 9 Phasen:

1. **Data Loading & Preprocessing**: Daten laden und bereinigen
2. **Multi-Dimensional Embeddings**: Verschiedene Embedding-Strategien
3. **Dynamic Semantic Clustering**: Intelligentes Clustering
4. **Dynamic Keywords**: Kontext-bewusste Schlüsselwort-Extraktion
5. **Multi-Dimensional Sentiment**: Sentiment, Emotion, Financial Sentiment
6. **Enhanced Narrative Summaries**: Narrative Zusammenfassungen
7. **Enhanced Dashboard**: Interaktive Visualisierungen
8. **Situation Report**: Executive Briefing
9. **Enhanced Export**: Datenexport

## 🛠️ Entwicklung & Debugging

### Schnelle Test-Zyklen
```bash
# Minimaler Test (5-10 Sekunden)
python robertaNC_turbo.py --csv "daten.csv" --sample-size 10 --skip-dashboard

# Funktions-Test (30-60 Sekunden)
python robertaNC_turbo.py --csv "daten.csv" --sample-size 50 --skip-dashboard

# Vollständiger Test mit Dashboard (2-5 Minuten)
python robertaNC_turbo.py --csv "daten.csv" --sample-size 100
```

### Performance-Optimierung
- Für schnelle Tests: `--sample-size 20-50` und `--skip-dashboard`
- Für Entwicklung: `--sample-size 100-200`
- Für Produktion: Ohne `sample-size` Begrenzung

## 🔍 Dashboard anpassen

Um das Dashboard zu überspringen und eigene Visualisierungen zu erstellen:

1. **Analyse ohne Dashboard ausführen:**
   ```bash
   python robertaNC_turbo.py --csv "daten.csv" --skip-dashboard
   ```

2. **Exportierte Daten verwenden:**
   - Lade `enhanced_analysis_export.csv` für eigene Visualisierungen
   - Verwende die anderen CSV-Exports je nach Bedarf

3. **Eigenes Dashboard-Script erstellen:**
   ```python
   import pandas as pd
   import plotly.graph_objects as go
   
   # Analysedaten laden
   df = pd.read_csv("enhanced_analysis_export.csv")
   
   # Eigene Visualisierungen erstellen
   # ...
   ```

## ❓ Troubleshooting

### Häufige Probleme

**ModuleNotFoundError:**
```bash
pip install [fehlendes_paket]
```

**Memory-Probleme bei großen Datensätzen:**
```bash
# Kleinere Sample-Size verwenden
python robertaNC_turbo.py --csv "daten.csv" --sample-size 1000
```

**Slow Performance:**
```bash
# Dashboard überspringen für schnellere Tests
python robertaNC_turbo.py --csv "daten.csv" --skip-dashboard
```

### Logs und Debugging

Das System gibt detaillierte Fortschrittsmeldungen aus:
- 🔥 **PHASE X**: Aktuelle Analysephase
- ✅ **Completed**: Erfolgreich abgeschlossene Schritte  
- ⚠️ **Warning**: Optionale Features nicht verfügbar
- ❌ **Error**: Kritische Fehler

## 🎯 Empfohlene Workflows

### Neue Daten erkunden
```bash
python robertaNC_turbo.py --csv "neue_daten.csv" --sample-size 100 --skip-dashboard
```

### Vollständige Analyse
```bash
python robertaNC_turbo.py --csv "daten.csv" --sample-size 2000
```

### Themen-spezifische Deep-Dive
```bash
python robertaNC_turbo.py --csv "daten.csv" --topic "KI" "OpenAI" --sample-size 500
```

---

## 🏆 Pro-Tips

1. **Entwicklung**: Immer mit `--skip-dashboard` und kleiner `--sample-size` beginnen
2. **Testing**: Verwende 50-100 Artikel für Funktions-Tests
3. **Produktion**: Ohne Limits für vollständige Insights
4. **Custom Dashboards**: Nutze die CSV-Exports für eigene Visualisierungen
5. **Performance**: Bei großen Datensätzen schrittweise die Sample-Size erhöhen