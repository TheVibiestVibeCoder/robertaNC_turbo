Clustering Logik wurde bei diesem Script auch geändert, für das original clustering, das nicht auf die 50 runtersampled, zu V4 zurückgehen



**Hauptänderungen in der Clustering-Pipeline:**

1. **Dimensionsreduktion vor Clustering**: 
   - **Neu**: Reduziere Embeddings von 384 auf 50 Dimensionen mit UMAP/PCA
   - **Alt**: Direkt auf 384-dimensionalen Embeddings clustern
   - **Grund**: HDBSCAN funktioniert viel besser auf reduzierten Dimensionen

2. **Metric-Wechsel**:
   - **Neu**: `metric='euclidean'` auf den reduzierten Embeddings
   - **Alt**: `metric='cosine'` auf den ursprünglichen Embeddings
   - **Grund**: Nach Dimensionsreduktion ist Euclidean oft stabiler

3. **Normalisierte Embeddings**:
   - **Neu**: `normalize_embeddings=True` beim Encoding
   - **Alt**: Unnormalisierte Embeddings
   - **Grund**: Notwendig für korrekte Cosine-Similarity-Berechnungen

4. **Angepasste HDBSCAN-Parameter**:
   ```python
   # Neu:
   clusterer = HDBSCAN(
       min_cluster_size=min_cluster_size,
       metric='euclidean',  # Geändert von 'cosine'
       min_samples=5,       # Erhöht von 3
       cluster_selection_epsilon=0.1,  # Reduziert von 0.15
       cluster_selection_method='eom'
   )
   ```

**Warum diese Änderungen?**

Die ursprüngliche Logik hatte das Problem, dass HDBSCAN mit Cosine-Metric auf hochdimensionalen, unnormalisierten Embeddings instabile Ergebnisse produzierte - zu viele "Noise"-Artikel und aufgeteilte Themen. Die neue Pipeline:

1. Normalisiert erst die Embeddings
2. Reduziert dann die Dimensionen 
3. Clustert auf den reduzierten Dimensionen mit Euclidean-Metric

Das führt zu deutlich stabileren und kohärenteren Clustern. Die semantische Qualität bleibt erhalten, da die Dimensionsreduktion die wichtigsten semantischen Beziehungen bewahrt, aber die "Curse of Dimensionality"-Probleme für HDBSCAN eliminiert.