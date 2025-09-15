 It will run on small corpora, yet quality and clustering robustness are held back by a few fixable issues.

### What to expect “right now”

* **Narratives**: often fluent but inconsistent. Sampling + missing T5 task prefix + overlong prompts → drift and hallucinations in cluster stories.
* **Clustering**: works, but noisier than necessary. Cosine HDBSCAN on *unnormalized* 384-d MiniLM vectors, without prior dim-reduction, tends to over-label noise and split themes.
* **Speed/scale**: fine for \~1–2k articles on CPU/GPU. Beyond that, HDBSCAN distance computations and per-cluster TF-IDF fits will bite. Visual annotation can choke for large clusters.

### Top 8 changes to make it strong (minimal edits, big gains)

1. **Import the right HDBSCAN**

```python
from hdbscan import HDBSCAN
```

2. **Normalize embeddings (crucial for cosine)**

```python
self.embeddings = self.sentence_model.encode(
    texts, batch_size=64, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True
)
```

3. **Reduce dimensions before HDBSCAN**
   UMAP(50) or PCA(50) → HDBSCAN on 50-d; big stability and speed boost.

```python
X = (umap.UMAP(n_components=50, random_state=42).fit_transform(self.embeddings)
     if UMAP_AVAILABLE else PCA(n_components=50, random_state=42).fit_transform(self.embeddings))
clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', min_samples=5)
self.clusters = clusterer.fit_predict(X)
```

4. **Fix T5 usage for summaries**
   Use a task prefix and deterministic decoding.

```python
self.summarizer = pipeline("summarization", model="t5-base",
                           device=0 if self.device=='cuda' else -1)
summary = self.summarizer(
    "summarize: " + chunk, max_length=220, min_length=90,
    num_beams=4, do_sample=False, no_repeat_ngram_size=3
)[0]["summary_text"]
```

5. **Map-reduce long cluster text**
   Your current prompt can exceed T5 limits. Chunk to \~300–400 words, summarize each, then summarize the concatenation.
6. **Sparse TF-IDF averaging**
   Avoid `toarray()` for large clusters.

```python
avg = tfidf_matrix.mean(axis=0).A1
```

7. **Better token pattern for news terms**
   Capture hyphenated/slash terms.

```python
token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-\/&]*\b"
```

8. **Logging and determinism**
   Fix the “T5-large loaded” message, and set seeds for repeatability:

```python
import numpy as np, torch, random
seed=42; np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
```

### Bottom line performance

* **Quality after fixes**: noticeably tighter clusters, fewer orphans, narratives track frames better and drift less.
* **Throughput**: with PCA-50 + normalized embeddings, HDBSCAN runtime drops a lot; 5–10k articles becomes feasible on a decent CPU, comfortably fast on GPU.
* **Stability**: beam search + chunking removes most summary weirdness.

If you want, I can inline those edits into your file in one pass.
