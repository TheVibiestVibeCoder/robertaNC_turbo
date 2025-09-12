#!/usr/bin/env python3
"""
NARRATIVE INTELLIGENCE PLATFORM - ENHANCED VERSION (FIXED IMPORTS)
Advanced narrative analysis with dynamic semantic clustering, multi-dimensional actor networks,
and adaptive narrative evolution tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# Advanced dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️ UMAP not available - will use PCA instead")

import spacy
from collections import Counter, defaultdict, deque
import re
import ast

# Text analysis libraries
try:
    from textstat import flesch_reading_ease, dale_chall_readability_score
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("⚠️ textstat not available - will skip readability metrics")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import networkx as nx
try:
    from community import community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("⚠️ python-louvain not available - will skip community detection")

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False
    print("⚠️ python-igraph not available - will use networkx only")

# Advanced NLP libraries
try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    print("⚠️ YAKE not available - will skip advanced keyword extraction")

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    print("⚠️ KeyBERT not available - will use basic keyword extraction")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️ TextBlob not available - will skip some text features")

class AdvancedNarrativeIntelligence:
    """
    Enhanced Narrative Intelligence Platform with:
    - Dynamic semantic clustering
    - Multi-dimensional actor analysis  
    - Narrative evolution tracking
    - Advanced semantic context extraction
    - Influence propagation modeling
    """
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.embeddings = None
        self.clusters = None

        # Enhanced data structures
        self.narrative_evolution = {}
        self.semantic_contexts = {}
        self.actor_influence_matrix = None
        self.narrative_networks = {}
        self.temporal_clusters = {}
        self.dynamic_keywords = {}
        self.narrative_genealogy = {}

        # Determine compute device once for use across models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_index = 0 if self.device == 'cuda' else -1
        
        print("🧠 Advanced Narrative Intelligence Platform NEW")
        print("=" * 60)
        self.setup_enhanced_models()
        
    def setup_enhanced_models(self):
        """Setup advanced NLP models and tools"""
        try:
            print("📦 Loading Advanced Model Suite...")
            
            # Primary embedding model
            print("  🧠 Loading Sentence-BERT (all-MiniLM-L6-v2)...")
            self.sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2', device=self.device
            )
            
            # Secondary embedding for semantic diversity
            print("  🔬 Loading domain-specific embedding model...")
            self.domain_model = SentenceTransformer(
                'sentence-transformers/all-mpnet-base-v2', device=self.device
            )
            
            # Advanced sentiment with emotion detection
            print("  🎭 Loading multi-emotion sentiment pipeline...")
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=self.device_index
            )
            
            # Financial sentiment (domain-specific)
            print("  💰 Loading financial sentiment analyzer...")
            try:
                self.financial_sentiment = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=self.device_index
                )
            except:
                print("    ⚠️ FinBERT not available, using generic sentiment")
                self.financial_sentiment = pipeline("sentiment-analysis")
            
            # Advanced keyword extraction
            print("  🔑 Setting up advanced keyword extractors...")
            if KEYBERT_AVAILABLE:
                self.keybert_model = KeyBERT(model=self.sentence_model)
            else:
                self.keybert_model = None
                
            if YAKE_AVAILABLE:
                self.yake_extractor = yake.KeywordExtractor(
                    lan="en", n=3, dedupLim=0.7, top=20
                )
            else:
                self.yake_extractor = None
            
            # Enhanced NER
            print("  🏷️ Loading enhanced NER pipeline...")
            try:
                self.nlp = spacy.load("en_core_web_lg")  # Large model for better accuracy
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    print("    ⚠️ spaCy model not found - NER will be limited")
                    self.nlp = None
            
            # Advanced TF-IDF with domain adaptation
            print("  📝 Setting up adaptive TF-IDF...")
            self.adaptive_tfidf = TfidfVectorizer(
                max_features=2000, 
                ngram_range=(1, 4),  # Extended n-grams
                min_df=2,
                max_df=0.85,
                sublinear_tf=True,
                use_idf=True
            )
            
            print("✅ Enhanced Model Suite Ready!")
            
        except Exception as e:
            print(f"❌ Error loading enhanced models: {e}")
            raise
    
    def load_and_preprocess_data(self, topic_filter=None):
        """Enhanced data preprocessing with optional topic filtering"""
        print("\n📊 Enhanced Data Preprocessing...")

        self.df = pd.read_csv(self.csv_path)
        print(f"  📈 {len(self.df)} articles loaded")
        
        # Parse dates
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Enhanced journalist parsing
        def parse_journalists(journalists_str):
            try:
                return ast.literal_eval(journalists_str) if isinstance(journalists_str, str) else [journalists_str]
            except:
                return [journalists_str] if journalists_str else []
        
        self.df['Journalists_List'] = self.df['Journalists'].apply(parse_journalists)

        # Enhanced text features
        self.df['Full_Text'] = self.df['Headline'] + " " + self.df['Article'].fillna("")
        self.df['Text_Length'] = self.df['Full_Text'].str.len()
        self.df['Word_Count'] = self.df['Full_Text'].str.split().str.len()
        self.df['Sentence_Count'] = self.df['Full_Text'].str.count(r'[.!?]+')
        
        # Readability metrics
        if TEXTSTAT_AVAILABLE:
            print("  📖 Computing readability metrics...")
            self.df['Readability_Flesch'] = self.df['Full_Text'].apply(
                lambda x: flesch_reading_ease(x) if len(x) > 10 else 0
            )
        else:
            print("  ⚠️ Skipping readability metrics (textstat not available)")
            self.df['Readability_Flesch'] = 50.0  # Default neutral score
        
        # Enhanced temporal features
        self.df['Hour'] = self.df['Date'].dt.hour
        self.df['DayOfWeek'] = self.df['Date'].dt.dayofweek
        self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6])
        self.df['Week'] = self.df['Date'].dt.isocalendar().week
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Quarter'] = self.df['Date'].dt.quarter
        
        # Market timing features (assuming business news)
        self.df['IsMarketHours'] = (
            (self.df['Hour'] >= 9) & (self.df['Hour'] <= 16) & (~self.df['IsWeekend'])
        )
        self.df['IsAfterHours'] = (
            ((self.df['Hour'] < 9) | (self.df['Hour'] > 16)) & (~self.df['IsWeekend'])
        )
        
        # Content complexity indicators
        self.df['Avg_Word_Length'] = self.df['Full_Text'].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
        )

        # Optional topic filtering
        if topic_filter:
            if isinstance(topic_filter, str):
                filter_expr = topic_filter
            else:
                filter_expr = '|'.join(topic_filter)
            mask = self.df['Full_Text'].str.contains(filter_expr, case=False, na=False)
            self.df = self.df[mask]
            print(f"  🔍 Topic filter applied ({filter_expr}) - {len(self.df)} articles remaining")

        # Filter and clean
        initial_count = len(self.df)
        self.df = self.df[self.df['Text_Length'] > 50]
        self.df = self.df.dropna(subset=['Full_Text'])
        filtered_count = len(self.df)

        print(f"  🧹 {initial_count - filtered_count} articles removed in cleaning")
        print(f"  ✅ {filtered_count} articles ready for analysis")
        
        return self.df
    
    def generate_multi_dimensional_embeddings(self, sample_size=None):
        """Generate multiple embedding representations for richer analysis"""
        print("\n🧠 Multi-Dimensional Embedding Generation...")
        
        if sample_size and len(self.df) > sample_size:
            df_sample = self.df.sample(n=sample_size, random_state=42)
            print(f"  🎯 Analyzing sample of {sample_size} articles")
        else:
            df_sample = self.df
            print(f"  🎯 Analyzing all {len(df_sample)} articles")
        
        self.analysis_df = df_sample.copy().reset_index(drop=True)
        texts = self.analysis_df['Full_Text'].tolist()
        headlines = self.analysis_df['Headline'].tolist()
        
        # Primary semantic embeddings
        print("  ⚙️ Computing primary semantic embeddings...")
        self.embeddings = self.sentence_model.encode(
            texts, show_progress_bar=True, batch_size=32
        )
        
        # Secondary domain-specific embeddings
        print("  🔬 Computing domain-specific embeddings...")
        self.domain_embeddings = self.domain_model.encode(
            texts, show_progress_bar=True, batch_size=32
        )
        
        # Headline-specific embeddings for narrative focus
        print("  📰 Computing headline embeddings...")
        self.headline_embeddings = self.sentence_model.encode(
            headlines, show_progress_bar=True, batch_size=32
        )
        
        # Enhanced TF-IDF with adaptive features
        print("  📝 Computing adaptive TF-IDF features...")
        tfidf_matrix = self.adaptive_tfidf.fit_transform(texts)
        self.tfidf_features = tfidf_matrix.toarray()
        self.feature_names = self.adaptive_tfidf.get_feature_names_out()
        
        # Hybrid embeddings (weighted combination)
        print("  🔄 Creating hybrid embedding representations...")
        self.hybrid_embeddings = np.concatenate([
            self.embeddings * 0.6,  # Primary weight
            self.domain_embeddings * 0.3,  # Domain weight
            self.headline_embeddings * 0.1  # Focus weight
        ], axis=1)
        
        print(f"  ✅ Multi-dimensional embeddings generated")
        print(f"      Primary: {self.embeddings.shape}")
        print(f"      Domain: {self.domain_embeddings.shape}")
        print(f"      Headlines: {self.headline_embeddings.shape}")
        print(f"      Hybrid: {self.hybrid_embeddings.shape}")
        
        return self.embeddings
    
    def dynamic_semantic_clustering(self, min_cluster_size=5, temporal_window_days=7):
        """Advanced clustering with temporal and semantic considerations"""
        print(f"\n🔍 Dynamic Semantic Clustering...")
        
        # Multi-level clustering approach
        clustering_results = {}
        
        # 1. Primary semantic clustering
        print("  🎯 Primary semantic clustering...")
        primary_clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='cosine',
            cluster_selection_epsilon=0.15,
            min_samples=3,
            cluster_selection_method='eom'
        )
        primary_clusters = primary_clusterer.fit_predict(self.embeddings)
        clustering_results['primary'] = primary_clusters
        
        # 2. Domain-specific clustering
        print("  🔬 Domain-specific clustering...")
        domain_clusterer = HDBSCAN(
            min_cluster_size=max(3, min_cluster_size//2),
            metric='cosine',
            cluster_selection_epsilon=0.2,
            min_samples=2
        )
        domain_clusters = domain_clusterer.fit_predict(self.domain_embeddings)
        clustering_results['domain'] = domain_clusters
        
        # 3. Headline-focused clustering
        print("  📰 Headline-focused clustering...")
        headline_clusterer = HDBSCAN(
            min_cluster_size=max(3, min_cluster_size//2),
            metric='cosine',
            cluster_selection_epsilon=0.1,
            min_samples=2
        )
        headline_clusters = headline_clusterer.fit_predict(self.headline_embeddings)
        clustering_results['headlines'] = headline_clusters
        
        # 4. Temporal clustering within semantic groups
        print("  ⏰ Temporal-semantic clustering...")
        self.temporal_clusters = self.compute_temporal_clusters(
            primary_clusters, temporal_window_days
        )
        
        # 5. Consensus clustering (ensemble approach)
        print("  🤝 Computing consensus clusters...")
        self.clusters = self.compute_consensus_clusters(clustering_results)
        
        # Store all clustering results
        self.all_clusters = clustering_results
        self.analysis_df['Cluster_Primary'] = primary_clusters
        self.analysis_df['Cluster_Domain'] = domain_clusters
        self.analysis_df['Cluster_Headlines'] = headline_clusters
        self.analysis_df['Cluster_Consensus'] = self.clusters
        
        # Analyze cluster quality
        unique_clusters = np.unique(self.clusters)
        n_clusters = len(unique_clusters[unique_clusters >= 0])
        n_noise = np.sum(self.clusters == -1)
        
        print(f"  📊 Clustering Results:")
        print(f"      Primary clusters: {len(np.unique(primary_clusters[primary_clusters >= 0]))}")
        print(f"      Domain clusters: {len(np.unique(domain_clusters[domain_clusters >= 0]))}")
        print(f"      Headline clusters: {len(np.unique(headline_clusters[headline_clusters >= 0]))}")
        print(f"      Consensus clusters: {n_clusters}")
        print(f"      Noise articles: {n_noise}")
        
        return self.clusters
    
    def compute_temporal_clusters(self, primary_clusters, window_days):
        """Compute temporal sub-clusters within semantic clusters"""
        temporal_clusters = {}
        
        for cluster_id in np.unique(primary_clusters[primary_clusters >= 0]):
            cluster_data = self.analysis_df[primary_clusters == cluster_id].copy()
            
            if len(cluster_data) < 3:
                continue
                
            # Sort by date
            cluster_data = cluster_data.sort_values('Date')
            
            # Create temporal windows
            temporal_groups = []
            current_group = []
            current_start = cluster_data.iloc[0]['Date']
            
            for idx, row in cluster_data.iterrows():
                if (row['Date'] - current_start).days <= window_days:
                    current_group.append(idx)
                else:
                    if len(current_group) >= 2:
                        temporal_groups.append(current_group)
                    current_group = [idx]
                    current_start = row['Date']
            
            if len(current_group) >= 2:
                temporal_groups.append(current_group)
            
            temporal_clusters[cluster_id] = temporal_groups
        
        return temporal_clusters
    
    def compute_consensus_clusters(self, clustering_results):
        """Compute consensus clusters using ensemble approach"""
        n_articles = len(self.analysis_df)
        
        # Create co-occurrence matrix
        co_occurrence = np.zeros((n_articles, n_articles))
        
        for cluster_type, clusters in clustering_results.items():
            for cluster_id in np.unique(clusters[clusters >= 0]):
                cluster_indices = np.where(clusters == cluster_id)[0]
                for i in cluster_indices:
                    for j in cluster_indices:
                        if i != j:
                            co_occurrence[i, j] += 1
        
        # Normalize by number of clustering methods
        co_occurrence = co_occurrence / len(clustering_results)
        
        # Use threshold-based consensus
        threshold = 0.5  # Articles must agree in at least 50% of clusterings
        
        # Convert to distance matrix and cluster
        distance_matrix = 1 - co_occurrence
        
        # Use DBSCAN on the consensus matrix
        from sklearn.cluster import DBSCAN
        consensus_clusterer = DBSCAN(
            metric='precomputed',
            eps=0.6,
            min_samples=3
        )
        
        consensus_clusters = consensus_clusterer.fit_predict(distance_matrix)
        
        return consensus_clusters
    
    def extract_dynamic_keywords(self):
        """Extract dynamic, context-aware keywords for each cluster"""
        print("\n🔑 Dynamic Keyword Extraction...")
        
        self.dynamic_keywords = {}
        
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            cluster_data = self.analysis_df[self.clusters == cluster_id]
            cluster_texts = cluster_data['Full_Text'].tolist()
            
            if len(cluster_texts) == 0:
                continue
            
            # Combine all texts in cluster
            combined_text = ' '.join(cluster_texts)
            
            # 1. KeyBERT extraction (semantic similarity)
            if KEYBERT_AVAILABLE and self.keybert_model:
                try:
                    keybert_keywords = self.keybert_model.extract_keywords(
                        combined_text,
                        keyphrase_ngram_range=(1, 3),
                        stop_words='english',
                        use_maxsum=True,
                        nr_candidates=50,
                        top_k=15
                    )
                    keybert_words = [kw[0] for kw in keybert_keywords]
                except:
                    keybert_words = []
            else:
                keybert_words = []
            
            # 2. YAKE extraction (unsupervised)
            if YAKE_AVAILABLE and self.yake_extractor:
                try:
                    yake_keywords = self.yake_extractor.extract_keywords(combined_text)
                    yake_words = [kw[0] for kw in yake_keywords[:15]]
                except:
                    yake_words = []
            else:
                yake_words = []
            
            # 3. TF-IDF based (cluster-specific)
            cluster_indices = np.where(self.clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_tfidf = self.tfidf_features[cluster_indices]
                avg_tfidf = np.mean(cluster_tfidf, axis=0)
                top_indices = avg_tfidf.argsort()[-15:][::-1]
                tfidf_words = [self.feature_names[i] for i in top_indices]
            else:
                tfidf_words = []
            
            # 4. Named entities (high frequency)
            entity_words = []
            if self.nlp:
                for text in cluster_texts[:5]:  # Sample for efficiency
                    doc = self.nlp(text[:1000])
                    entities = [ent.text.lower() for ent in doc.ents 
                              if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
                    entity_words.extend(entities)
                
                entity_counter = Counter(entity_words)
                top_entities = [entity for entity, count in entity_counter.most_common(10)]
            else:
                top_entities = []
            
            # 5. Temporal keywords (trending terms)
            temporal_keywords = self.extract_temporal_keywords(cluster_data)
            
            # Combine and rank all keywords
            all_keywords = {
                'semantic': keybert_words,
                'statistical': yake_words, 
                'tfidf': tfidf_words,
                'entities': top_entities,
                'temporal': temporal_keywords
            }
            
            # Create unified ranking
            keyword_scores = defaultdict(float)
            
            # Weight different extraction methods
            weights = {
                'semantic': 0.3,
                'statistical': 0.25,
                'tfidf': 0.2,
                'entities': 0.15,
                'temporal': 0.1
            }
            
            for method, keywords in all_keywords.items():
                # Ensure we only process valid, non-empty keywords
                cleaned_keywords = []
                for kw in keywords:
                    # Skip missing values
                    if isinstance(kw, (float, np.floating)) and np.isnan(kw):
                        continue
                    # Convert non-string keywords to strings
                    cleaned_keywords.append(str(kw))

                for i, keyword in enumerate(cleaned_keywords):
                    if not keyword:
                        continue
                    # Higher rank = higher score
                    score = weights[method] * (len(cleaned_keywords) - i) / len(cleaned_keywords)
                    keyword_scores[keyword.lower()] += score
            
            # Get top dynamic keywords
            top_dynamic_keywords = sorted(
                keyword_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:12]
            
            self.dynamic_keywords[cluster_id] = {
                'unified_ranking': [kw[0] for kw in top_dynamic_keywords],
                'by_method': all_keywords,
                'scores': dict(top_dynamic_keywords)
            }
        
        print(f"  ✅ Dynamic keywords extracted for {len(self.dynamic_keywords)} clusters")
        
        return self.dynamic_keywords
    
    def extract_temporal_keywords(self, cluster_data):
        """Extract keywords that are trending within the cluster's timeframe"""
        if len(cluster_data) < 3:
            return []
        
        # Sort by date
        sorted_data = cluster_data.sort_values('Date')
        
        # Split into early/late periods
        mid_point = len(sorted_data) // 2
        early_texts = ' '.join(sorted_data.iloc[:mid_point]['Full_Text'].tolist())
        late_texts = ' '.join(sorted_data.iloc[mid_point:]['Full_Text'].tolist())
        
        # Extract keywords from each period
        def extract_keywords_from_text(text):
            if not text.strip():
                return []
            if YAKE_AVAILABLE and self.yake_extractor:
                try:
                    keywords = self.yake_extractor.extract_keywords(text)
                    return [kw[0] for kw in keywords[:20]]
                except:
                    return []
            else:
                # Fallback: simple word frequency
                words = re.findall(r'\b\w+\b', text.lower())
                word_freq = Counter(words)
                return [word for word, count in word_freq.most_common(20) if len(word) > 3]
        
        early_keywords = set(extract_keywords_from_text(early_texts))
        late_keywords = set(extract_keywords_from_text(late_texts))
        
        # Find trending keywords (more prominent in later period)
        trending = late_keywords - early_keywords
        
        return list(trending)[:8]
    
    def create_simplified_dashboard(self):
        """Create a simplified but still enhanced dashboard"""
        print("\n📊 Creating Enhanced Dashboard...")
        
        # Create 2x3 dashboard (simplified from 3x4)
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Narrative Clusters Overview', 'Sentiment Distribution', 'Temporal Patterns',
                'Actor Network', 'Keyword Analysis', 'Cluster Insights'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Narrative Clusters Overview
        if UMAP_AVAILABLE:
            try:
                reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1)
                embeddings_2d = reducer.fit_transform(self.embeddings)
            except:
                pca = PCA(n_components=2, random_state=42)
                embeddings_2d = pca.fit_transform(self.embeddings)
        else:
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(self.embeddings)
        
        colors = px.colors.qualitative.Set3
        cluster_colors = {-1: 'lightgray'}
        
        valid_clusters = np.unique(self.clusters[self.clusters >= 0])
        for i, cluster_id in enumerate(valid_clusters):
            cluster_colors[cluster_id] = colors[i % len(colors)]
        
        scatter_colors = [cluster_colors[cluster] for cluster in self.clusters]
        
        fig.add_trace(
            go.Scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                mode='markers',
                marker=dict(
                    color=scatter_colors,
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[f"Cluster {c}<br>{self.analysis_df.iloc[i]['Headline'][:50]}..." 
                      for i, c in enumerate(self.clusters)],
                hovertemplate="<b>%{text}</b><extra></extra>",
                name='Articles',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Sentiment Distribution
        if 'Sentiment_Label' in self.analysis_df.columns:
            sentiment_counts = self.analysis_df['Sentiment_Label'].value_counts()
            
            fig.add_trace(
                go.Bar(
                    x=list(sentiment_counts.index),
                    y=list(sentiment_counts.values),
                    marker_color=['green' if s=='positive' else 'red' if s=='negative' else 'gray' 
                                 for s in sentiment_counts.index],
                    text=list(sentiment_counts.values),
                    textposition='auto',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Temporal Patterns
        daily_counts = self.analysis_df.groupby(self.analysis_df['Date'].dt.date).size()
        
        fig.add_trace(
            go.Scatter(
                x=daily_counts.index,
                y=daily_counts.values,
                mode='lines+markers',
                name='Daily Articles',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                showlegend=False
            ),
            row=1, col=3
        )
        
        # 4. Simple Actor Network (if available)
        # Simplified version - just show top actors
        if hasattr(self, 'dynamic_keywords') and self.dynamic_keywords:
            # Show keyword cloud as bars instead of network
            all_keywords = []
            for cluster_keywords in self.dynamic_keywords.values():
                if 'unified_ranking' in cluster_keywords:
                    all_keywords.extend(cluster_keywords['unified_ranking'][:5])
            
            keyword_freq = Counter(all_keywords)
            top_keywords = keyword_freq.most_common(10)
            
            if top_keywords:
                fig.add_trace(
                    go.Bar(
                        x=[kw[1] for kw in top_keywords],
                        y=[kw[0] for kw in top_keywords],
                        orientation='h',
                        marker_color='lightblue',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 5. Keyword Analysis by Cluster
        if hasattr(self, 'dynamic_keywords'):
            cluster_sizes = []
            cluster_labels = []
            
            for cluster_id in valid_clusters[:10]:  # Top 10 clusters
                cluster_size = len(self.analysis_df[self.clusters == cluster_id])
                cluster_sizes.append(cluster_size)
                cluster_labels.append(f'C{cluster_id}')
            
            fig.add_trace(
                go.Bar(
                    x=cluster_labels,
                    y=cluster_sizes,
                    marker_color='lightcoral',
                    text=cluster_sizes,
                    textposition='auto',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 6. Cluster Quality Metrics
        cluster_quality_data = []
        for cluster_id in valid_clusters[:8]:
            cluster_data = self.analysis_df[self.clusters == cluster_id]
            avg_confidence = cluster_data.get('Sentiment_Confidence', pd.Series([0.5] * len(cluster_data))).mean()
            cluster_quality_data.append(avg_confidence)
        
        if cluster_quality_data:
            fig.add_trace(
                go.Bar(
                    x=[f'C{cid}' for cid in valid_clusters[:8]],
                    y=cluster_quality_data,
                    marker_color='lightgreen',
                    text=[f'{q:.2f}' for q in cluster_quality_data],
                    textposition='auto',
                    showlegend=False
                ),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1400,
            title_text="🧠 Enhanced Narrative Intelligence Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Dimension 1", row=1, col=1)
        fig.update_yaxes(title_text="Dimension 2", row=1, col=1)
        
        fig.update_xaxes(title_text="Sentiment", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_xaxes(title_text="Date", row=1, col=3)
        fig.update_yaxes(title_text="Articles", row=1, col=3)
        
        fig.update_xaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Keywords", row=2, col=1)
        
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Size", row=2, col=2)
        
        fig.update_xaxes(title_text="Cluster", row=2, col=3)
        fig.update_yaxes(title_text="Quality", row=2, col=3)
        
        fig.show()
        fig.write_html("enhanced_narrative_dashboard.html")
        
        print("  ✅ Enhanced dashboard created and saved as 'enhanced_narrative_dashboard.html'")
        
        return fig
    
    def advanced_sentiment_analysis(self):
        """Multi-dimensional sentiment and emotion analysis"""
        print("\n🎭 Advanced Sentiment & Emotion Analysis...")
        
        # Pre-truncate texts to model input limit for efficient batch processing
        texts = [t[:512] for t in self.analysis_df['Full_Text']]

        # Run emotion and financial sentiment pipelines in batches
        try:
            emotion_results = self.emotion_pipeline(
                texts, batch_size=32, truncation=True
            )
        except Exception:
            emotion_results = [{'label': 'neutral', 'score': 0.5} for _ in texts]

        try:
            financial_results = self.financial_sentiment(
                texts, batch_size=32, truncation=True
            )
        except Exception:
            financial_results = [{'label': 'neutral', 'score': 0.5} for _ in texts]

        sentiments = []
        emotions = []
        financial_sentiments = []

        for emo, fin in zip(emotion_results, financial_results):
            # Map emotions to a basic positive/negative/neutral sentiment
            emotion_label = emo.get('label', 'neutral').lower()
            if emotion_label in ['joy', 'love', 'optimism']:
                sentiment_label = 'positive'
            elif emotion_label in ['anger', 'fear', 'sadness', 'pessimism']:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'

            sentiments.append({
                'label': sentiment_label,
                'confidence': emo.get('score', 0.5)
            })
            emotions.append({
                'emotion': emo.get('label', 'neutral'),
                'confidence': emo.get('score', 0.5)
            })
            financial_sentiments.append({
                'label': fin.get('label', 'neutral').lower(),
                'confidence': fin.get('score', 0.5)
            })
        
        # Add to dataframe
        self.analysis_df['Sentiment_Label'] = [s['label'] for s in sentiments]
        self.analysis_df['Sentiment_Confidence'] = [s['confidence'] for s in sentiments]
        self.analysis_df['Emotion'] = [e['emotion'] for e in emotions]
        self.analysis_df['Emotion_Confidence'] = [e['confidence'] for e in emotions]
        self.analysis_df['Financial_Sentiment'] = [f['label'] for f in financial_sentiments]
        self.analysis_df['Financial_Confidence'] = [f['confidence'] for f in financial_sentiments]
        
        print(f"  ✅ Multi-dimensional sentiment analysis complete")
    
    def generate_enhanced_narrative_summaries(self):
        """Generate comprehensive narrative summaries with all insights"""
        print("\n📝 Generating Enhanced Narrative Summaries...")
        
        self.narrative_summaries = {}
        
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            cluster_data = self.analysis_df[self.clusters == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Basic metrics
            cluster_size = len(cluster_data)
            date_range = f"{cluster_data['Date'].min().strftime('%Y-%m-%d')} to {cluster_data['Date'].max().strftime('%Y-%m-%d')}"
            duration_days = (cluster_data['Date'].max() - cluster_data['Date'].min()).days + 1
            
            # Dynamic keywords
            keywords_info = self.dynamic_keywords.get(cluster_id, {})
            
            # Actor analysis
            cluster_actors = set()
            for authors_list in cluster_data['Journalists_List']:
                cluster_actors.update([a for a in authors_list if a and str(a) != 'nan'])
            
            # Multi-dimensional sentiment
            sentiment_analysis = {}
            if 'Sentiment_Label' in cluster_data.columns:
                sentiment_analysis = {
                    'basic_sentiment': cluster_data['Sentiment_Label'].value_counts().to_dict(),
                    'emotions': cluster_data['Emotion'].value_counts().to_dict(),
                    'financial_sentiment': cluster_data['Financial_Sentiment'].value_counts().to_dict(),
                    'avg_confidence': {
                        'sentiment': cluster_data['Sentiment_Confidence'].mean(),
                        'emotion': cluster_data['Emotion_Confidence'].mean(),
                        'financial': cluster_data['Financial_Confidence'].mean()
                    }
                }
            
            # Content quality metrics
            content_metrics = {
                'avg_readability': cluster_data['Readability_Flesch'].mean(),
                'avg_word_count': cluster_data['Word_Count'].mean(),
                'market_timing_ratio': cluster_data['IsMarketHours'].mean() if 'IsMarketHours' in cluster_data.columns else 0
            }
            
            # Top headlines by different criteria
            top_headlines = {
                'most_recent': cluster_data.nlargest(3, 'Date')['Headline'].tolist(),
                'highest_confidence': cluster_data.nlargest(3, 'Sentiment_Confidence')['Headline'].tolist() if 'Sentiment_Confidence' in cluster_data.columns else [],
                'longest': cluster_data.nlargest(3, 'Word_Count')['Headline'].tolist()
            }
            
            # Narrative classification
            narrative_type = self.classify_advanced_narrative_type(cluster_data, keywords_info)
            
            # Generate narrative description
            narrative_description = self.generate_advanced_narrative_description(
                cluster_data, keywords_info, sentiment_analysis
            )
            
            summary = {
                'cluster_id': cluster_id,
                'basic_info': {
                    'size': cluster_size,
                    'date_range': date_range,
                    'duration_days': duration_days,
                    'narrative_type': narrative_type
                },
                'keywords_analysis': keywords_info,
                'actor_analysis': {
                    'total_actors': len(cluster_actors),
                    'all_actors': list(cluster_actors)
                },
                'sentiment_analysis': sentiment_analysis,
                'content_metrics': content_metrics,
                'top_headlines': top_headlines,
                'narrative_description': narrative_description,
                'insights': self.generate_narrative_insights(cluster_data, keywords_info)
            }
            
            self.narrative_summaries[cluster_id] = summary
        
        print(f"  ✅ Enhanced summaries generated for {len(self.narrative_summaries)} narratives")
    
    def classify_advanced_narrative_type(self, cluster_data, keywords_info):
        """Advanced narrative type classification"""
        
        # Combine headline and keyword analysis
        all_text = ' '.join(cluster_data['Headline'].tolist()).lower()
        
        if keywords_info and 'unified_ranking' in keywords_info:
            all_text += ' ' + ' '.join(keywords_info['unified_ranking'])
        
        # Enhanced classification rules
        classification_rules = [
            (['crisis', 'crash', 'fall', 'decline', 'drop', 'plunge', 'collapse'], 'Market Crisis'),
            (['growth', 'rise', 'surge', 'boom', 'rally', 'gains', 'jump'], 'Market Growth'),
            (['merger', 'acquisition', 'deal', 'takeover', 'buyout'], 'M&A Activity'),
            (['earnings', 'profit', 'revenue', 'quarterly', 'results'], 'Earnings/Results'),
            (['regulation', 'policy', 'government', 'fed', 'central bank', 'law'], 'Regulatory/Policy'),
            (['technology', 'ai', 'digital', 'innovation', 'tech', 'cyber'], 'Technology/Innovation'),
            (['ipo', 'public', 'listing', 'debut', 'offering'], 'IPO/Public Markets'),
            (['leadership', 'ceo', 'executive', 'management', 'resignation'], 'Leadership Changes'),
            (['investigation', 'lawsuit', 'legal', 'court', 'settlement'], 'Legal/Compliance'),
            (['environment', 'climate', 'sustainability', 'green', 'carbon'], 'ESG/Sustainability')
        ]
        
        for keywords, category in classification_rules:
            if any(keyword in all_text for keyword in keywords):
                return category
        
        return 'General Business'
    
    def generate_advanced_narrative_description(self, cluster_data, keywords_info, sentiment_analysis):
        """Generate comprehensive narrative description"""
        
        description_parts = []
        
        # Basic narrative intro
        size = len(cluster_data)
        duration = (cluster_data['Date'].max() - cluster_data['Date'].min()).days + 1
        
        if keywords_info and 'unified_ranking' in keywords_info:
            main_themes = ', '.join(keywords_info['unified_ranking'][:3])
            description_parts.append(f"This narrative centers around {main_themes}")
        
        # Temporal context
        if duration == 1:
            description_parts.append(f"with {size} articles published on a single day")
        else:
            description_parts.append(f"spanning {duration} days with {size} articles")
        
        # Sentiment context
        if sentiment_analysis and 'basic_sentiment' in sentiment_analysis:
            dominant_sentiment = max(sentiment_analysis['basic_sentiment'].items(), key=lambda x: x[1])[0]
            sentiment_intensity = sentiment_analysis.get('avg_confidence', {}).get('sentiment', 0.5)
            
            sentiment_desc = {
                'positive': f"positive sentiment (confidence: {sentiment_intensity:.2f})",
                'negative': f"negative sentiment (confidence: {sentiment_intensity:.2f})",
                'neutral': f"neutral tone (confidence: {sentiment_intensity:.2f})"
            }.get(dominant_sentiment, "mixed sentiment")
            
            description_parts.append(f"The coverage shows {sentiment_desc}")
        
        # Financial sentiment if different from basic
        if sentiment_analysis and 'financial_sentiment' in sentiment_analysis:
            fin_sentiment = max(sentiment_analysis['financial_sentiment'].items(), key=lambda x: x[1])[0]
            if 'basic_sentiment' in sentiment_analysis:
                basic_sentiment = max(sentiment_analysis['basic_sentiment'].items(), key=lambda x: x[1])[0]
                if fin_sentiment != basic_sentiment:
                    description_parts.append(f"with specifically {fin_sentiment} financial implications")
        
        return '. '.join(description_parts) + '.'
    
    def generate_narrative_insights(self, cluster_data, keywords_info):
        """Generate actionable insights for each narrative"""
        
        insights = []
        
        # Temporal insights
        duration = (cluster_data['Date'].max() - cluster_data['Date'].min()).days + 1
        article_density = len(cluster_data) / duration
        
        if article_density > 2:
            insights.append("High coverage intensity suggests significant market impact")
        
        # Actor insights
        unique_actors = set()
        for authors_list in cluster_data['Journalists_List']:
            unique_actors.update([a for a in authors_list if a and str(a) != 'nan'])
        
        if len(unique_actors) > len(cluster_data) * 0.8:  # Many different authors
            insights.append("Broad journalistic coverage indicates widespread industry attention")
        elif len(unique_actors) < len(cluster_data) * 0.3:  # Few authors, many articles
            insights.append("Concentrated reporting suggests specialized coverage or breaking news")
        
        # Market timing insights
        if 'IsMarketHours' in cluster_data.columns:
            market_ratio = cluster_data['IsMarketHours'].mean()
            if market_ratio > 0.7:
                insights.append("High proportion of market-hours coverage suggests trading relevance")
            elif market_ratio < 0.3:
                insights.append("Off-hours coverage pattern indicates planned announcements or global events")
        
        # Keyword insights
        if keywords_info and 'by_method' in keywords_info:
            entity_keywords = keywords_info['by_method'].get('entities', [])
            if len(entity_keywords) > 5:
                insights.append("Rich entity content suggests complex stakeholder dynamics")
        
        return insights
    
    def create_comprehensive_report(self):
        """Create comprehensive narrative intelligence report"""
        print("\n📋 Creating Comprehensive Narrative Intelligence Report...")
        
        print("\n" + "="*100)
        print("🧠 ENHANCED NARRATIVE INTELLIGENCE PLATFORM - COMPREHENSIVE REPORT")
        print("="*100)
        
        # Executive Summary
        print(f"\n📊 EXECUTIVE SUMMARY")
        print(f"   • Total Articles Analyzed: {len(self.analysis_df):,}")
        print(f"   • Narrative Clusters Identified: {len(self.narrative_summaries)}")
        print(f"   • Analysis Period: {self.analysis_df['Date'].min().strftime('%Y-%m-%d')} to {self.analysis_df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"   • Average Daily Coverage: {len(self.analysis_df) / (self.analysis_df['Date'].max() - self.analysis_df['Date'].min()).days:.1f} articles")
        
        # Top Narratives Analysis
        print(f"\n📚 TOP NARRATIVE CLUSTERS (Detailed Analysis)")
        
        sorted_narratives = sorted(
            self.narrative_summaries.items(),
            key=lambda x: x[1]['basic_info']['size'],
            reverse=True
        )
        
        for i, (cluster_id, summary) in enumerate(sorted_narratives[:8], 1):
            basic_info = summary['basic_info']
            keywords_analysis = summary.get('keywords_analysis', {})
            actor_analysis = summary['actor_analysis']
            sentiment_analysis = summary['sentiment_analysis']
            
            print(f"\n   {i}. NARRATIVE CLUSTER {cluster_id}")
            print(f"      📊 Size: {basic_info['size']} articles")
            print(f"      📅 Duration: {basic_info['duration_days']} days ({basic_info['date_range']})")
            print(f"      🏷️ Type: {basic_info['narrative_type']}")
            print(f"      📝 Description: {summary['narrative_description']}")
            
            # Dynamic Keywords
            if 'unified_ranking' in keywords_analysis:
                print(f"      🔑 Dynamic Keywords: {', '.join(keywords_analysis['unified_ranking'][:6])}")
            
            # Sentiment Distribution
            basic_sentiment = sentiment_analysis.get('basic_sentiment', {})
            if basic_sentiment:
                sentiment_str = ', '.join([f"{k}: {v}" for k, v in basic_sentiment.items()])
                print(f"      🎭 Sentiment: {sentiment_str}")
            
            # Financial Sentiment
            financial_sentiment = sentiment_analysis.get('financial_sentiment', {})
            if financial_sentiment:
                fin_sentiment_str = ', '.join([f"{k}: {v}" for k, v in financial_sentiment.items()])
                print(f"      💰 Financial Sentiment: {fin_sentiment_str}")
            
            # Key Insights
            insights = summary.get('insights', [])
            if insights:
                print(f"      💡 Key Insights:")
                for insight in insights[:2]:
                    print(f"          • {insight}")
        
        # Temporal Analysis
        print(f"\n⏰ TEMPORAL INTELLIGENCE")
        
        # Peak activity analysis
        daily_counts = self.analysis_df.groupby(self.analysis_df['Date'].dt.date).size()
        peak_day = daily_counts.idxmax()
        peak_count = daily_counts.max()
        
        print(f"   • Peak Activity: {peak_count} articles on {peak_day}")
        print(f"   • Average Daily Volume: {daily_counts.mean():.1f} articles")
        print(f"   • Coverage Consistency: {daily_counts.std():.1f} standard deviation")
        
        # Market timing analysis
        if 'IsMarketHours' in self.analysis_df.columns:
            market_ratio = self.analysis_df['IsMarketHours'].mean()
            print(f"   • Market Hours Coverage: {market_ratio:.1%}")
        
        # Sentiment Intelligence
        print(f"\n🎭 SENTIMENT & EMOTION INTELLIGENCE")
        
        if 'Sentiment_Label' in self.analysis_df.columns:
            overall_sentiment = self.analysis_df['Sentiment_Label'].value_counts(normalize=True)
            print(f"   • Overall Sentiment Distribution:")
            for sentiment, ratio in overall_sentiment.items():
                emoji = "😊" if sentiment == "positive" else "😟" if sentiment == "negative" else "😐"
                print(f"     {emoji} {sentiment.title()}: {ratio:.1%}")
        
        if 'Financial_Sentiment' in self.analysis_df.columns:
            financial_sentiment = self.analysis_df['Financial_Sentiment'].value_counts(normalize=True)
            print(f"   • Financial Sentiment Distribution:")
            for sentiment, ratio in financial_sentiment.items():
                print(f"     💰 {sentiment.title()}: {ratio:.1%}")
        
        # Content Quality Analysis
        avg_readability = self.analysis_df['Readability_Flesch'].mean()
        print(f"   • Average Readability (Flesch): {avg_readability:.1f}")
        
        # Strategic Insights
        print(f"\n💡 STRATEGIC INSIGHTS & RECOMMENDATIONS")
        
        # Generate strategic insights
        strategic_insights = self.generate_strategic_insights()
        for i, insight in enumerate(strategic_insights, 1):
            print(f"   {i}. {insight}")
        
        print("\n" + "="*100)
        print("📄 Report generated by Enhanced Narrative Intelligence Platform")
        print("🔗 Interactive dashboard: enhanced_narrative_dashboard.html")
        print("="*100)
    
    def generate_strategic_insights(self):
        """Generate strategic insights from the analysis"""
        insights = []
        
        # Narrative diversity insight
        narrative_types = [summary['basic_info']['narrative_type'] for summary in self.narrative_summaries.values()]
        type_diversity = len(set(narrative_types))
        insights.append(f"Narrative diversity spans {type_diversity} distinct categories, indicating {'broad' if type_diversity > 5 else 'focused'} coverage scope")
        
        # Temporal concentration insight
        daily_counts = self.analysis_df.groupby(self.analysis_df['Date'].dt.date).size()
        temporal_concentration = daily_counts.std() / daily_counts.mean()
        if temporal_concentration > 1.0:
            insights.append("High temporal concentration suggests event-driven coverage patterns - monitor for breaking news cycles")
        else:
            insights.append("Steady coverage flow indicates ongoing systematic reporting - suitable for trend analysis")
        
        # Sentiment volatility insight
        if 'Sentiment_Label' in self.analysis_df.columns:
            sentiment_by_day = self.analysis_df.groupby(self.analysis_df['Date'].dt.date)['Sentiment_Label'].apply(
                lambda x: (x == 'positive').mean() - (x == 'negative').mean()
            )
            sentiment_volatility = sentiment_by_day.std()
            if sentiment_volatility > 0.3:
                insights.append("High sentiment volatility detected - market sentiment shifts rapidly, requires real-time monitoring")
        
        return insights
    
    def export_enhanced_results(self, output_prefix="enhanced_narrative_intelligence"):
        """Export all enhanced analysis results"""
        print(f"\n💾 Exporting Enhanced Results...")
        
        # 1. Complete analysis dataset
        self.analysis_df.to_csv(f"{output_prefix}_complete_analysis.csv", index=False)
        
        # 2. Enhanced narrative summaries
        summaries_data = []
        for cluster_id, summary in self.narrative_summaries.items():
            basic_info = summary['basic_info']
            keywords_analysis = summary.get('keywords_analysis', {})
            actor_analysis = summary['actor_analysis']
            sentiment_analysis = summary['sentiment_analysis']
            
            row = {
                'cluster_id': cluster_id,
                'size': basic_info['size'],
                'duration_days': basic_info['duration_days'],
                'narrative_type': basic_info['narrative_type'],
                'narrative_description': summary['narrative_description'],
                'dynamic_keywords': ', '.join(keywords_analysis.get('unified_ranking', [])[:10]),
                'total_actors': actor_analysis['total_actors'],
                'key_insights': '; '.join(summary.get('insights', []))
            }
            
            # Add sentiment data if available
            if sentiment_analysis:
                basic_sentiment = sentiment_analysis.get('basic_sentiment', {})
                row.update({
                    'sentiment_positive': basic_sentiment.get('positive', 0),
                    'sentiment_negative': basic_sentiment.get('negative', 0),
                    'sentiment_neutral': basic_sentiment.get('neutral', 0)
                })
                
                financial_sentiment = sentiment_analysis.get('financial_sentiment', {})
                row.update({
                    'financial_positive': financial_sentiment.get('positive', 0),
                    'financial_negative': financial_sentiment.get('negative', 0),
                    'financial_neutral': financial_sentiment.get('neutral', 0)
                })
            
            summaries_data.append(row)
        
        summaries_df = pd.DataFrame(summaries_data)
        summaries_df.to_csv(f"{output_prefix}_narrative_summaries.csv", index=False)
        
        # 3. Dynamic keywords analysis
        if hasattr(self, 'dynamic_keywords'):
            keywords_data = []
            for cluster_id, keyword_info in self.dynamic_keywords.items():
                if 'by_method' in keyword_info:
                    for method, keywords in keyword_info['by_method'].items():
                        for keyword in keywords[:10]:  # Top 10 per method
                            keywords_data.append({
                                'cluster_id': cluster_id,
                                'extraction_method': method,
                                'keyword': keyword,
                                'score': keyword_info.get('scores', {}).get(keyword, 0)
                            })
            
            keywords_df = pd.DataFrame(keywords_data)
            keywords_df.to_csv(f"{output_prefix}_dynamic_keywords.csv", index=False)
        
        # 4. Multi-dimensional embeddings
        np.save(f"{output_prefix}_primary_embeddings.npy", self.embeddings)
        np.save(f"{output_prefix}_domain_embeddings.npy", self.domain_embeddings)
        np.save(f"{output_prefix}_headline_embeddings.npy", self.headline_embeddings)
        np.save(f"{output_prefix}_hybrid_embeddings.npy", self.hybrid_embeddings)
        
        print(f"  ✅ Enhanced export completed:")
        print(f"     • {output_prefix}_complete_analysis.csv")
        print(f"     • {output_prefix}_narrative_summaries.csv")
        print(f"     • {output_prefix}_dynamic_keywords.csv")
        print(f"     • Multiple embedding files (.npy)")
        print(f"     • enhanced_narrative_dashboard.html")

    def situation_awareness_overview(self, top_n=5):
        """Provide a concise overview of top narratives for quick situation awareness"""
        if not getattr(self, 'narrative_summaries', None):
            print("No narrative summaries available. Run analysis first.")
            return

        print("\n🔎 SITUATION AWARENESS OVERVIEW")
        sorted_narratives = sorted(
            self.narrative_summaries.items(),
            key=lambda x: x[1]['basic_info']['size'],
            reverse=True
        )

        for cluster_id, summary in sorted_narratives[:top_n]:
            basic = summary['basic_info']
            sentiments = summary['sentiment_analysis'].get('basic_sentiment', {})
            dominant_sentiment = max(sentiments, key=sentiments.get) if sentiments else 'unknown'
            actors = summary['actor_analysis'].get('all_actors', [])[:5]
            keywords = summary['keywords_analysis'].get('unified_ranking', [])[:5]
            headline = summary['top_headlines'].get('most_recent', [''])
            example_headline = headline[0] if headline else ''

            print(f"\n📌 Cluster {cluster_id} – {basic['narrative_type']}")
            print(f"   Articles: {basic['size']} | Sentiment: {dominant_sentiment}")
            if actors:
                print(f"   Actors: {', '.join(actors)}")
            if keywords:
                print(f"   Keywords: {', '.join(keywords)}")
            if example_headline:
                print(f"   Example: {example_headline}")
    
    def run_enhanced_analysis(self, sample_size=5000, min_cluster_size=5, temporal_window_days=7, topic_filter=None):
        """Main method for enhanced narrative intelligence analysis"""
        print("🧠 ENHANCED NARRATIVE INTELLIGENCE PLATFORM")
        print("=" * 80)
        print("Enhanced analysis with dynamic clustering, multi-dimensional actor networks,")
        print("narrative evolution tracking, and comprehensive semantic context extraction")
        print("=" * 80)
        
        try:
            # Phase 1: Data Processing
            print("\n🔥 PHASE 1: ENHANCED DATA PROCESSING")
            self.load_and_preprocess_data(topic_filter=topic_filter)
            
            # Phase 2: Multi-Dimensional Embedding
            print("\n🔥 PHASE 2: MULTI-DIMENSIONAL EMBEDDING GENERATION")
            self.generate_multi_dimensional_embeddings(sample_size=sample_size)
            
            # Phase 3: Dynamic Semantic Clustering
            print("\n🔥 PHASE 3: DYNAMIC SEMANTIC CLUSTERING")
            self.dynamic_semantic_clustering(min_cluster_size=min_cluster_size, temporal_window_days=temporal_window_days)
            
            # Phase 4: Dynamic Keyword Extraction
            print("\n🔥 PHASE 4: DYNAMIC KEYWORD EXTRACTION")
            self.extract_dynamic_keywords()
            
            # Phase 5: Advanced Sentiment Analysis
            print("\n🔥 PHASE 5: MULTI-DIMENSIONAL SENTIMENT ANALYSIS")
            self.advanced_sentiment_analysis()
            
            # Phase 6: Enhanced Narrative Summaries
            print("\n🔥 PHASE 6: ENHANCED NARRATIVE SUMMARIES")
            self.generate_enhanced_narrative_summaries()
            
            # Phase 7: Dashboard Creation
            print("\n🔥 PHASE 7: ENHANCED DASHBOARD CREATION")
            self.create_simplified_dashboard()
            
            # Phase 8: Comprehensive Reporting
            print("\n🔥 PHASE 8: COMPREHENSIVE REPORTING")
            self.create_comprehensive_report()
            
            # Phase 9: Enhanced Export
            print("\n🔥 PHASE 9: ENHANCED EXPORT")
            self.export_enhanced_results()

            # Final situation awareness summary
            self.situation_awareness_overview()
            
            print("\n🎉 ENHANCED NARRATIVE INTELLIGENCE ANALYSIS COMPLETED!")
            print("\n📊 DELIVERABLES:")
            print("   📈 Interactive Dashboard: enhanced_narrative_dashboard.html")
            print("   📋 Comprehensive Analysis Report: Printed above")
            print("   💾 Enhanced Data Exports: Multiple CSV files with detailed metrics")
            print("   🧠 Multi-Dimensional Embeddings: Primary, domain, headline, and hybrid representations")
            print("   🔑 Dynamic Keywords: Context-aware, temporal, and semantic keyword extraction")
            print("   🎭 Multi-Dimensional Sentiment: Basic, emotional, and financial sentiment analysis")
            
            print("\n🚀 NEXT STEPS:")
            print("   1. Open the interactive dashboard for visual exploration")
            print("   2. Review the comprehensive report for strategic insights")
            print("   3. Analyze exported data for deeper investigation")
            print("   4. Use dynamic keywords for content strategy optimization")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during enhanced analysis: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Command line entry point for the narrative intelligence platform"""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced narrative intelligence analysis")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--sample-size", type=int, default=5000, help="Maximum articles to analyze")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum cluster size")
    parser.add_argument("--temporal-window-days", type=int, default=7, help="Temporal clustering window")
    parser.add_argument("--topic", nargs="*", help="Optional keyword(s) to filter articles")
    args = parser.parse_args()

    print("🧠 ENHANCED NARRATIVE INTELLIGENCE PLATFORM")
    print("=" * 70)
    print("Next-Generation Narrative Analysis with:")
    print("• Dynamic Semantic Clustering")
    print("• Multi-Dimensional Embeddings")
    print("• Advanced Sentiment & Emotion Analysis")
    print("• Context-Aware Keyword Extraction")
    print("• Comprehensive Intelligence Reports")
    print("=" * 70)
    print(f"Dataset: {args.csv}")
    if args.topic:
        print(f"Topic filter: {', '.join(args.topic)}")
    print()

    try:
        analyzer = AdvancedNarrativeIntelligence(args.csv)
        analyzer.run_enhanced_analysis(
            sample_size=args.sample_size,
            min_cluster_size=args.min_cluster_size,
            temporal_window_days=args.temporal_window_days,
            topic_filter=args.topic
        )
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")
        raise

if __name__ == "__main__":
    main()
