#!/usr/bin/env python3
"""
ENHANCED THEME CLUSTERING SCRIPT WITH AI NARRATIVES - IMPROVED VERSION
Addresses performance and quality issues with better embeddings, clustering, and narrative generation
"""
from transformers import pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
import torch
import random

# Fix HDBSCAN import
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional advanced dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available - will use PCA instead")

import ast
import re
from collections import Counter

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class EnhancedThemeClustering:
    """
    Enhanced theme clustering with improved:
    - Normalized embeddings for cosine similarity
    - Dimensionality reduction before clustering
    - Better T5 usage with task prefixes
    - Map-reduce for long texts
    - Optimized TF-IDF processing
    """
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.clusters = None
        self.cluster_summaries = {}
        self.cluster_narratives = {}
        self.cluster_names = {}
        self.cluster_keywords = {}
        self.current_cluster_id = None
        
        # Determine compute device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Enhanced Theme Clustering Tool with Advanced AI Narratives - Improved Version")
        print("=" * 80)
        self.setup_models()
        
    def setup_models(self):
        """Setup enhanced models with better configuration"""
        try:
            print("Loading Sentence-BERT model...")
            self.sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2', device=self.device
            )
            
            print("Loading T5-base for enhanced narrative generation...")
            try:
                self.summarizer = pipeline(
                    "summarization", 
                    model="t5-base",
                    device=0 if self.device == 'cuda' else -1,
                    max_length=220,
                    min_length=90
                )
                print(f"Device set to use {self.device}")
                print("T5-base loaded successfully!")
            except Exception as e:
                print(f"T5-base failed, falling back to DistilBART: {e}")
                self.summarizer = pipeline(
                    "summarization", 
                    model="sshleifer/distilbart-cnn-12-6",
                    device=0 if self.device == 'cuda' else -1,
                    max_length=200,
                    min_length=80
                )
                print("DistilBART loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Will use fallback narrative generation...")
            self.summarizer = None
    
    def load_and_clean_data(self, topic_filter=None):
        """Load and clean the dataset"""
        print("\nLoading and cleaning data...")
        
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} articles")
        
        # Parse dates
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Parse journalists (handle string representation of lists)
        def parse_journalists(journalists_str):
            try:
                return ast.literal_eval(journalists_str) if isinstance(journalists_str, str) else [journalists_str]
            except:
                return [journalists_str] if journalists_str else []
        
        self.df['Journalists_List'] = self.df['Journalists'].apply(parse_journalists)
        
        # Create full text for analysis
        self.df['Full_Text'] = self.df['Headline'] + " " + self.df['Article'].fillna("")
        self.df['Text_Length'] = self.df['Full_Text'].str.len()
        self.df['Word_Count'] = self.df['Full_Text'].str.split().str.len()
        
        # Optional topic filtering
        if topic_filter:
            if isinstance(topic_filter, str):
                filter_expr = topic_filter
            else:
                filter_expr = '|'.join(topic_filter)
            mask = self.df['Full_Text'].str.contains(filter_expr, case=False, na=False)
            self.df = self.df[mask]
            print(f"Topic filter applied - {len(self.df)} articles remaining")
        
        # Clean data
        initial_count = len(self.df)
        self.df = self.df[self.df['Text_Length'] > 50]  # Remove very short articles
        self.df = self.df.dropna(subset=['Full_Text'])
        filtered_count = len(self.df)
        
        print(f"Removed {initial_count - filtered_count} articles in cleaning")
        print(f"Final dataset: {filtered_count} articles ready for clustering")
        
        return self.df
    
    def generate_embeddings(self, sample_size=None):
        """Generate normalized semantic embeddings for clustering"""
        print("\nGenerating semantic embeddings...")
        
        if sample_size and len(self.df) > sample_size:
            df_sample = self.df.sample(n=sample_size, random_state=SEED)
            print(f"Analyzing sample of {sample_size} articles")
        else:
            df_sample = self.df
            print(f"Analyzing all {len(df_sample)} articles")
        
        self.analysis_df = df_sample.copy().reset_index(drop=True)
        texts = self.analysis_df['Full_Text'].tolist()
        
        # Generate normalized embeddings (crucial for cosine similarity)
        print("Computing normalized semantic embeddings...")
        self.embeddings = self.sentence_model.encode(
            texts, 
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Key improvement
        )
        
        print(f"Generated embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def reduce_dimensions(self, n_components=50):
        """Reduce dimensions before clustering for better performance and stability"""
        print(f"\nReducing dimensions to {n_components} components...")
        
        if UMAP_AVAILABLE:
            try:
                reducer = umap.UMAP(
                    n_components=n_components, 
                    random_state=SEED,
                    min_dist=0.1,
                    n_neighbors=15
                )
                self.reduced_embeddings = reducer.fit_transform(self.embeddings)
                method = "UMAP"
            except Exception as e:
                print(f"UMAP failed: {e}, falling back to PCA")
                reducer = PCA(n_components=n_components, random_state=SEED)
                self.reduced_embeddings = reducer.fit_transform(self.embeddings)
                method = "PCA"
        else:
            reducer = PCA(n_components=n_components, random_state=SEED)
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)
            method = "PCA"
        
        print(f"Reduced embeddings using {method}: {self.reduced_embeddings.shape}")
        return self.reduced_embeddings
    
    def cluster_themes(self, min_cluster_size=5):
        """Cluster articles using improved HDBSCAN on reduced dimensions"""
        print(f"\nClustering themes (min cluster size: {min_cluster_size})...")
        
        # Reduce dimensions first
        self.reduce_dimensions()
        
        # Use HDBSCAN on reduced dimensions with euclidean metric
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',  # Better for reduced dimensions
            min_samples=5,
            cluster_selection_epsilon=0.1,
            cluster_selection_method='eom'
        )
        
        self.clusters = clusterer.fit_predict(self.reduced_embeddings)
        
        # Add cluster assignments to dataframe
        self.analysis_df['Cluster'] = self.clusters
        
        # Analyze results
        unique_clusters = np.unique(self.clusters)
        n_clusters = len(unique_clusters[unique_clusters >= 0])
        n_noise = np.sum(self.clusters == -1)
        
        print(f"Clustering Results:")
        print(f"  - Found {n_clusters} thematic clusters")
        print(f"  - {n_noise} articles classified as noise/outliers")
        
        # Show cluster sizes
        cluster_sizes = Counter(self.clusters[self.clusters >= 0])
        print(f"  - Cluster sizes: {dict(cluster_sizes)}")
        
        return self.clusters
    
    def extract_cluster_keywords(self):
        """Extract representative keywords with optimized TF-IDF processing"""
        print("\nExtracting enhanced cluster keywords...")
        
        # Enhanced stop words for business/news content
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        
        business_stop_words = {
            'percent', 'said', 'says', 'company', 'new', 'year', 'years', 'million', 'billion',
            'according', 'reported', 'reuters', 'bloomberg', 'told', 'chief', 'executive',
            'market', 'stock', 'share', 'shares', 'trading', 'traded', 'close', 'opened',
            'analyst', 'analysts', 'estimate', 'estimates', 'quarter', 'quarterly', 'annual',
            'revenue', 'profit', 'earnings', 'sales', 'financial', 'fiscal', 'business',
            'news', 'today', 'yesterday', 'week', 'month', 'time', 'times', 'report',
            'announcement', 'announced', 'statement', 'press', 'release', 'conference'
        }
        
        enhanced_stop_words = set(ENGLISH_STOP_WORDS) | business_stop_words
        
        # Better token pattern for news terms (captures hyphenated/slash terms)
        tfidf = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 4),
            min_df=1,
            max_df=0.6,
            stop_words=list(enhanced_stop_words),
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-\/&]*\b"  # Improved pattern
        )
        
        self.cluster_keywords = {}
        
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            cluster_data = self.analysis_df[self.clusters == cluster_id]
            cluster_texts = cluster_data['Full_Text'].tolist()
            
            if len(cluster_texts) == 0:
                continue
            
            try:
                # Optimized TF-IDF processing
                tfidf_matrix = tfidf.fit_transform(cluster_texts)
                feature_names = tfidf.get_feature_names_out()
                
                # Use sparse matrix operations (avoid toarray())
                avg_scores = tfidf_matrix.mean(axis=0).A1  # More efficient
                
                # Get top keywords
                top_indices = avg_scores.argsort()[-15:][::-1]
                tfidf_keywords = [feature_names[i] for i in top_indices]
                
                # Extract proper nouns from headlines
                headline_text = ' '.join(cluster_data['Headline'].tolist())
                proper_nouns = self.extract_proper_nouns(headline_text)
                
                # Combine keywords
                all_keywords = []
                
                # Add TF-IDF keywords
                for i, keyword in enumerate(tfidf_keywords):
                    score = avg_scores[top_indices[i]]
                    all_keywords.append((keyword, score, 'tfidf'))
                
                # Add proper nouns
                for noun, freq in proper_nouns:
                    score = freq * 2.0
                    all_keywords.append((noun, score, 'proper_noun'))
                
                # Remove duplicates and finalize
                seen = set()
                final_keywords = []
                for keyword, score, source in sorted(all_keywords, key=lambda x: x[1], reverse=True):
                    if keyword.lower() not in seen and len(keyword) > 2:
                        seen.add(keyword.lower())
                        final_keywords.append(keyword)
                        if len(final_keywords) >= 10:
                            break
                
                self.cluster_keywords[cluster_id] = final_keywords
                
            except Exception as e:
                print(f"Error extracting keywords for cluster {cluster_id}: {e}")
                self.cluster_keywords[cluster_id] = self.fallback_keyword_extraction(cluster_texts)
        
        return self.cluster_keywords
    
    def extract_proper_nouns(self, text):
        """Extract proper nouns and named entities from text"""
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        common_caps = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But', 'So', 'For', 'Nor', 'Yet'}
        proper_nouns = [word for word in words if word not in common_caps and len(word) > 2]
        noun_counts = Counter(proper_nouns)
        return noun_counts.most_common(10)
    
    def fallback_keyword_extraction(self, cluster_texts):
        """Fallback keyword extraction using simple word frequency"""
        combined_text = ' '.join(cluster_texts).lower()
        words = re.findall(r'\b[a-z]{3,}\b', combined_text)
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after'}
        filtered_words = [word for word in words if word not in stop_words]
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(10)]
    
    def create_cluster_summaries(self):
        """Create summary information for each cluster"""
        print("\nCreating cluster summaries...")
        
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            cluster_data = self.analysis_df[self.clusters == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Basic statistics
            cluster_size = len(cluster_data)
            date_range = f"{cluster_data['Date'].min().strftime('%Y-%m-%d')} to {cluster_data['Date'].max().strftime('%Y-%m-%d')}"
            avg_word_count = cluster_data['Word_Count'].mean()
            
            # Get representative headlines
            top_headlines = cluster_data.nlargest(3, 'Word_Count')['Headline'].tolist()
            
            # Get keywords
            keywords = self.cluster_keywords.get(cluster_id, [])
            
            # Get authors
            all_authors = []
            for authors_list in cluster_data['Journalists_List']:
                if isinstance(authors_list, list):
                    all_authors.extend([a for a in authors_list if a and str(a) != 'nan'])
            author_counts = Counter(all_authors)
            top_authors = [author for author, count in author_counts.most_common(5)]
            
            summary = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'date_range': date_range,
                'avg_word_count': avg_word_count,
                'keywords': keywords[:5],
                'top_headlines': top_headlines,
                'top_authors': top_authors,
                'sample_articles': cluster_data.head(3)[['Date', 'Headline', 'Journalists']].to_dict('records')
            }
            
            self.cluster_summaries[cluster_id] = summary
    
    def chunk_text_for_summarization(self, text, max_chunk_size=400):
        """Split long text into manageable chunks for summarization"""
        words = text.split()
        if len(words) <= max_chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(words), max_chunk_size):
            chunk_words = words[i:i + max_chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
        
        return chunks
    
    def map_reduce_summarization(self, text):
        """Use map-reduce approach for long texts"""
        if not self.summarizer:
            return text[:500] + "..."
        
        try:
            # Step 1: Chunk the text
            chunks = self.chunk_text_for_summarization(text, max_chunk_size=350)
            
            if len(chunks) == 1:
                # Single chunk - direct summarization with T5 task prefix
                prompt = f"summarize: {chunks[0]}"
                result = self.summarizer(
                    prompt,
                    max_length=220,
                    min_length=90,
                    num_beams=4,
                    do_sample=False,
                    no_repeat_ngram_size=3
                )
                return result[0]['summary_text']
            
            # Step 2: Summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                prompt = f"summarize: {chunk}"
                try:
                    result = self.summarizer(
                        prompt,
                        max_length=150,
                        min_length=50,
                        num_beams=4,
                        do_sample=False,
                        no_repeat_ngram_size=3
                    )
                    chunk_summaries.append(result[0]['summary_text'])
                except Exception as e:
                    print(f"Warning: chunk summarization failed: {e}")
                    chunk_summaries.append(chunk[:200] + "...")
            
            # Step 3: Combine and summarize the summaries
            combined_summaries = ' '.join(chunk_summaries)
            if len(combined_summaries.split()) > 300:
                prompt = f"summarize: {combined_summaries}"
                final_result = self.summarizer(
                    prompt,
                    max_length=250,
                    min_length=100,
                    num_beams=4,
                    do_sample=False,
                    no_repeat_ngram_size=3
                )
                return final_result[0]['summary_text']
            else:
                return combined_summaries
            
        except Exception as e:
            print(f"Map-reduce summarization failed: {e}")
            return text[:500] + "..."
    
    def prepare_cluster_text_for_summarization(self, cluster_data):
        """Enhanced text preparation using semantic similarity"""
        if len(self.embeddings) == 0:
            return self._fallback_text_preparation(cluster_data)
        
        try:
            # Get embeddings for this cluster
            cluster_indices = cluster_data.index.tolist()
            cluster_embeddings = self.embeddings[cluster_indices]
            
            # Calculate cluster centroid
            cluster_center = np.mean(cluster_embeddings, axis=0)
            
            # Find most representative articles
            similarities = cosine_similarity([cluster_center], cluster_embeddings)[0]
            top_indices = similarities.argsort()[-5:][::-1]
            representative_articles = cluster_data.iloc[top_indices]
            
            # Prepare text sections
            text_sections = []
            for _, article in representative_articles.iterrows():
                headline = article['Headline']
                article_text = article['Article'] if pd.notna(article['Article']) else ""
                full_text = f"{headline}. {article_text}"
                
                # Use complete sentences
                words = full_text.split()
                if len(words) > 400:
                    truncated_text = ' '.join(words[:400])
                    sentences = truncated_text.split('. ')
                    if len(sentences) > 1:
                        complete_text = '. '.join(sentences[:-1]) + '.'
                    else:
                        complete_text = truncated_text + '.'
                else:
                    complete_text = full_text
                
                text_sections.append(complete_text)
            
            combined_text = '\n\n'.join(text_sections)
            return combined_text
            
        except Exception as e:
            print(f"Error in semantic text preparation: {e}")
            return self._fallback_text_preparation(cluster_data)
    
    def _fallback_text_preparation(self, cluster_data):
        """Fallback text preparation method"""
        recent_articles = cluster_data.nlargest(3, 'Date')
        longest_articles = cluster_data.nlargest(3, 'Word_Count')
        combined_indices = set(recent_articles.index) | set(longest_articles.index)
        representative_articles = cluster_data.loc[list(combined_indices)]
        
        text_parts = []
        for _, article in representative_articles.iterrows():
            headline = article['Headline']
            article_snippet = article['Article'][:300] if pd.notna(article['Article']) else ""
            text_parts.append(f"{headline}. {article_snippet}")
        
        return '\n\n'.join(text_parts)
    
    def generate_ai_narrative(self, cluster_text, cluster_name, cluster_size):
        """Enhanced narrative generation with map-reduce and better prompting"""
        if not self.summarizer:
            return f"AI summarization not available. {cluster_name} contains {cluster_size} articles covering related themes."
        
        try:
            # Use map-reduce for long texts
            ai_summary = self.map_reduce_summarization(cluster_text)
            
            # Create narrative with better framing
            narrative = f"The narrative across {cluster_size} articles reveals: {ai_summary}"
            
            return narrative
            
        except Exception as e:
            print(f"    Warning: AI narrative generation failed for cluster {cluster_name}: {e}")
            return self._generate_fallback_narrative(cluster_text, cluster_name, cluster_size)
    
    def _generate_fallback_narrative(self, cluster_text, cluster_name, cluster_size):
        """Generate fallback narrative when AI fails"""
        sentences = cluster_text.split('. ')
        key_sentences = [s for s in sentences[:5] if len(s.split()) > 8]
        
        if key_sentences:
            fallback_summary = '. '.join(key_sentences[:2]) + '.'
            return (f"The {cluster_name} cluster contains {cluster_size} articles covering: "
                   f"{fallback_summary}")
        else:
            return (f"The {cluster_name} cluster contains {cluster_size} articles covering "
                   f"related developments in this thematic area.")
    
    def extract_theme_from_summary(self, ai_summary):
        """Extract a concise theme name from AI summary"""
        summary_clean = ai_summary.replace("The narrative across", "")
        summary_clean = summary_clean.replace("articles reveals:", "").strip()
        
        # Extract key phrases
        sentences = summary_clean.split('. ')
        first_sentence = sentences[0] if sentences else summary_clean
        words = first_sentence.split()
        
        # Find proper nouns and important terms
        important_terms = []
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                entity = [word]
                j = i + 1
                while j < len(words) and j < i + 3:
                    if words[j][0].isupper() and len(words[j]) > 2:
                        entity.append(words[j])
                        j += 1
                    else:
                        break
                
                if len(entity) > 0:
                    important_terms.append(' '.join(entity))
        
        # Remove duplicates and select best terms
        unique_terms = []
        seen = set()
        for term in important_terms:
            if term.lower() not in seen and len(term) > 3:
                seen.add(term.lower())
                unique_terms.append(term)
                if len(unique_terms) >= 2:
                    break
        
        if unique_terms:
            if len(unique_terms) == 1:
                return unique_terms[0]
            else:
                return " & ".join(unique_terms[:2])
        else:
            meaningful_words = [w for w in words[:10] 
                             if len(w) > 3 and w.lower() not in 
                             ['that', 'this', 'with', 'from', 'they', 'have', 'been', 'were']]
            
            if meaningful_words:
                return ' '.join(meaningful_words[:3]).title()
            else:
                return "News Theme"
    
    def generate_cluster_narratives(self):
        """Enhanced cluster narrative generation"""
        print("\nGenerating enhanced AI-powered cluster narratives...")
        
        self.cluster_narratives = {}
        
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            cluster_data = self.analysis_df[self.clusters == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            print(f"  Analyzing cluster {cluster_id} with enhanced AI...")
            
            # Store current cluster ID
            self.current_cluster_id = cluster_id
            
            # Get cluster metadata
            cluster_size = len(cluster_data)
            keywords = self.cluster_keywords.get(cluster_id, [])[:5]
            
            # Prepare text and generate narrative
            cluster_text = self.prepare_cluster_text_for_summarization(cluster_data)
            ai_narrative = self.generate_ai_narrative(cluster_text, f"cluster {cluster_id}", cluster_size)
            
            # Add analytical context
            analytical_context = self.add_analytical_context(cluster_data, keywords)
            
            # Combine narratives
            full_narrative = f"{ai_narrative}\n\n{analytical_context}"
            
            # Generate theme name
            theme_name = self.extract_theme_from_summary(ai_narrative)
            
            self.cluster_narratives[cluster_id] = {
                'cluster_name': theme_name,
                'narrative_text': full_narrative,
                'ai_summary': ai_narrative,
                'analytical_context': analytical_context,
                'size': cluster_size,
                'key_keywords': keywords
            }
        
        # Update cluster names
        for cluster_id, narrative_data in self.cluster_narratives.items():
            self.cluster_names[cluster_id] = narrative_data['cluster_name']
        
        return self.cluster_narratives
    
    def add_analytical_context(self, cluster_data, keywords):
        """Add analytical context about coverage patterns"""
        date_range = cluster_data['Date'].max() - cluster_data['Date'].min()
        days_span = date_range.days + 1
        
        articles_per_day = len(cluster_data) / max(days_span, 1)
        if articles_per_day > 2:
            intensity = "intensive daily coverage"
        elif articles_per_day > 0.5:
            intensity = "regular coverage"
        else:
            intensity = "sporadic coverage"
        
        # Author diversity
        all_authors = []
        for authors_list in cluster_data['Journalists_List']:
            if isinstance(authors_list, list):
                all_authors.extend([a for a in authors_list if a and str(a) != 'nan'])
        unique_authors = len(set(all_authors))
        
        if unique_authors > len(cluster_data) * 0.7:
            author_pattern = "diverse journalistic perspectives"
        elif unique_authors < len(cluster_data) * 0.3:
            author_pattern = "concentrated reporting"
        else:
            author_pattern = "mixed coverage"
        
        # Article depth
        avg_words = cluster_data['Word_Count'].mean()
        if avg_words > 800:
            depth = "in-depth analysis"
        elif avg_words > 500:
            depth = "detailed reporting"
        else:
            depth = "concise coverage"
        
        context_parts = [
            f"Coverage Analysis: This story received {intensity} over {days_span} days, "
            f"characterized by {author_pattern} and {depth}."
        ]
        
        if keywords:
            context_parts.append(f"Key themes identified: {', '.join(keywords[:5])}.")
        
        if len(cluster_data) > 15:
            significance = "This represents a major ongoing story with sustained media attention."
        elif len(cluster_data) > 8:
            significance = "This constitutes a significant news development."
        else:
            significance = "This reflects a notable news event."
        
        context_parts.append(significance)
        
        return " ".join(context_parts)
    
    def visualize_clusters(self):
        """Create enhanced visualization"""
        print("\nCreating enhanced cluster visualization...")
        
        if not self.cluster_names:
            print("Warning: No cluster names found, using fallback naming...")
            self.cluster_names = {}
            for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
                self.cluster_names[cluster_id] = f"Theme {cluster_id}"
        
        # Use reduced embeddings for visualization
        if UMAP_AVAILABLE:
            try:
                reducer = umap.UMAP(n_components=2, random_state=SEED, min_dist=0.1)
                embeddings_2d = reducer.fit_transform(self.embeddings)
                method = "UMAP"
            except:
                pca = PCA(n_components=2, random_state=SEED)
                embeddings_2d = pca.fit_transform(self.embeddings)
                method = "PCA"
        else:
            pca = PCA(n_components=2, random_state=SEED)
            embeddings_2d = pca.fit_transform(self.embeddings)
            method = "PCA"
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot each cluster with different colors
        unique_clusters = np.unique(self.clusters)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        # Store points for hover functionality (simulated with annotations)
        cluster_centers = {}
        
        for cluster_id, color in zip(unique_clusters, colors):
            mask = self.clusters == cluster_id
            cluster_points = embeddings_2d[mask]
            
            if cluster_id == -1:
                # Noise points in grey
                scatter = ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                   c='lightgrey', alpha=0.6, s=50, label='Uncategorized')
            else:
                # Get cluster name
                cluster_name = self.cluster_names.get(cluster_id, f"Theme {cluster_id}")
                
                # Plot cluster points
                scatter = ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                   c=[color], alpha=0.7, s=60, 
                                   label=f'{cluster_name} ({len(cluster_points)} articles)')
                
                # Calculate cluster center for label placement
                center_x = np.mean(cluster_points[:, 0])
                center_y = np.mean(cluster_points[:, 1])
                cluster_centers[cluster_id] = (center_x, center_y)
                
                # Add cluster name annotation at center
                ax.annotate(cluster_name, 
                          (center_x, center_y),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=10, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                          ha='center')
        
        # Enhance the plot
        ax.set_title(f'Article Clusters by Theme ({method} Visualization)', fontsize=16, fontweight='bold')
        ax.set_xlabel(f'{method} Component 1', fontsize=12)
        ax.set_ylabel(f'{method} Component 2', fontsize=12)
        
        # Customize legend
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        legend.set_title("Clusters", prop={'size': 12, 'weight': 'bold'})
        
        plt.tight_layout()
        plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create a second plot with article titles visible
        self.create_detailed_visualization(embeddings_2d, method)
        
        print("Visualizations saved:")
        print("- cluster_visualization.png (overview)")
        print("- detailed_cluster_view.png (with article titles)")

    def create_detailed_visualization(self, embeddings_2d, method):
        """Create a detailed view showing individual article information"""
        fig, ax = plt.subplots(figsize=(20, 12))
        
        unique_clusters = np.unique(self.clusters)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster_id, color in zip(unique_clusters, colors):
            mask = self.clusters == cluster_id
            cluster_points = embeddings_2d[mask]
            cluster_articles = self.analysis_df[mask]
            
            if cluster_id == -1:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c='lightgrey', alpha=0.6, s=30)
            else:
                cluster_name = self.cluster_names.get(cluster_id, f"Theme {cluster_id}")
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=[color], alpha=0.7, s=40, label=cluster_name)
                
                # Add article titles for smaller clusters (to avoid overcrowding)
                if len(cluster_articles) <= 15:  # Reduced from 20 to prevent overcrowding
                    for i, (point, article) in enumerate(zip(cluster_points, cluster_articles.itertuples())):
                        headline = article.Headline[:40] + "..." if len(article.Headline) > 40 else article.Headline
                        ax.annotate(headline, 
                                  (point[0], point[1]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=7, alpha=0.8,
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Detailed Article View ({method})', fontsize=16, fontweight='bold')
        ax.set_xlabel(f'{method} Component 1', fontsize=12)
        ax.set_ylabel(f'{method} Component 2', fontsize=12)
        
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.set_title("Themes", prop={'size': 12, 'weight': 'bold'})
        
        plt.tight_layout()
        plt.savefig('detailed_cluster_view.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close to save memory

    def print_narrative_report(self):
        """Print comprehensive narrative analysis report"""
        print("\n" + "="*90)
        print("ENHANCED NARRATIVE INTELLIGENCE REPORT")
        print("="*90)
        
        print(f"\nExecutive Summary:")
        print(f"  Total articles analyzed: {len(self.analysis_df)}")
        print(f"  Distinct narratives identified: {len(self.cluster_narratives)}")
        print(f"  Analysis period: {self.analysis_df['Date'].min().strftime('%Y-%m-%d')} to {self.analysis_df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Sort by importance (cluster size)
        sorted_narratives = sorted(
            self.cluster_narratives.items(),
            key=lambda x: x[1]['size'],
            reverse=True
        )
        
        for cluster_id, narrative_data in sorted_narratives:
            print(f"\n" + "─" * 80)
            print(f"📰 {narrative_data['cluster_name'].upper()} ({narrative_data['size']} Articles)")
            print("─" * 80)
            
            print(f"\n{narrative_data['narrative_text']}")
            
            print(f"\n📊 Coverage Metrics:")
            print(f"   • Articles: {narrative_data['size']}")
            
            if narrative_data['key_keywords']:
                print(f"   • Key themes: {', '.join(narrative_data['key_keywords'])}")
        
        print("\n" + "="*90)
    
    def export_results(self, output_prefix="enhanced_theme_clustering"):
        """Export clustering results to CSV files"""
        print(f"\nExporting results...")
        
        # Export main dataset with cluster assignments
        self.analysis_df.to_csv(f"{output_prefix}_articles.csv", index=False)
        
        # Export cluster summaries
        summaries_data = []
        for cluster_id, summary in self.cluster_summaries.items():
            row = {
                'cluster_id': cluster_id,
                'size': summary['size'],
                'date_range': summary['date_range'],
                'avg_word_count': summary['avg_word_count'],
                'keywords': ', '.join(summary['keywords']),
                'top_authors': ', '.join(summary['top_authors'])
            }
            summaries_data.append(row)
        
        summaries_df = pd.DataFrame(summaries_data)
        summaries_df.to_csv(f"{output_prefix}_summaries.csv", index=False)
        
        # Export narratives if available
        if hasattr(self, 'cluster_narratives') and self.cluster_narratives:
            narratives_data = []
            for cluster_id, narrative in self.cluster_narratives.items():
                row = {
                    'cluster_id': cluster_id,
                    'cluster_name': narrative['cluster_name'],
                    'size': narrative['size'],
                    'ai_summary': narrative['ai_summary'],
                    'analytical_context': narrative['analytical_context'],
                    'full_narrative': narrative['narrative_text'],
                    'keywords': ', '.join(narrative['key_keywords'])
                }
                narratives_data.append(row)
            
            narratives_df = pd.DataFrame(narratives_data)
            narratives_df.to_csv(f"{output_prefix}_narratives.csv", index=False)
            
            print(f"Results exported:")
            print(f"  - {output_prefix}_articles.csv (articles with cluster assignments)")
            print(f"  - {output_prefix}_summaries.csv (cluster summaries)")
            print(f"  - {output_prefix}_narratives.csv (AI-generated narratives)")
            print(f"  - cluster_visualization.png (visual representation)")
        else:
            print(f"Results exported:")
            print(f"  - {output_prefix}_articles.csv (articles with cluster assignments)")
            print(f"  - {output_prefix}_summaries.csv (cluster summaries)")
            print(f"  - cluster_visualization.png (visual representation)")
    
    def run_clustering_analysis(self, sample_size=None, min_cluster_size=5, topic_filter=None):
        """Enhanced main method to run the complete clustering analysis"""
        print("Starting enhanced theme clustering analysis...")
        print("="*80)
        
        try:
            # Step 1: Load and clean data
            self.load_and_clean_data(topic_filter=topic_filter)
            
            # Step 2: Generate normalized embeddings
            self.generate_embeddings(sample_size=sample_size)
            
            # Step 3: Cluster themes with dimension reduction
            self.cluster_themes(min_cluster_size=min_cluster_size)
            
            # Step 4: Extract optimized keywords
            self.extract_cluster_keywords()
            
            # Step 5: Create summaries
            self.create_cluster_summaries()
            
            # Step 6: Generate enhanced narratives with map-reduce
            self.generate_cluster_narratives()
            
            # Step 7: Visualize results
            self.visualize_clusters()
            
            # Step 8: Print narrative report
            self.print_narrative_report()
            
            # Step 9: Export results
            self.export_results()
            
            print("\nEnhanced clustering analysis completed successfully!")
            print("\nKey Improvements Applied:")
            print("  ✓ Normalized embeddings for better cosine similarity")
            print("  ✓ Dimensionality reduction before clustering for stability")
            print("  ✓ Fixed T5 usage with proper task prefixes")
            print("  ✓ Map-reduce approach for long text summarization")
            print("  ✓ Optimized TF-IDF processing with sparse matrices")
            print("  ✓ Better token patterns for news terminology")
            print("  ✓ Deterministic beam search for consistent results")
            print("  ✓ Improved error handling and logging")
            
            return True
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Command line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced theme clustering with improved performance and quality")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--sample-size", type=int, help="Maximum articles to analyze")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum cluster size")
    parser.add_argument("--topic", nargs="*", help="Optional keyword(s) to filter articles")
    
    args = parser.parse_args()
    
    print("Enhanced Theme Clustering Tool - Improved Version")
    print("="*60)
    print(f"Dataset: {args.csv}")
    if args.topic:
        print(f"Topic filter: {', '.join(args.topic)}")
    print()
    
    try:
        clusterer = EnhancedThemeClustering(args.csv)
        success = clusterer.run_clustering_analysis(
            sample_size=args.sample_size,
            min_cluster_size=args.min_cluster_size,
            topic_filter=args.topic
        )
        
        if success:
            print("\nAnalysis completed! Check the generated files:")
            print("- cluster_visualization.png")
            print("- detailed_cluster_view.png")
            print("- enhanced_theme_clustering_articles.csv") 
            print("- enhanced_theme_clustering_summaries.csv")
            print("- enhanced_theme_clustering_narratives.csv")
        
    except Exception as e:
        print(f"Critical Error: {e}")
        raise


if __name__ == "__main__":
    main()