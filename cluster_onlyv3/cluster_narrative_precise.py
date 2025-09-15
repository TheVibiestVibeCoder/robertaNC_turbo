#!/usr/bin/env python3
"""
SIMPLE THEME CLUSTERING SCRIPT WITH AI NARRATIVES
Focuses on data ingestion, clustering articles by thematic similarity, and AI-powered narrative generation
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

from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

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


class SimpleThemeClustering:
    """
    Simple theme clustering that focuses on:
    - Clean data ingestion
    - Semantic embedding generation
    - High-quality clustering
    - AI-powered narrative generation
    - Clear visualization of results
    """
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.embeddings = None
        self.clusters = None
        self.cluster_summaries = {}
        self.cluster_narratives = {}
        
        # Determine compute device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Simple Theme Clustering Tool")
        print("=" * 50)
        self.setup_models()
        
    def setup_models(self):
        """Setup embedding model and narrative generation for semantic similarity"""
        try:
            print("Loading Sentence-BERT model...")
            self.sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2', device=self.device
            )
            
            print("Loading DistilBART for narrative generation...")
            # Use DistilBART - lightweight but effective for news summarization
            self.summarizer = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6",
                device=0 if self.device == 'cuda' else -1,
                max_length=200,
                min_length=80,
                do_sample=False
            )
            
            print("Models loaded successfully!")
            
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
        """Generate semantic embeddings for clustering"""
        print("\nGenerating semantic embeddings...")
        
        if sample_size and len(self.df) > sample_size:
            df_sample = self.df.sample(n=sample_size, random_state=42)
            print(f"Analyzing sample of {sample_size} articles")
        else:
            df_sample = self.df
            print(f"Analyzing all {len(df_sample)} articles")
        
        self.analysis_df = df_sample.copy().reset_index(drop=True)
        texts = self.analysis_df['Full_Text'].tolist()
        
        # Generate embeddings
        print("Computing semantic embeddings...")
        self.embeddings = self.sentence_model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=32
        )
        
        print(f"Generated embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def cluster_themes(self, min_cluster_size=5):
        """Cluster articles by thematic similarity"""
        print(f"\nClustering themes (min cluster size: {min_cluster_size})...")
        
        # Use HDBSCAN for density-based clustering
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='cosine',
            cluster_selection_epsilon=0.15,
            min_samples=3,
            cluster_selection_method='eom'
        )
        
        self.clusters = clusterer.fit_predict(self.embeddings)
        
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
        """Extract representative keywords for each cluster with domain-specific improvements"""
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
        
        # Setup enhanced TF-IDF for keyword extraction
        tfidf = TfidfVectorizer(
            max_features=1500,               # Increased from 1000
            ngram_range=(1, 4),              # Capture longer phrases like "Steve Jobs"
            min_df=1,                        # Lower threshold for specific names/terms
            max_df=0.6,                      # More aggressive filtering of common terms
            stop_words=list(enhanced_stop_words),
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Better token recognition
        )
        
        self.cluster_keywords = {}
        
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            cluster_data = self.analysis_df[self.clusters == cluster_id]
            cluster_texts = cluster_data['Full_Text'].tolist()
            
            if len(cluster_texts) == 0:
                continue
            
            # Enhanced keyword extraction with multiple approaches
            try:
                # Primary TF-IDF approach
                tfidf_matrix = tfidf.fit_transform(cluster_texts)
                feature_names = tfidf.get_feature_names_out()
                
                # Get average TF-IDF scores
                avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                
                # Get top keywords from TF-IDF
                top_indices = avg_scores.argsort()[-15:][::-1]
                tfidf_keywords = [feature_names[i] for i in top_indices]
                
                # Extract proper nouns and named entities from headlines
                headline_text = ' '.join(cluster_data['Headline'].tolist())
                proper_nouns = self.extract_proper_nouns(headline_text)
                
                # Combine and rank keywords
                all_keywords = []
                
                # Add TF-IDF keywords with scores
                for i, keyword in enumerate(tfidf_keywords):
                    score = avg_scores[top_indices[i]]
                    all_keywords.append((keyword, score, 'tfidf'))
                
                # Add proper nouns with frequency-based scores
                for noun, freq in proper_nouns:
                    # Boost score for proper nouns as they're often more meaningful
                    score = freq * 2.0
                    all_keywords.append((noun, score, 'proper_noun'))
                
                # Sort by score and remove duplicates
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
                # Fallback to simple word frequency
                self.cluster_keywords[cluster_id] = self.fallback_keyword_extraction(cluster_texts)
        
        return self.cluster_keywords
    
    def extract_proper_nouns(self, text):
        """Extract proper nouns and named entities from text"""
        # Simple proper noun extraction based on capitalization patterns
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Filter out common words that might be capitalized
        common_caps = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But', 'So', 'For', 'Nor', 'Yet'}
        proper_nouns = [word for word in words if word not in common_caps and len(word) > 2]
        
        # Count frequency
        noun_counts = Counter(proper_nouns)
        
        # Return top proper nouns with their frequencies
        return noun_counts.most_common(10)
    
    def fallback_keyword_extraction(self, cluster_texts):
        """Fallback keyword extraction using simple word frequency"""
        # Combine all cluster texts
        combined_text = ' '.join(cluster_texts).lower()
        
        # Extract words, filtering out stop words and short words
        words = re.findall(r'\b[a-z]{3,}\b', combined_text)
        
        # Basic stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'over', 'after'}
        
        # Filter and count
        filtered_words = [word for word in words if word not in stop_words]
        word_counts = Counter(filtered_words)
        
        # Return top words
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
                'keywords': keywords[:5],  # Top 5 keywords
                'top_headlines': top_headlines,
                'top_authors': top_authors,
                'sample_articles': cluster_data.head(3)[['Date', 'Headline', 'Journalists']].to_dict('records')
            }
            
            self.cluster_summaries[cluster_id] = summary
    
    def generate_cluster_names(self):
        """Generate descriptive names for clusters based on keywords"""
        print("\nGenerating cluster names...")
        
        self.cluster_names = {}
        
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            keywords = self.cluster_keywords.get(cluster_id, [])
            
            if len(keywords) >= 2:
                # Use top 2-3 keywords to create a descriptive name
                name = " & ".join(keywords[:2]).title()
            elif len(keywords) == 1:
                name = keywords[0].title()
            else:
                # Fallback to most common words in headlines
                cluster_data = self.analysis_df[self.clusters == cluster_id]
                headlines = ' '.join(cluster_data['Headline'].tolist()).lower()
                words = re.findall(r'\b\w+\b', headlines)
                common_words = [word for word, count in Counter(words).most_common(5) 
                              if len(word) > 3 and word not in ['news', 'says', 'said', 'will', 'year']]
                name = common_words[0].title() if common_words else f"Theme {cluster_id}"
            
            self.cluster_names[cluster_id] = name
        
        return self.cluster_names

    def generate_cluster_narratives(self):
        """Generate AI-powered natural language narratives for each cluster"""
        print("\nGenerating AI-powered cluster narratives...")
        
        self.cluster_narratives = {}
        
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            cluster_data = self.analysis_df[self.clusters == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            print(f"  Analyzing cluster {cluster_id} with DistilBART...")
            
            # Get cluster metadata
            cluster_name = self.cluster_names.get(cluster_id, f"Theme {cluster_id}")
            cluster_size = len(cluster_data)
            keywords = self.cluster_keywords.get(cluster_id, [])[:5]
            
            # Prepare text for summarization
            cluster_text = self.prepare_cluster_text_for_summarization(cluster_data)
            
            # Generate AI narrative
            ai_narrative = self.generate_ai_narrative(cluster_text, cluster_name, cluster_size)
            
            # Add analytical context
            analytical_context = self.add_analytical_context(cluster_data, keywords)
            
            # Combine AI narrative with analytical insights
            full_narrative = f"{ai_narrative}\n\n{analytical_context}"
            
            self.cluster_narratives[cluster_id] = {
                'cluster_name': cluster_name,
                'narrative_text': full_narrative,
                'ai_summary': ai_narrative,
                'analytical_context': analytical_context,
                'size': cluster_size,
                'key_keywords': keywords
            }
        
        return self.cluster_narratives

    def prepare_cluster_text_for_summarization(self, cluster_data):
        """Prepare cluster articles for AI summarization"""
        # Get the most representative articles (by length and recency)
        recent_articles = cluster_data.nlargest(3, 'Date')
        longest_articles = cluster_data.nlargest(3, 'Word_Count')
        
        # Combine unique articles from both selections using index-based deduplication
        combined_indices = set(recent_articles.index) | set(longest_articles.index)
        representative_articles = cluster_data.loc[list(combined_indices)]
        
        # Combine headlines and first parts of articles
        text_parts = []
        for _, article in representative_articles.iterrows():
            # Use headline + first 300 characters of article for context
            headline = article['Headline']
            article_snippet = article['Article'][:300] if pd.notna(article['Article']) else ""
            text_parts.append(f"{headline}. {article_snippet}")
        
        # Combine all text, but limit total length for model processing
        combined_text = " ".join(text_parts)
        
        # Truncate to approximately 1000 tokens (roughly 4000 characters for safety)
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000] + "..."
        
        return combined_text

    def generate_ai_narrative(self, cluster_text, cluster_name, cluster_size):
        """Generate narrative using DistilBART"""
        if not self.summarizer:
            return f"AI summarization not available. {cluster_name} contains {cluster_size} articles covering related themes."
        
        try:
            # Create a prompt that encourages narrative-style output
            prompt_text = f"Articles about {cluster_name}: {cluster_text}"
            
            # Generate summary using DistilBART
            summary_result = self.summarizer(
                prompt_text,
                max_length=180,
                min_length=60,
                do_sample=False,
                truncation=True
            )
            
            ai_summary = summary_result[0]['summary_text']
            
            # Enhance with narrative framing
            narrative = (f"The main narrative in the {cluster_name} cluster ({cluster_size} articles) "
                        f"reveals: {ai_summary}")
            
            return narrative
            
        except Exception as e:
            print(f"    Warning: AI narrative generation failed for cluster {cluster_name}: {e}")
            # Fallback to simple description
            return (f"The {cluster_name} cluster contains {cluster_size} articles covering "
                   f"related developments in this area.")

    def add_analytical_context(self, cluster_data, keywords):
        """Add analytical context about coverage patterns"""
        # Temporal analysis
        date_range = cluster_data['Date'].max() - cluster_data['Date'].min()
        days_span = date_range.days + 1
        
        # Coverage intensity
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
        
        # Build analytical context
        context_parts = [
            f"Coverage Analysis: This story received {intensity} over {days_span} days, "
            f"characterized by {author_pattern} and {depth}."
        ]
        
        if keywords:
            context_parts.append(f"Key themes identified: {', '.join(keywords[:5])}.")
        
        # Add significance assessment
        if len(cluster_data) > 15:
            significance = "This represents a major ongoing story with sustained media attention."
        elif len(cluster_data) > 8:
            significance = "This constitutes a significant news development."
        else:
            significance = "This reflects a notable news event."
        
        context_parts.append(significance)
        
        return " ".join(context_parts)

    def visualize_clusters(self):
        """Create enhanced visualization with context"""
        print("\nCreating enhanced cluster visualization...")
        
        # Generate cluster names first
        self.generate_cluster_names()
        
        # Reduce dimensions for visualization
        if UMAP_AVAILABLE:
            try:
                reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1)
                embeddings_2d = reducer.fit_transform(self.embeddings)
                method = "UMAP"
            except:
                pca = PCA(n_components=2, random_state=42)
                embeddings_2d = pca.fit_transform(self.embeddings)
                method = "PCA"
        else:
            pca = PCA(n_components=2, random_state=42)
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
                if len(cluster_articles) <= 20:
                    for i, (point, article) in enumerate(zip(cluster_points, cluster_articles.itertuples())):
                        headline = article.Headline[:50] + "..." if len(article.Headline) > 50 else article.Headline
                        ax.annotate(headline, 
                                  (point[0], point[1]),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.8,
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
        print("NARRATIVE INTELLIGENCE REPORT")
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
    
    def print_cluster_report(self):
        """Print a comprehensive report of all clusters"""
        print("\n" + "="*80)
        print("THEME CLUSTERING REPORT")
        print("="*80)
        
        print(f"\nDataset Summary:")
        print(f"  Total articles analyzed: {len(self.analysis_df)}")
        print(f"  Date range: {self.analysis_df['Date'].min().strftime('%Y-%m-%d')} to {self.analysis_df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"  Clusters found: {len(self.cluster_summaries)}")
        
        # Sort clusters by size
        sorted_clusters = sorted(self.cluster_summaries.items(), 
                               key=lambda x: x[1]['size'], reverse=True)
        
        for cluster_id, summary in sorted_clusters:
            print(f"\n" + "-"*60)
            print(f"CLUSTER {cluster_id} ({summary['size']} articles)")
            print("-"*60)
            
            print(f"Date Range: {summary['date_range']}")
            print(f"Average Word Count: {summary['avg_word_count']:.0f}")
            
            if summary['keywords']:
                print(f"Key Themes: {', '.join(summary['keywords'])}")
            
            if summary['top_authors']:
                print(f"Main Authors: {', '.join(summary['top_authors'][:3])}")
            
            print(f"\nRepresentative Headlines:")
            for i, headline in enumerate(summary['top_headlines'], 1):
                print(f"  {i}. {headline}")
            
        print("\n" + "="*80)
    
    def export_results(self, output_prefix="theme_clustering"):
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
        """Main method to run the complete clustering analysis"""
        print("Starting theme clustering analysis...")
        print("="*60)
        
        try:
            # Step 1: Load and clean data
            self.load_and_clean_data(topic_filter=topic_filter)
            
            # Step 2: Generate embeddings
            self.generate_embeddings(sample_size=sample_size)
            
            # Step 3: Cluster themes
            self.cluster_themes(min_cluster_size=min_cluster_size)
            
            # Step 4: Extract keywords
            self.extract_cluster_keywords()
            
            # Step 5: Create summaries
            self.create_cluster_summaries()
            
            # Step 6: Generate cluster names
            self.generate_cluster_names()
            
            # Step 7: Generate narratives
            self.generate_cluster_narratives()
            
            # Step 8: Visualize results
            self.visualize_clusters()
            
            # Step 9: Print narrative report
            self.print_narrative_report()
            
            # Step 9: Export results
            self.export_results()
            
            print("\nClustering analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Command line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple theme clustering for articles with AI narratives")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--sample-size", type=int, help="Maximum articles to analyze")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum cluster size")
    parser.add_argument("--topic", nargs="*", help="Optional keyword(s) to filter articles")
    
    args = parser.parse_args()
    
    print("Simple Theme Clustering Tool with AI Narratives")
    print("="*50)
    print(f"Dataset: {args.csv}")
    if args.topic:
        print(f"Topic filter: {', '.join(args.topic)}")
    print()
    
    try:
        clusterer = SimpleThemeClustering(args.csv)
        success = clusterer.run_clustering_analysis(
            sample_size=args.sample_size,
            min_cluster_size=args.min_cluster_size,
            topic_filter=args.topic
        )
        
        if success:
            print("\nAnalysis completed! Check the generated files:")
            print("- cluster_visualization.png")
            print("- detailed_cluster_view.png")
            print("- theme_clustering_articles.csv") 
            print("- theme_clustering_summaries.csv")
            print("- theme_clustering_narratives.csv")
        
    except Exception as e:
        print(f"Critical Error: {e}")
        raise


if __name__ == "__main__":
    main()