#!/usr/bin/env python3
"""
SIMPLE THEME CLUSTERING SCRIPT
Focuses only on data ingestion and clustering articles by thematic similarity
"""

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
    - Clear visualization of results
    """
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.embeddings = None
        self.clusters = None
        self.cluster_summaries = {}
        
        # Determine compute device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Simple Theme Clustering Tool")
        print("=" * 50)
        self.setup_models()
        
    def setup_models(self):
        """Setup embedding model for semantic similarity"""
        try:
            print("Loading Sentence-BERT model...")
            self.sentence_model = SentenceTransformer(
                'all-MiniLM-L6-v2', device=self.device
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
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
        """Extract representative keywords for each cluster"""
        print("\nExtracting cluster keywords...")
        
        # Setup TF-IDF for keyword extraction
        tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        self.cluster_keywords = {}
        
        for cluster_id in np.unique(self.clusters[self.clusters >= 0]):
            cluster_data = self.analysis_df[self.clusters == cluster_id]
            cluster_texts = cluster_data['Full_Text'].tolist()
            
            if len(cluster_texts) == 0:
                continue
            
            # Fit TF-IDF on cluster texts
            try:
                tfidf_matrix = tfidf.fit_transform(cluster_texts)
                feature_names = tfidf.get_feature_names_out()
                
                # Get average TF-IDF scores
                avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
                
                # Get top keywords
                top_indices = avg_scores.argsort()[-10:][::-1]
                top_keywords = [feature_names[i] for i in top_indices]
                
                self.cluster_keywords[cluster_id] = top_keywords
                
            except Exception as e:
                print(f"Error extracting keywords for cluster {cluster_id}: {e}")
                self.cluster_keywords[cluster_id] = []
        
        return self.cluster_keywords
    
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
            
            # Step 6: Visualize results
            self.visualize_clusters()
            
            # Step 7: Print report
            self.print_cluster_report()
            
            # Step 8: Export results
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
    
    parser = argparse.ArgumentParser(description="Simple theme clustering for articles")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--sample-size", type=int, help="Maximum articles to analyze")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum cluster size")
    parser.add_argument("--topic", nargs="*", help="Optional keyword(s) to filter articles")
    
    args = parser.parse_args()
    
    print("Simple Theme Clustering Tool")
    print("="*40)
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
            print("- theme_clustering_articles.csv") 
            print("- theme_clustering_summaries.csv")
        
    except Exception as e:
        print(f"Critical Error: {e}")
        raise


if __name__ == "__main__":
    main()