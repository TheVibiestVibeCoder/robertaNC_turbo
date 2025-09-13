PLEASE DO THE FOLLOWING BELOW


# Narrative Intelligence Enhancement Specifications

## 🎯 Core Transformation: Cluster-Centric → Story-Centric

Replace technical clusters with narrative intelligence briefings. Each cluster becomes a named story with clear actors and implications.

## 🚀 Critical Code Additions

### 1. Narrative Intelligence Cards

```python
class NarrativeCard:
    def __init__(self):
        self.narrative_title = ""           # Auto-generated: "Fed Rate Hike Speculation" 
        self.executive_summary = ""         # 2-sentence story overview
        self.key_actors = []               # Top 3-5 journalists with influence scores
        self.critical_facts = []           # 3-5 essential bullet points
        self.sentiment_trajectory = {}     # Time-based sentiment evolution
        self.market_impact = ""            # Business implications
        self.lifecycle_stage = ""          # Emerging/Peak/Declining

def generate_narrative_titles(self):
    """Create descriptive titles from keywords + entities + context"""
    # Use: top 3 keywords + dominant entity + temporal context
    # Output: "Silicon Valley Bank Collapse Coverage", "Fed Hawkish Policy Shift"

def create_executive_summaries(self):
    """Generate 2-sentence story summaries"""
    # Template: What happened + Why it matters
    # Use extractive summarization from top articles
```

### 2. Actor-Narrative Correlation Engine

```python
def build_actor_narrative_matrix(self):
    """Create journalist -> narrative influence mapping"""
    influence_matrix = {}
    for cluster_id in clusters:
        for journalist in cluster_journalists:
            influence_matrix[journalist][cluster_id] = {
                'article_count': count,
                'sentiment_influence': sentiment_impact,
                'temporal_leadership': early_reporting_score,
                'headline_prominence': headline_feature_score
            }
    return influence_matrix

def calculate_influence_scores(self):
    """Multi-factor influence calculation"""
    # Factors: article count (30%), early reporting (25%), 
    # sentiment impact (25%), headline prominence (20%)
    
def identify_narrative_drivers(self):
    """Find who initiates vs amplifies narratives"""
    # Driver: reports early, sets tone
    # Amplifier: follows, expands coverage
```

### 3. Enhanced Dashboard Structure

```python
def create_narrative_dashboard(self):
    """Replace technical dashboard with story-focused intelligence"""
    
    # Layout: 2x3 grid
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Active Narratives Map', 'Actor Influence Matrix', 'Sentiment Evolution',
            'Narrative Network', 'Timeline Flow', 'Intelligence Summary'
        ]
    )
    
    # 1. Active Narratives Map (replace cluster scatter)
    self.add_narrative_bubbles(fig, row=1, col=1)
    
    # 2. Actor-Narrative Heatmap  
    self.add_influence_heatmap(fig, row=1, col=2)
    
    # 3. Sentiment Evolution Lines
    self.add_sentiment_trajectories(fig, row=1, col=3)
    
    # 4. Interactive Network (nodes=narratives, edges=shared actors)
    self.add_narrative_network(fig, row=2, col=1)
    
    # 5. Temporal Flow
    self.add_narrative_timeline(fig, row=2, col=2)
    
    # 6. Intelligence Cards Table
    self.add_intelligence_summary(fig, row=2, col=3)

def add_narrative_bubbles(self, fig, row, col):
    """Narrative overview with meaningful names"""
    for narrative_id, card in self.narrative_cards.items():
        fig.add_trace(go.Scatter(
            x=[card.importance_score],
            y=[card.sentiment_score], 
            mode='markers+text',
            marker_size=card.article_count*2,
            text=card.narrative_title,
            name=card.narrative_title,
            hovertemplate=f"<b>{card.narrative_title}</b><br>" +
                         f"{card.executive_summary}<br>" +
                         f"Articles: {card.article_count}<br>" +
                         f"Key Actors: {', '.join(card.key_actors[:3])}"
        ), row=row, col=col)
```

### 4. Intelligence Reporting

```python
def generate_situation_report(self):
    """Create executive briefing format"""
    
    print("🔍 NARRATIVE INTELLIGENCE BRIEFING")
    print("="*60)
    
    # Sort by importance
    top_narratives = sorted(self.narrative_cards.items(), 
                           key=lambda x: x[1].importance_score, reverse=True)
    
    for i, (nid, card) in enumerate(top_narratives[:5], 1):
        print(f"\n{i}. {card.narrative_title}")
        print(f"   📊 {card.article_count} articles | {card.lifecycle_stage}")
        print(f"   👥 Key Actors: {', '.join(card.key_actors[:3])}")  
        print(f"   📝 {card.executive_summary}")
        print(f"   💡 Impact: {card.market_impact}")
        if card.critical_facts:
            print(f"   🔑 Key Facts:")
            for fact in card.critical_facts[:3]:
                print(f"      • {fact}")

def export_intelligence_report(self):
    """Export actionable intelligence document"""
    # Generate HTML report with:
    # - Executive dashboard screenshot
    # - Top 10 narrative briefings  
    # - Actor influence rankings
    # - Strategic recommendations
```

### 5. Required Method Modifications

**Modify existing methods:**

```python
# IN: generate_enhanced_narrative_summaries()
# ADD: Call generate_narrative_titles() and create_executive_summaries()

# IN: create_simplified_dashboard() 
# REPLACE: With create_narrative_dashboard()

# IN: create_comprehensive_report()
# REPLACE: With generate_situation_report()

# ADD NEW: Main analysis method integration
def run_enhanced_analysis(self):
    # ... existing code ...
    
    # ADD AFTER phase 6:
    self.build_actor_narrative_matrix()
    self.calculate_influence_scores() 
    self.create_narrative_cards()
    
    # MODIFY phase 7:
    self.create_narrative_dashboard()  # Instead of create_simplified_dashboard()
    
    # MODIFY phase 8: 
    self.generate_situation_report()   # Instead of create_comprehensive_report()
```

### 6. Data Structure Updates

```python
# ADD to __init__:
self.narrative_cards = {}
self.actor_influence_matrix = None
self.narrative_network = None

# MODIFY analysis_df to include:
self.analysis_df['Narrative_Title'] = narrative_titles
self.analysis_df['Actor_Influence'] = influence_scores
self.analysis_df['Narrative_Stage'] = lifecycle_stages
```

## 🎯 Implementation Checklist

### Phase 1 - Core Transformation:
- [ ] Create NarrativeCard class
- [ ] Implement generate_narrative_titles()  
- [ ] Build actor_influence_matrix
- [ ] Replace dashboard with narrative-focused version
- [ ] Update reporting to situation briefing format

### Phase 2 - Advanced Features:
- [ ] Add interactive narrative network
- [ ] Implement sentiment trajectory tracking
- [ ] Create HTML intelligence export
- [ ] Add narrative lifecycle detection

## 🔧 Key Integration Points

1. **Keep existing data ingestion** - no changes to load_and_preprocess_data()
2. **Keep all embedding/clustering logic** - enhance presentation layer only
3. **Add narrative layer** - between clustering and dashboard creation
4. **Replace all cluster references** in user-facing outputs with narrative titles
5. **Maintain all export functionality** - enhance with narrative context

## 📊 Expected Output Changes

**Before**: "Cluster 7 has 23 articles about financial topics"
**After**: "Fed Rate Hike Speculation - 23 articles over 4 days show growing hawkish sentiment. Led by Sarah Johnson (Bloomberg) and Mike Chen (WSJ). Market expecting 0.75% increase."

This transforms your technical clustering tool into an actionable narrative intelligence platform.