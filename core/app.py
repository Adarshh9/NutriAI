import streamlit as st
import pandas as pd
import joblib
import torch
import pickle
import faiss
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from collections import OrderedDict
from groq import Groq
from configs import GROQ_API_KEY
import time
import os
import json

# Set page config
st.set_page_config(
    page_title="NutriAI - Food-Disease Recommender",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_resource
def load_models_and_data():
    """Load all models and data at startup"""
    # --- KG Model ---
    kg_data = joblib.load('/home/adarsh/Desktop/Personal Projects/NutriAI/core/kg_inference/food_matching_data.pkl')
    kg_df = pd.read_csv("/home/adarsh/Desktop/Personal Projects/NutriAI/core/kg_inference/kg_data.csv")
    nutrient_data = pd.read_csv("/home/adarsh/Desktop/Personal Projects/NutriAI/core/kg_inference/nutrient_data.csv")
    df_nutri_value = pd.read_csv("/home/adarsh/Desktop/Personal Projects/NutriAI/core/kg_inference/df_extra.csv")
    
    # --- GNN Model ---
    with open('/home/adarsh/Desktop/Personal Projects/NutriAI/core/gnn_inference/gnn_assets.pkl', 'rb') as f:
        gnn_assets = pickle.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_data = Data(x=gnn_assets['x'], edge_index=gnn_assets['edge_index']).to(device)
    
    class GNNLinkPredictor(nn.Module):
        def __init__(self, in_channels, hidden_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.link_predictor = nn.Sequential(
                nn.Linear(2 * hidden_channels, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x, edge_index, edge_pairs):
            h = self.conv1(x, edge_index)
            h = F.relu(h)
            h = self.conv2(h, edge_index)
            src, dst = edge_pairs
            h_src, h_dst = h[src], h[dst]
            edge_feat = torch.cat([h_src, h_dst], dim=1)
            return torch.sigmoid(self.link_predictor(edge_feat))
    
    gnn_model = GNNLinkPredictor(
        in_channels=gnn_assets['x'].size(1), 
        hidden_channels=64
    ).to(device)
    gnn_model.load_state_dict(torch.load(
        '/home/adarsh/Desktop/Personal Projects/NutriAI/core/gnn_inference/gnn_model.pt',
        map_location=device
    ))
    gnn_model.eval()
    
    # --- Text Retriever ---
    text_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    text_index = faiss.read_index("/home/adarsh/Desktop/Personal Projects/NutriAI/core/text_Retrieval/food_disease_faiss_index.bin")
    
    with open("/home/adarsh/Desktop/Personal Projects/NutriAI/core/text_Retrieval/metadata.pkl", "rb") as f:
        text_metadata = pickle.load(f)
    
    with open("/home/adarsh/Desktop/Personal Projects/NutriAI/core/text_Retrieval/texts.pkl", "rb") as f:
        text_texts = pickle.load(f)
    
    return {
        'kg': {
            'data': kg_data,
            'df': kg_df,
            'nutrient_data': nutrient_data,
            'df_nutri_value': df_nutri_value
        },
        'gnn': {
            'model': gnn_model,
            'data': gnn_data,
            'assets': gnn_assets
        },
        'text': {
            'model': text_model,
            'index': text_index,
            'metadata': text_metadata,
            'texts': text_texts
        }
    }

# --- Persistent RL Cache Manager ---
class PersistentRLCache:
    def __init__(self, db_path="cache_db.json", max_size=100):
        self.db_path = db_path
        self.max_size = max_size
        self.cache = OrderedDict()
        self.q_values = {}
        self._load_db()
        
        # RL parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9   # Discount factor
        self.epsilon = 0.2 # Exploration rate
    
    def _load_db(self):
        """Load cache from JSON file"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                self.cache = OrderedDict(data.get('cache', {}))
                self.q_values = data.get('q_values', {})
        else:
            self.cache = OrderedDict()
            self.q_values = {}
    
    def _save_db(self):
        """Save cache to JSON file"""
        with open(self.db_path, 'w') as f:
            json.dump({
                'cache': dict(self.cache),
                'q_values': self.q_values
            }, f, indent=2)
    
    def should_cache(self, key, response_time):
        """RL decision to cache or not"""
        state = (min(int(response_time*10), 5), len(self.cache)//10)
        
        # Initialize Q-values if new state
        if str(state) not in self.q_values:  # JSON keys must be strings
            self.q_values[str(state)] = {'cache': 1, 'no_cache': 0}
        
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            decision = np.random.choice(['cache', 'no_cache'])
        else:
            decision = max(self.q_values[str(state)], 
                          key=self.q_values[str(state)].get)
        
        # Reward function
        reward = np.log(response_time + 1)  # Higher reward for slower responses
        
        # Q-learning update
        best_next = max(self.q_values[str(state)].values())
        self.q_values[str(state)][decision] = (1 - self.alpha) * self.q_values[str(state)][decision] + \
                                            self.alpha * (reward + self.gamma * best_next)
        
        self._save_db()  # Persist learning
        return decision == 'cache'
    
    def add_to_cache(self, key, value, response_time):
        key_str = str(key)  # JSON keys must be strings
        if self.should_cache(key_str, response_time):
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key_str] = {
                'value': value,
                'metadata': {
                    'last_used': time.time(),
                    'usage_count': 1,
                    'response_time': response_time,
                    'created_at': time.time()
                }
            }
            self._save_db()
    
    def get_from_cache(self, key):
        key_str = str(key)
        if key_str in self.cache:
            entry = self.cache[key_str]
            entry['metadata']['last_used'] = time.time()
            entry['metadata']['usage_count'] += 1
            self.cache.move_to_end(key_str)
            self._save_db()
            return entry['value']
        return None

# --- Groq Integration ---
@st.cache_resource
def get_cache_manager():
    return PersistentRLCache(
        db_path=".nutriai_cache.json",
        max_size=200
    )

@st.cache_resource
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

def get_food_explanation(food, disease):
    cache = get_cache_manager()
    cache_key = (food.lower(), disease.lower())
    
    # Check cache first
    cached = cache.get_from_cache(cache_key)
    if cached:
        st.toast("⚡ Using cached explanation", icon="💾")
        return cached
    
    # Call API if not cached
    prompt = f"""
    Explain in one concise sentence (max 40 words) why {food} might be beneficial or harmful for {disease}.
    Use medical/scientific reasoning. Only state facts, no disclaimers.
    """
    
    start_time = time.time()
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50,
        )
        explanation = response.choices[0].message.content
        response_time = time.time() - start_time
        
        # Add to persistent cache
        cache.add_to_cache(cache_key, explanation, response_time)
        
        return explanation
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return "Explanation unavailable"

def clean_text(text):
    return text.strip().lower()

# --- KG Recommendation Function ---
def kg_recommend(disease_query, kg_df, nutrient_data, df_nutri_value, food_maps):
    disease_query = clean_text(disease_query)
    treat_df = kg_df[(kg_df['relationship'] == 'treats') & (kg_df['disease'].str.lower() == disease_query)]
    
    if treat_df.empty:
        return None
    
    output = []
    for _, row in treat_df.iterrows():
        food = row['food']
        mapped_nutrient_food = food_maps['food_nutrient_map'].get(food)
        mapped_extra_food = food_maps['food_extra_map'].get(food)

        nutrients_list = []
        extra_nutrients_list = []

        if mapped_nutrient_food:
            nutrients = nutrient_data[nutrient_data['food'] == mapped_nutrient_food]
            nutrients_list = (
                nutrients[['nutrient', 'value']]
                .drop_duplicates(subset='nutrient', keep='first')
                .sort_values(by='value', ascending=False)
                .head(5)
                .to_dict(orient='records')
            )

        if mapped_extra_food:
            extra_row = df_nutri_value[df_nutri_value['Main food description'] == mapped_extra_food]
            if not extra_row.empty:
                nutrient_cols = ['Energy (kcal)', 'Protein (g)', 'Carbohydrate (g)', 
                               'Sugars, total\n(g)', 'Fiber, total dietary (g)', 'Total Fat (g)']
                for col in nutrient_cols:
                    if col in extra_row.columns:
                        val = extra_row.iloc[0][col]
                        extra_nutrients_list.append({'nutrient': col.strip(), 'value': val})

        output.append({
            "food": food.title(),
            "evidence": row['evidence'],
            "nutrients": nutrients_list,
            "extra_nutrients": extra_nutrients_list
        })
    
    return output

# --- GNN Recommendation Function ---
@torch.no_grad()
def gnn_recommend(disease_name, model, data, node_ids, kg_df, top_k=10):
    disease_idx = node_ids.get(disease_name)
    if disease_idx is None:
        return []
    
    food_nodes = kg_df['food'].unique()
    food_indices = [node_ids[f] for f in food_nodes]
    edge_pairs = torch.tensor([[disease_idx]*len(food_indices), food_indices], 
                             dtype=torch.long).to(data.x.device)
    
    scores = model(data.x, data.edge_index, edge_pairs).view(-1)
    top_indices = scores.argsort(descending=True)[:top_k]
    
    return [food_nodes[i] for i in top_indices.cpu().numpy()]

# --- Streamlit UI Integration ---
def display_gnn_recommendations(disease, gnn_recs):
    st.subheader("AI-Predicted Suggestions")
    
    for food in gnn_recs[:5]:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {food}")
                
            with col2:
                if st.button(f"Explain", key=f"explain_{food}"):
                    with st.spinner("Generating explanation..."):
                        explanation = get_food_explanation(food, disease)
                        st.session_state[f"explanation_{food}"] = explanation
            
            if f"explanation_{food}" in st.session_state:
                st.info(f"**Scientific reason**: {st.session_state[f'explanation_{food}']}")
                
            st.divider()

# --- Cache Management UI ---
def show_cache_controls():
    st.sidebar.subheader("Cache Management")
    cache = get_cache_manager()
    
    # Cache stats
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Entries", len(cache.cache))
    col2.metric("Max Size", cache.max_size)
    
    # Search cache
    search_term = st.sidebar.text_input("Search cache:")
    if search_term:
        matches = [
            (k, v) for k, v in cache.cache.items()
            if search_term.lower() in str(k).lower()
        ]
        st.sidebar.write(f"Found {len(matches)} matches")
    
    # Manual cache controls
    if st.sidebar.button("Clear Cache"):
        cache.cache.clear()
        cache._save_db()
        st.sidebar.success("Cache cleared!")
    
    if st.sidebar.checkbox("Show raw cache"):
        st.sidebar.json(dict(list(cache.cache.items())[:3]))

# --- Enhanced Recommendation Display ---
def display_food_recommendation(food, disease):
    with st.container(border=True):
        cols = st.columns([4, 1])
        cols[0].subheader(food)
        
        # Explanation button with cached indicator
        with cols[1]:
            cache = get_cache_manager()
            is_cached = str((food.lower(), disease.lower())) in cache.cache
            btn_label = "🔍 Explain" + (" (cached)" if is_cached else "")
            
            if st.button(btn_label, key=f"explain_{food}_{disease}"):
                with st.spinner("Getting explanation..."):
                    explanation = get_food_explanation(food, disease)
                    st.session_state[f"ex_{food}_{disease}"] = explanation
        
        # Display explanation if available
        if f"ex_{food}_{disease}" in st.session_state:
            st.info(st.session_state[f"ex_{food}_{disease}"])
            
            # Show cache metadata if available
            cache_key = str((food.lower(), disease.lower()))
            if cache_key in cache.cache:
                meta = cache.cache[cache_key]['metadata']
                st.caption(f"Last used: {time.ctime(meta['last_used'])} | "
                          f"Used {meta['usage_count']} times | "
                          f"Response time: {meta['response_time']:.2f}s")
                
# --- Text Search Function ---
def text_search(query, model, index, metadata, texts, top_k=5):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        results.append({
            "text": texts[idx],
            "source": metadata[idx]["source"],
            "id": metadata[idx]["id"],
            "score": float(1 / (1 + distance))
        })
    
    return results

# --- Visualization Function ---
def visualize_network(disease_name, food_name, model, data, node_ids, kg_df):
    disease_idx = node_ids.get(disease_name)
    food_idx = node_ids.get(food_name)
    
    if disease_idx is None or food_idx is None:
        return None
    
    G = to_networkx(data, to_undirected=False)
    neighbors = set([disease_idx, food_idx])
    for node in [disease_idx, food_idx]:
        neighbors.update(G.neighbors(node))
    
    subgraph = G.subgraph(neighbors)
    pos = nx.spring_layout(subgraph)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    node_colors = []
    for node in subgraph.nodes():
        if node == disease_idx:
            node_colors.append('red')
        elif node == food_idx:
            node_colors.append('green')
        elif model.id_to_node[node] in kg_df['disease'].values:
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgreen')
    
    nx.draw_networkx(
        subgraph, pos, 
        labels={idx: model.id_to_node[idx] for idx in subgraph.nodes()},
        node_color=node_colors,
        node_size=800,
        font_size=12,
        font_weight='bold',
        edge_color='gray',
        ax=ax
    )
    ax.set_title(f"Knowledge Graph Connections: {disease_name} → {food_name}", fontsize=14)
    ax.axis('off')
    
    return fig

# --- Main App ---
def main():
    # Load all models and data
    with st.spinner("Loading models and data..."):
        models = load_models_and_data()
    
    # Custom CSS
    st.markdown("""
    <style>
    .recommendation-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
    }
    .kg-card {
        border-left: 5px solid #4CAF50;
    }
    .gnn-card {
        border-left: 5px solid #2196F3;
    }
    .nutrient-badge {
        display: inline-block;
        padding: 0.25em 0.4em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        background-color: #f8f9fa;
        color: #212529;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://via.placeholder.com/150x50?text=NutriAI", use_column_width=True)
    st.sidebar.title("Navigation")
    disease_query = st.sidebar.selectbox(
        "Select a disease:", 
        sorted(models['gnn']['assets']['kg_df']['disease'].unique()),
        index=0
    )
    show_cache_controls()
    # Main content
    st.title("🍏 NutriAI: Food-Disease Recommendation System")
    st.markdown("""
    This system recommends foods based on:
    - **Knowledge Graph**: Structured medical knowledge
    - **Graph Neural Network**: AI-powered pattern recognition
    - **Medical Literature**: Evidence from scientific studies
    """)
    
    # Get recommendations
    with st.spinner(f"Finding best foods for {disease_query}..."):
        # KG recommendations
        kg_recs = kg_recommend(
            disease_query,
            models['kg']['df'],
            models['kg']['nutrient_data'],
            models['kg']['df_nutri_value'],
            models['kg']['data']
        )
        
        # GNN recommendations
        gnn_recs = gnn_recommend(
            disease_query.lower(),
            models['gnn']['model'],
            models['gnn']['data'],
            models['gnn']['assets']['node_ids'],
            models['gnn']['assets']['kg_df'],
            top_k=10
        )
        
        # Text evidence (for all foods)
        if kg_recs:
            text_evidence = {}
            for food in [r['food'] for r in kg_recs] + gnn_recs:
                results = text_search(
                    f"{food} AND {disease_query}",
                    models['text']['model'],
                    models['text']['index'],
                    models['text']['metadata'],
                    models['text']['texts'],
                    top_k=2
                )
                if results:
                    text_evidence[food] = results
    
    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["Recommendations", "Scientific Evidence", "Nutrient Analysis"])
    
    with tab1:
        st.header("🌟 Top Recommendations")
        
        # KG-validated recommendations
        if kg_recs:
            st.subheader("Validated by Medical Knowledge")
            for rec in kg_recs[:5]:
                with st.container():
                    st.markdown(
                        f"""
                        <div class="recommendation-card kg-card">
                            <h3>✅ {rec['food']}</h3>
                            <p><i>{rec['evidence']}</i></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Nutrients
                    if rec['nutrients'] or rec['extra_nutrients']:
                        with st.expander("View nutrients"):
                            cols = st.columns(2)
                            with cols[0]:
                                if rec['nutrients']:
                                    st.markdown("**Key Nutrients (per 100g)**")
                                    for nut in rec['nutrients']:
                                        st.markdown(
                                            f'<span class="nutrient-badge">{nut["nutrient"].title()}: {nut["value"]}</span>',
                                            unsafe_allow_html=True
                                        )
                            with cols[1]:
                                if rec['extra_nutrients']:
                                    st.markdown("**Additional Nutrients**")
                                    for nut in rec['extra_nutrients']:
                                        st.markdown(
                                            f'<span class="nutrient-badge">{nut["nutrient"]}: {nut["value"]}</span>',
                                            unsafe_allow_html=True
                                        )
        
        # GNN predictions
        if gnn_recs:
            for food in gnn_recs[:5]:
                display_food_recommendation(food, disease_query)
    
    with tab2:
        st.header("📚 Scientific Evidence")
        
        if not text_evidence:
            st.warning("No literature evidence found for these recommendations")
        else:
            for food, evidence in text_evidence.items():
                with st.expander(f"Evidence for {food}"):
                    for i, doc in enumerate(evidence, 1):
                        st.markdown(f"**Study {i}** (Relevance: {doc['score']:.0%})")
                        st.write(doc['text'])
                        st.caption(f"Source: {doc['source']} | ID: {doc['id']}")
                        st.divider()
    
    with tab3:
        st.header("🧪 Nutrient Explorer")
        
        if kg_recs:
            selected_food = st.selectbox(
                "Select food for detailed analysis:",
                [rec['food'] for rec in kg_recs]
            )
            
            for rec in kg_recs:
                if rec['food'] == selected_food:
                    # Nutrient visualization
                    if rec['nutrients']:
                        nutrients_df = pd.DataFrame(rec['nutrients'])
                        st.bar_chart(
                            nutrients_df.set_index('nutrient')['value'],
                            use_container_width=True
                        )
                    
                    # Detailed table
                    if rec['extra_nutrients']:
                        st.markdown("**Complete Nutritional Profile**")
                        extra_df = pd.DataFrame(rec['extra_nutrients'])
                        st.dataframe(
                            extra_df,
                            column_config={
                                "nutrient": "Nutrient",
                                "value": st.column_config.NumberColumn("Amount")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
        
        # Network visualization
        if kg_recs and st.checkbox("Show knowledge graph connections"):
            selected_food = st.selectbox(
                "Select food to visualize:",
                [rec['food'] for rec in kg_recs]
            )
            
            # Create visualization
            class Explainer:
                def __init__(self, node_ids):
                    self.id_to_node = {v: k for k, v in node_ids.items()}
            
            fig = visualize_network(
                disease_query.lower(),
                selected_food.lower(),
                Explainer(models['gnn']['assets']['node_ids']),
                models['gnn']['data'],
                models['gnn']['assets']['node_ids'],
                models['gnn']['assets']['kg_df']
            )
            
            if fig:
                st.pyplot(fig)
                st.caption("Graph explanation: Red = Disease, Green = Recommended Food, Pink = Other Diseases, Light Green = Other Foods")
            else:
                st.warning("Could not generate visualization for this combination")

    # Footer
    st.divider()
    st.caption("""
    **Disclaimer**: These recommendations are generated by AI models. 
    Consult a healthcare professional before making dietary changes for medical conditions.
    """)

if __name__ == "__main__":
    main()