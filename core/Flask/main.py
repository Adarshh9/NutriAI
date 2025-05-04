from flask import Flask, render_template, request, jsonify, send_from_directory
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
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
# from flask.json import JSONEncoder

# class CustomJSONEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, (np.integer, np.floating, np.bool_)):
#             return int(obj) if isinstance(obj, np.integer) else float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super().default(obj)


app = Flask(__name__)
@app.after_request
def add_header(response):
    response.headers['ngrok-skip-browser-warning'] = 'any-value'
    return response
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.json_encoder = CustomJSONEncoder

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Helper Functions ---
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

# Load models at startup
models = load_models_and_data()

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

# Initialize cache
cache_manager = PersistentRLCache(db_path=".nutriai_cache.json", max_size=35)

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

def get_food_explanation(food, disease):
    cache_key = (food.lower(), disease.lower())
    
    # Check cache first
    cached = cache_manager.get_from_cache(cache_key)
    if cached:
        return cached
    
    # Call API if not cached
    prompt = f"""
    Explain in one concise sentence (max 40 words) why {food} might be beneficial or harmful for {disease}.
    Use medical/scientific reasoning. Only state facts, no disclaimers.
    """
    
    start_time = time.time()
    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50,
        )
        explanation = response.choices[0].message.content
        response_time = time.time() - start_time
        
        # Add to persistent cache
        cache_manager.add_to_cache(cache_key, explanation, response_time)
        
        return explanation
    except Exception as e:
        return f"Explanation unavailable: {str(e)}"

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
                .apply(lambda x: x.astype(str) if x.dtype == 'object' else x)
                .to_dict(orient='records')
            )
            # Convert numpy types to native Python types
            for nut in nutrients_list:
                if isinstance(nut['value'], (np.integer, np.floating)):
                    nut['value'] = float(nut['value'])

        if mapped_extra_food:
            extra_row = df_nutri_value[df_nutri_value['Main food description'] == mapped_extra_food]
            if not extra_row.empty:
                nutrient_cols = ['Energy (kcal)', 'Protein (g)', 'Carbohydrate (g)', 
                               'Sugars, total\n(g)', 'Fiber, total dietary (g)', 'Total Fat (g)']
                for col in nutrient_cols:
                    if col in extra_row.columns:
                        val = extra_row.iloc[0][col]
                        # Convert numpy types to native Python types
                        if isinstance(val, (np.integer, np.floating)):
                            val = float(val)
                        extra_nutrients_list.append({
                            'nutrient': col.strip(), 
                            'value': val
                        })

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
    
    # Save the figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Encode the image as base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Get list of diseases for dropdown
    diseases = sorted(models['gnn']['assets']['kg_df']['disease'].unique())
    
    if request.method == 'POST':
        disease_query = request.form.get('disease')
        return recommend(disease_query)
    
    return render_template('index.html', diseases=diseases)

@app.route('/recommend', methods=['POST'])
def recommend():
    disease_query = request.form.get('disease')
    
    if not disease_query:
        return redirect('/')
    
    # Get recommendations
    kg_recs = kg_recommend(
        disease_query,
        models['kg']['df'],
        models['kg']['nutrient_data'],
        models['kg']['df_nutri_value'],
        models['kg']['data']
    )
    
    gnn_recs = gnn_recommend(
        disease_query.lower(),
        models['gnn']['model'],
        models['gnn']['data'],
        models['gnn']['assets']['node_ids'],
        models['gnn']['assets']['kg_df'],
        top_k=10
    )
    
    # Get text evidence for all foods
    text_evidence = {}
    if kg_recs:
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
    
    # Get explanations for GNN recommendations
    gnn_explanations = {}
    for food in gnn_recs[:5]:
        gnn_explanations[food] = get_food_explanation(food, disease_query)
    
    return render_template('results.html', 
                         disease=disease_query,
                         kg_recs=kg_recs,
                         gnn_recs=gnn_recs[:5],
                         gnn_explanations=gnn_explanations,
                         text_evidence=text_evidence)

@app.route('/get_explanation', methods=['POST'])
def get_explanation():
    food = request.form.get('food')
    disease = request.form.get('disease')
    explanation = get_food_explanation(food, disease)
    print(explanation)
    return jsonify({'explanation': explanation})

@app.route('/visualize', methods=['POST'])
def visualize():
    disease = request.form.get('disease')
    food = request.form.get('food')
    
    class Explainer:
        def __init__(self, node_ids):
            self.id_to_node = {v: k for k, v in node_ids.items()}
    
    image_base64 = visualize_network(
        disease.lower(),
        food.lower(),
        Explainer(models['gnn']['assets']['node_ids']),
        models['gnn']['data'],
        models['gnn']['assets']['node_ids'],
        models['gnn']['assets']['kg_df']
    )
    
    if image_base64:
        return jsonify({'image': image_base64})
    else:
        return jsonify({'error': 'Could not generate visualization'}), 400

@app.route('/cache_management', methods=['GET', 'POST'])
def cache_management():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'clear':
            cache_manager.cache.clear()
            cache_manager._save_db()
        elif action == 'search':
            search_term = request.form.get('search_term')
            matches = [
                (k, v) for k, v in cache_manager.cache.items()
                if search_term.lower() in str(k).lower()
            ]
            return render_template('cache.html', 
                                cache=cache_manager.cache,
                                matches=matches,
                                search_term=search_term,
                                cache_manager=cache_manager)  # Add this
    
    return render_template('cache.html', 
                         cache=cache_manager.cache,
                         matches=None,
                         search_term=None,
                         cache_manager=cache_manager)  # Add this

# --- Template Filters ---
@app.template_filter('format_time')
def format_time(timestamp):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))

# --- Static Files ---
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)