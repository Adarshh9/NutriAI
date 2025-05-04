import torch
import pickle
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Define the GNN model again
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

# Load saved assets
with open('/home/adarsh/Desktop/Personal Projects/NutriAI/core/gnn_inference/gnn_assets.pkl', 'rb') as f:
    assets = pickle.load(f)

node_ids = assets['node_ids']
kg_df = assets['kg_df']
x = assets['x']
edge_index = assets['edge_index']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare Data object
data = Data(x=x, edge_index=edge_index).to(device)

# Initialize and load model
model = GNNLinkPredictor(in_channels=x.size(1), hidden_channels=64).to(device)
model.load_state_dict(torch.load('/home/adarsh/Desktop/Personal Projects/NutriAI/core/gnn_inference/gnn_model.pt', map_location=device))
model.eval()


@torch.no_grad()
def recommend(disease_name, top_k=10, recommend_type="eat"):  # or "avoid"
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data.x = data.x.to(device)
    
    data.edge_index = data.edge_index.to(device)
    disease_idx = node_ids.get(disease_name)
    if disease_idx is None:
        return "Disease not found."

    food_nodes = kg_df['food'].unique()
    food_indices = [node_ids[f] for f in food_nodes]

    edge_pairs = torch.tensor([[disease_idx]*len(food_indices), food_indices], dtype=torch.long).to(device)
    scores = model(data.x, data.edge_index, edge_pairs).view(-1)

    sorted_indices = scores.argsort(descending=True)
    if recommend_type == "avoid":
        sorted_indices = scores.argsort()  # ascending = worst matches

    top_indices = sorted_indices[:top_k]
    return [food_nodes[i] for i in top_indices.cpu().numpy()]


class RecommendationExplainer:
    def __init__(self, model, data, node_ids, kg_df):
        self.model = model
        self.data = data
        self.node_ids = node_ids
        self.kg_df = kg_df
        
        # Reverse the node_ids mapping
        self.id_to_node = {idx: name for name, idx in node_ids.items()}
        
        # Extract node embeddings
        self.node_embeddings = self._extract_node_embeddings()
        
        # Create a networkx graph from the data
        self.G = to_networkx(data, to_undirected=False)
        
    def _extract_node_embeddings(self):
        """Extract the node embeddings from the trained GNN model"""
        self.model.eval()
        with torch.no_grad():
            # Pass through the first GCN layer
            h = self.model.conv1(self.data.x, self.data.edge_index)
            h = torch.nn.functional.relu(h)
            # Pass through the second GCN layer
            h = self.model.conv2(h, self.data.edge_index)
            return h.cpu().numpy()
    
    def get_similar_diseases(self, disease_name, top_k=5):
        """Find diseases with similar embeddings"""
        disease_idx = self.node_ids.get(disease_name)
        if disease_idx is None:
            return "Disease not found."
        
        # Get the embedding of the target disease
        disease_embedding = self.node_embeddings[disease_idx].reshape(1, -1)
        
        # Get all disease indices and names
        disease_nodes = self.kg_df['disease'].unique()
        disease_indices = [self.node_ids[d] for d in disease_nodes if d != disease_name]
        
        # Calculate similarity
        similarities = cosine_similarity(
            disease_embedding, 
            self.node_embeddings[disease_indices]
        )[0]
        
        # Get top k similar diseases
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(disease_nodes[i], similarities[i]) for i in top_indices]
    
    def get_common_treatments(self, disease_name, food_name):
        """Find diseases that are also treated by the recommended food"""
        food_idx = self.node_ids.get(food_name)
        
        # Get all edges where this food is the target
        related_diseases = []
        for i, row in self.kg_df.iterrows():
            if row['food'] == food_name and row['relationship'] == 'treats':
                related_diseases.append(row['disease'])
                
        return related_diseases
    
    def get_subgraph_influence(self, disease_name, food_name, n_hops=2):
        """Analyze the local subgraph to find influence paths"""
        disease_idx = self.node_ids.get(disease_name)
        food_idx = self.node_ids.get(food_name)
        
        if disease_idx is None or food_idx is None:
            return "Node not found."
            
        # Get n-hop neighborhood of disease
        paths = []
        for path in nx.all_simple_paths(self.G, source=disease_idx, target=food_idx, cutoff=n_hops):
            node_names = [self.id_to_node[idx] for idx in path]
            paths.append(node_names)
            
        return paths
    
    def explain_recommendation(self, disease_name, food_name):
        """Generate a comprehensive explanation for why a food is recommended"""
        # Get embedding similarity score
        disease_idx = self.node_ids.get(disease_name)
        food_idx = self.node_ids.get(food_name)
        
        if disease_idx is None or food_idx is None:
            return "Node not found."
        
        # Direct edge check
        direct_relation = None
        for i, row in self.kg_df.iterrows():
            if row['disease'] == disease_name and row['food'] == food_name:
                direct_relation = row['relationship']
                break
        
        # Get similar diseases that have a known relation with this food
        similar_diseases = self.get_similar_diseases(disease_name, top_k=3)
        
        # Check if any similar diseases have a direct relation with this food
        similar_disease_relations = []
        for sim_disease, sim_score in similar_diseases:
            for i, row in self.kg_df.iterrows():
                if row['disease'] == sim_disease and row['food'] == food_name:
                    similar_disease_relations.append((sim_disease, row['relationship'], sim_score))
                    break
        
        # Get other diseases treated by this food
        common_treatments = self.get_common_treatments(disease_name, food_name)
        
        # Craft the explanation
        explanation = {
            "food": food_name,
            "direct_relation": direct_relation,
            "similar_diseases": similar_disease_relations,
            "common_treatments": common_treatments
        }
        
        return explanation
    
    def get_explanation_text(self, disease_name, food_name):
        """Generate a human-readable explanation for the recommendation"""
        explanation = self.explain_recommendation(disease_name, food_name)
        
        if isinstance(explanation, str):  # Error message
            return explanation
            
        text = f"Explanation for recommending {explanation['food']} for {disease_name}:\n\n"
        
        # Direct relation
        if explanation['direct_relation']:
            rel = "beneficial for" if explanation['direct_relation'] == 'treats' else "should be avoided for"
            text += f"✓ There is a known direct relation: {explanation['food']} {rel} {disease_name}.\n\n"
        else:
            text += f"✓ The model has identified {explanation['food']} as potentially beneficial for {disease_name} based on learned patterns.\n\n"
        
        # Similar diseases
        if explanation['similar_diseases']:
            text += "✓ Similar conditions with known relationships to this food:\n"
            for disease, rel, score in explanation['similar_diseases']:
                rel_text = "benefits from" if rel == 'treats' else "should avoid"
                text += f"   - {disease} (similarity: {score:.2f}) {rel_text} {explanation['food']}.\n"
            text += "\n"
        
        # Common treatments
        if explanation['common_treatments']:
            text += f"✓ {explanation['food']} is also known to be beneficial for these conditions:\n"
            for disease in explanation['common_treatments']:
                if disease != disease_name:  # Skip the disease we're explaining
                    text += f"   - {disease}\n"
            text += "\n"
            
        text += f"The model learned these patterns from the knowledge graph and determined that {explanation['food']} would be a good recommendation for {disease_name}."
        
        return text

    def visualize_recommendation_network(self, disease_name, food_name, save_path=None):
        """Visualize the network around the disease and recommended food"""
        disease_idx = self.node_ids.get(disease_name)
        food_idx = self.node_ids.get(food_name)
        
        if disease_idx is None or food_idx is None:
            return "Node not found."
        
        # Create a subgraph for visualization
        # Include the direct neighbors of both the disease and food
        neighbors = set([disease_idx, food_idx])
        for node in [disease_idx, food_idx]:
            neighbors.update(self.G.neighbors(node))
        
        subgraph = self.G.subgraph(neighbors)
        pos = nx.spring_layout(subgraph)
        
        # Create node labels
        labels = {idx: self.id_to_node[idx] for idx in subgraph.nodes()}
        
        # Create node colors (red for disease, green for food, blue for others)
        node_colors = []
        for node in subgraph.nodes():
            if node == disease_idx:
                node_colors.append('red')
            elif node == food_idx:
                node_colors.append('green')
            elif self.id_to_node[node] in self.kg_df['disease'].values:
                node_colors.append('lightcoral')  # Light red for other diseases
            else:
                node_colors.append('lightgreen')  # Light green for other foods
        
        plt.figure(figsize=(10, 8))
        nx.draw_networkx(
            subgraph, pos, 
            labels=labels, 
            node_color=node_colors,
            node_size=500, 
            font_size=10, 
            font_weight='bold',
            edge_color='gray'
        )
        plt.title(f"Network visualization for {disease_name} and {food_name}")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            return f"Visualization saved to {save_path}"
        else:
            plt.show()
            return "Visualization displayed"

# Function to explain multiple recommendations
def explain_recommendations(disease_name, recommended_foods, model, data, node_ids, kg_df, top_n=5):
    """Explain a list of food recommendations for a disease"""
    explainer = RecommendationExplainer(model, data, node_ids, kg_df)
    
    print(f"Explanations for top {top_n} recommended foods for {disease_name}:")
    print("-" * 80)
    
    for i, food in enumerate(recommended_foods[:top_n]):
        print(f"\n{i+1}. {food}")
        explanation = explainer.get_explanation_text(disease_name, food)
        print(explanation)
        print("-" * 80)
    
    return explainer

# After you've run your model and obtained recommendations
# from RecommendationExplainer import RecommendationExplainer, explain_recommendations
disease = 'asthma'
recommended_foods = recommend(disease)

# Get explanations for the asthma recommendations
explainer = explain_recommendations(disease, recommended_foods, model, data, node_ids, kg_df, top_n=5)

# For a single recommendation
food = recommended_foods[0]
print(f"\nDetailed explanation for {food}:")
explanation = explainer.get_explanation_text(disease, food)
print(explanation)

# Visualize the network for the top recommendation
explainer.visualize_recommendation_network(disease, food, save_path=f"{disease}_recommendation_network.png")

# You can also create an interactive dashboard function
def create_recommendation_dashboard(disease_name):
    """Create an interactive dashboard for food recommendations with explanations"""
    foods = recommend(disease_name)
    print(f"Top recommendations for {disease_name}:")
    
    explainer = RecommendationExplainer(model, data, node_ids, kg_df)
    
    for i, food in enumerate(foods[:10]):
        print(f"{i+1}. {food}")
        
    selected = int(input("\nEnter number for detailed explanation (1-10): ")) - 1
    if 0 <= selected < len(foods):
        selected_food = foods[selected]
        print("\n" + "="*80)
        print(f"DETAILED EXPLANATION FOR {selected_food.upper()}")
        print("="*80)
        explanation = explainer.get_explanation_text(disease_name, selected_food)
        print(explanation)
        
        if input("\nVisualize network? (y/n): ").lower() == 'y':
            explainer.visualize_recommendation_network(disease_name, selected_food)
    
    return foods, explainer

# Example usage
foods, explainer = create_recommendation_dashboard(disease)
