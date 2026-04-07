import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Define and merge the uploaded CSV files
files = [
    'ollama_semantic_network.csv', 
    'ollama_semantic_network_AI_marketing_workflow.csv', 
    'ollama_semantic_network_AI marketing automation.csv',
    'ollama_semantic_network_generative_AI_marketing.csv'
]

dfs = [pd.read_csv(f) for f in files]

# Concatenate and remove any duplicate relationships extracted from overlapping papers
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.drop_duplicates(inplace=True)
merged_df.to_csv('merged_semantic_network.csv', index=False)

# 2. Construct the Directed Graph
# We use DiGraph to preserve the directionality of the (source -> relationship -> target) structure
G = nx.from_pandas_edgelist(
    merged_df, 
    source='source', 
    target='target', 
    edge_attr='relationship', 
    create_using=nx.DiGraph()
)

# 3. Calculate Important Graph Properties
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)
degree_centrality = nx.degree_centrality(G)

print(f"Nodes: {num_nodes} | Edges: {num_edges} | Density: {density:.4f}")

# 4. Visualise the Distributions
sns.set_theme(style="whitegrid")

# Relationship Distribution
plt.figure(figsize=(10, 6))
rel_counts = merged_df['relationship'].value_counts().head(15)
sns.barplot(x=rel_counts.values, y=rel_counts.index, palette='viridis')
plt.title('Top 15 Most Frequent Relationships')
plt.xlabel('Frequency')
plt.ylabel('Relationship Type')
plt.tight_layout()
plt.savefig('relationship_distribution.png', dpi=300)
plt.close()

# Concept (Node) Distribution
plt.figure(figsize=(10, 6))
top_degree_nodes = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)[:15]
nodes, degrees = zip(*top_degree_nodes)
sns.barplot(x=list(degrees), y=list(nodes), palette='magma')
plt.title('Top 15 Semantic Concepts by Total Connections')
plt.xlabel('Total Connections (In + Out)')
plt.ylabel('Semantic Concept')
plt.tight_layout()
plt.savefig('concept_distribution.png', dpi=300)
plt.close()

# 5. Visualise the Network
plt.figure(figsize=(16, 16))
# A spring layout pushes disconnected nodes apart and pulls connected ones together
pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

# Scale node sizes based on their importance (centrality)
node_sizes = [v * 5000 for v in degree_centrality.values()]

# Draw the graph elements
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8, edgecolors='white')
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=10, alpha=0.4)

# Apply labels only to the top hubs to prevent text overlapping and clutter
threshold = sorted(degree_centrality.values(), reverse=True)[min(25, len(degree_centrality)-1)]
labels = {node: node for node, cent in degree_centrality.items() if cent >= threshold}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

plt.title('Semantic Network Graph of AI in Marketing', fontsize=20)
plt.axis('off')
plt.tight_layout()
plt.savefig('semantic_network_graph.png', dpi=300)
plt.close()