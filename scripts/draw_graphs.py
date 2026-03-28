import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ML-PLA Graph Structures (Simplified)', fontsize=20, fontweight='bold', y=1.05)

# ------------------------------------------------------------------
# 1. Ligand Graph (g1) - A small molecule graph
# ------------------------------------------------------------------
ax = axes[0]
ax.set_title('Ligand Graph (g1)\n(Drug Molecule)', fontsize=14)

G_ligand = nx.Graph()
# Atoms
G_ligand.add_node(1, type='C', color='#808080')
G_ligand.add_node(2, type='C', color='#808080')
G_ligand.add_node(3, type='O', color='#ff0000')
G_ligand.add_node(4, type='N', color='#0000ff')
G_ligand.add_node(5, type='C', color='#808080')
G_ligand.add_node(6, type='F', color='#00ff00')

# Bonds
G_ligand.add_edges_from([(1,2), (2,3), (2,4), (4,5), (5,6), (5,1)])

pos_ligand = nx.spring_layout(G_ligand, seed=42)
colors_ligand = [nx.get_node_attributes(G_ligand, 'color')[n] for n in G_ligand.nodes()]
labels_ligand = nx.get_node_attributes(G_ligand, 'type')

nx.draw(G_ligand, pos_ligand, ax=ax, with_labels=True, labels=labels_ligand, 
        node_color=colors_ligand, node_size=1000, font_color='white', font_weight='bold', edge_color='black', width=2)
ax.text(0.5, -0.15, 'Nodes: Atoms (C, N, O...)\nEdges: Chemical Bonds', 
        ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))


# ------------------------------------------------------------------
# 2. Protein Microenvironment Graph (residue / g4)
# ------------------------------------------------------------------
ax = axes[1]
ax.set_title('Protein Pocket Graph (g4)\n(Target Receptor)', fontsize=14)

G_protein = nx.Graph()
# Residues
G_protein.add_node(1, type='ALA', color='#fbb4ae')
G_protein.add_node(2, type='LEU', color='#fbb4ae')
G_protein.add_node(3, type='TYR', color='#ccebc5')
G_protein.add_node(4, type='GLU', color='#decbe4')
G_protein.add_node(5, type='LYS', color='#fed9a6')
G_protein.add_node(6, type='HIS', color='#ffffcc')

# Sequence and Spatial KNN Edges
G_protein.add_edges_from([(1,2), (2,3), (3,4), (4,5), (5,6)]) # sequence
G_protein.add_edges_from([(1,4), (2,6), (3,5)]) # Spatial proximity edges (KNN/Distance)

pos_protein = nx.spring_layout(G_protein, seed=10)
colors_protein = [nx.get_node_attributes(G_protein, 'color')[n] for n in G_protein.nodes()]
labels_protein = nx.get_node_attributes(G_protein, 'type')

nx.draw(G_protein, pos_protein, ax=ax, with_labels=True, labels=labels_protein, 
        node_color=colors_protein, node_size=1200, font_color='black', font_weight='bold', font_size=8, edge_color='gray', width=2, style='dashed')
ax.text(0.5, -0.15, 'Nodes: Amino Acids (TYR, LYS...)\nEdges: Sequence & Spatial Proximity (<10Å)', 
        ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

# ------------------------------------------------------------------
# 3. Protein-Ligand Interaction Graph (g2/g3)
# ------------------------------------------------------------------
ax = axes[2]
ax.set_title('Interaction Cross-Graph (g2/g3)\n(The Binding Mechanism)', fontsize=14)

G_interact = nx.Graph()

# Add Ligand Nodes (Left Side)
G_interact.add_node('L_O', pos=(0, 2), color='#ff0000', label='O\n(Ligand)')
G_interact.add_node('L_N', pos=(0, 1), color='#0000ff', label='N\n(Ligand)')
G_interact.add_node('L_F', pos=(0, 0), color='#00ff00', label='F\n(Ligand)')

# Add Protein Nodes (Right Side)
G_interact.add_node('P_TYR', pos=(1, 2.5), color='#ccebc5', label='TYR\n(Protein)')
G_interact.add_node('P_GLU', pos=(1, 1), color='#decbe4', label='GLU\n(Protein)')
G_interact.add_node('P_LYS', pos=(1, -0.5), color='#fed9a6', label='LYS\n(Protein)')

# Connect Non-covalent Interactions (e.g., Hydrogen bonds, Van der Waals)
G_interact.add_edges_from([('L_O', 'P_TYR'), ('L_N', 'P_GLU'), ('L_F', 'P_LYS'), ('L_O', 'P_GLU')])

pos_interact = nx.get_node_attributes(G_interact, 'pos')
colors_interact = [nx.get_node_attributes(G_interact, 'color')[n] for n in G_interact.nodes()]
labels_interact = nx.get_node_attributes(G_interact, 'label')

nx.draw(G_interact, pos_interact, ax=ax, with_labels=True, labels=labels_interact, 
        node_color=colors_interact, node_size=1500, font_color='black', font_weight='bold', font_size=8, edge_color='red', width=3, style='dotted')
ax.text(0.5, -0.15, 'Nodes: Ligand Atoms & Protein Residues\nEdges: Chemical Interactions (<5Å)', 
        ha='center', va='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))


plt.tight_layout()
plt.savefig('graph_examples.png', dpi=300, bbox_inches='tight')
print("Successfully generated graph_examples.png")
