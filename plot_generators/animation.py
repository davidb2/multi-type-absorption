import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation
import colorcet as cc
import itertools

from classes import barbell_graph
from absorption import _dB_trial_absorption_time
import random

random.seed(2024)

# Create Graph
N = 21
G = barbell_graph(N)
pos = nx.layout.spiral_layout(G) # nx.fruchterman_reingold_layout(G)


fig, ax = plt.subplots(figsize=(15,8))
# sns.color_palette("coolwarm", as_cmap=True)
palette = sns.color_palette(palette='coolwarm', n_colors=N)

absorbed = False
def update(t):
  steps, S = t
  ax.clear()

  # Background nodes
  nx.draw_networkx_edges(G, pos=pos, edge_color="black", ax=ax)
  # null_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=set(G.nodes()) - set(path), node_color="white",  ax=ax)
  # null_nodes.set_edgecolor("black")

  node_colors = [None] * N
  for color_idx, locs in S.items():
    for loc in locs:
      node_colors[loc] = palette[color_idx]
  nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, ax=ax)

  # Query nodes
  # query_nodes = nx.draw_networkx_nodes(G, pos=pos, nodelist=path, node_color=idx_colors[:len(path)], ax=ax)
  # query_nodes.set_edgecolor("white")
  # nx.draw_networkx_labels(G, pos=pos, labels=dict(zip(path,path)),  font_color="white", ax=ax)
  # edgelist = [path[k:k+2] for k in range(len(path) - 1)]
  # nx.draw_networkx_edges(G, pos=pos, edgelist=edgelist, width=idx_weights[:len(path)], ax=ax)

  # Scale plot ax
  ax.set_title(f"Steps: {steps}", fontweight="bold")
  ax.set_xticks([])
  ax.set_yticks([])


def process():
  for step, S in _dB_trial_absorption_time(
    G,
    max_steps=None,
    interactive=True,
    mutation_rate=0.0,
    num_initial_types=N,
    full_final_info=False,
  ):
    # Show the first frame many times, so it is an initial pause.
    if step == 0:
      yield from itertools.repeat((step, S), times=100)
    yield (step, S)
  
  yield from itertools.repeat((step, S), times=100)

ani = matplotlib.animation.FuncAnimation(fig, update, frames=process, interval=25, repeat=False)

Writer = matplotlib.animation.writers['pillow']
writer = Writer(fps=1000//25, bitrate=1800)
ani.save('plots/barbell-n21.gif', writer=writer)
# plt.show()