# GNN — Tutorial Notebooks

A collection of Jupyter notebooks covering **Graph Neural Networks (GNNs)** from the ground up.

Each notebook provides theory, worked implementations, and exercises.

---

## Notebooks

| # | Notebook | Topics |
|---|----------|--------|
| 1 | [`01_pytorch_and_pytorch_geometric.ipynb`](01_pytorch_and_pytorch_geometric.ipynb) | PyTorch tensors, autograd, `nn.Module`, graph data, PyG `Data`, `MessagePassing`, `DataLoader` |
| 2 | [`02_gcn_graphsage_gat.ipynb`](02_gcn_graphsage_gat.ipynb) | GCN, GraphSAGE (with neighbourhood sampling), GAT (multi-head attention), comparison on Cora |
| 3 | [`03_knowledge_graph_embeddings.ipynb`](03_knowledge_graph_embeddings.ipynb) | TransE, TransR, RotatE, ComplEx, LiteralE, Heterogeneous Graphs |
| 4 | [`04_advanced_gnns.ipynb`](04_advanced_gnns.ipynb) | Graph Transformer, Graphormer, Heterogeneous Graph Transformer (HGT), R-GCN |

---

## Requirements

```bash
pip install torch torchvision
pip install torch_geometric
# Optional (faster sparse operations):
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
# For visualisation:
pip install networkx matplotlib scikit-learn
```

## Topics Covered

- **PyTorch & PyTorch Geometric** — tensor operations, autograd, custom `nn.Module`, graph data representation, message passing framework, mini-batching  
- **GCNs, GraphSAGE, GATs** — spectral convolution, inductive learning, attention mechanisms, over-smoothing  
- **Knowledge Graph Embeddings** — TransE, TransR, RotatE, ComplEx, LiteralE, heterogeneous graphs with PyG `HeteroData`  
- **Advanced GNNs** — Graph Transformer (sparse MHA + LPE), Graphormer (spatial/centrality/edge biases), HGT (type-specific attention), R-GCN (relation-specific weights + basis decomposition)
