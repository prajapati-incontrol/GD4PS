# Review of topics covered in ML for Graph Data Course: CS4350

## Lecture 1: Graph terminology
- Simple graphs: w/o self-loops and multi-edges
- Multi-graphs: with self-loops and multi-edges
- Sub-graph = Induced graph = subset of graph
- Trail, path, walk, cycle
- Adjacency matrix $O(N^2)$, gaph density, adjacency list $O(E)$, edge list
- Directed vs. Undirected Graphs
- Connected vs. Disconnected Graphs
- Power of adjacency matrix
- Strongly vs. weakly connected: every pair is reachable
- In-degree, out-degree, degree matrix
- Incidence matrix: more suitable for directed graphs
- Graph-laplacian: difference operator
- Multi-dimensional node features
- Abstract networks, correlation networks, $\epsilon-nn$, k-nn
                 
## Lecture 2: Graph ML via Regularization
- What are the applications?
  1. Graph-signal reconstruction: missing values 
  2. Signal classification: or graph classification
  3. Signal prediction: Predict Traffic
  4. Node classification
  5. Link prediction
  6. Clustering & community detection
- Regularizers are cool! How can I build mine? Regularizers have fitting term and regularization term.
  1. Fitting term:
     - Measures the fidelity between what you observe and what you want to obtain
     - Semi-supervised and supervised setting
     - Best for regression problems and simple classification
     - No graph structure used here
  3. Regularization term:
     - Here use the coupling between graph and data
     - The measure should be monotonically non-decreasing 
   
- Measure of global signal variability = Laplacian which incorporates edge-derivative, node-gradient
- When prior is that signal around the neighborhood of a node has similar values, use Tikhonov regularizer bias-variance trade-off
- When prior is that signal around a group of nodes has similar values but can be arbitrarily different from other groups use Trend Filtering
- When prior is that the signal variation between true signal and that exchanged with neighbors is small use Total variational regularizer with L-2 Norm
- When prior is that the signal variation between true signal and that exchanged with neighbors is small but arbitrarily high at a few nodes, use Total variational regularizer with L-1 Norm

## Lecture 3: Traditional Graph ML Methods
- What is semi-supervised? How is it different than supervised and unsupervised learning?

  Supervised learning requires large amount of labeled data (data where both the input and output are known). The model is trained to map inputs to the correct outputs using the labeled data.

  Unsupervised learning uses *unlabeled* data, meaning the model works only with input data and must discover structure on its own. No need for labaled data.

  Semi-supervised learning uses a mix of labeled and unlabaled data, typically small portion of labelled data and a large portion of unlabeled data. In graph-based recommendation system, a small set of users' preferences may be labeled and the model learns to predict preferences for others by also considering relationships between users.
- Semi-supervised methods for node-classification: Label Propagation using random walks and its convergence, What is label connected?, Random walk with restart, Nearest neighbour regressor, Collaborative filtering
- Graph ML Pipeline: What are input features, loss function, train/validation/test, evaluation metrics? This depends on what is your task.
  1. Node-level prediction: Node-level features like eigenvector centrality, betweenness centrality, closeness centrality, clustering coefficient, GDV
  2. Edge-level prediction:
     1. Distance based features: Link score based on shortest path distance
     2. Local Neighborhood overlap: Common neighbors, Jaccard's coefficient, Adamic Adar index
     3. Global neighborhood overlap: Katz index
  3. Graph-level prediction:
     1. Graph feature vector: Bag of nodes, bag of graphlets, Iterative neighborhood aggregation like Weisfeiler-Lehman Algorithm
- Evaluation metrics for graph ML:
  1. Mutli-classification: Accuracy, Precision, Recall, F1 Score
  2. Regression: RMS, MAE
  3. Binary/Multi-label classification: Ac, Pre, Reca, F1 Score, AUROC Score, Average Precision
All above are hand crafted features. If you want automatic feature extraction then refer below chapter.
## Lecture 4: Unsupervised Graph Representation Learning

To automatically learn features/representations/embeddings from graph structure information: these features are called embeddings. This chapter only focuses on node embeddings extraction.
- What is representation learning? Automatically learn the embeddings.
- Node embeddings: Embed similar nodes closer in the embedding space
- Shallow Encoder-Decoder perspective: Encoder is a look-up table where each node is assigned a specific embedding vector. Decoder maps the similarity of embeddings to a similarity score between a pair of nodes. How to determine this similarity score? a. Are the nodes linked, do they share the same neighbors, do they have same structural roles? Following techniques define this similarity score.
  1. Random-walk embeddings: In n-step random walk if nodes co-occur then they are similar.
  2.  Bias (Node2Vec) vs. Uniform R.W (DeepWalk) vs. Alternative walks for directed Graphs (NERD)
- Negative sampling
- BFS, DFS
- Matrix Factorization: Learn low-dimensional approximation of a node-node similarity matrix S by factorization.
- HOPE: MatFact approach for directed graphs using polynomial matrices
- Problems with shallow encoding. To solve the problems of DW, N2V, HOPE etc, solution is GNNs.
## Lecture 5: Graph Convolutional Neural Networks 

## Lecture 6: GNN Architectures

## Lecture 7: Spectral Analysis 

## Lecture 8: Graph-Time Learning

## Lecture 9: Learning over Temporal Graphs

## Lecture 10: Scalability 

## Lecture 11: Interpretability

## Lecture 12: Privacy
