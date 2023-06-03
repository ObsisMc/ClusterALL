# ClusterALL

This is a bachelor thesis finished in 2023. 
This work studies **node-level tasks of graphs**, especially **node property prediction in large-scale graphs**. 

**Overview**

1. [Report](#report)
   1. [Abstract](#abstract)
   2. [Problems](#problems)
   3. [Proposed Method](#proposed-method)
   4. [Dataset](#dataset)
   5. [Experiments](#experiments)
2. [Project Structure](#project-structure)
3. [Build](#build)
4. [Q&A](#Q&A)



## Report

The whole report  is in the root directory named *Node feature enhancement in large-scale graphs based on deep clustering.pdf*, and you can also access it by clicking [here](https://github.com/ObsisMc/undergraduate_thesis/blob/main/Node%20feature%20enhancement%20in%20large-scale%20graphs%20based%20on%20deep%20clustering.pdf). However, it only has Chinese version.

### Abstract

Graph structures are widely used in various scenarios such as thesis citation networks, which contain three main types of tasks: node-level, edge-level and graph-level tasks, which classify and predict attributes of nodes, edges and the whole graph, respectively. As the most fundamental element of graphs, nodes have important value in all above tasks, and how to characterize them effectively is always an important topic. Many works characterize a node mainly by considering its information in the graph alone, but there is a lack of research on node representation from the perspective of node sets, but intuitively, node sets contain more detailed and unique information about the nodes than that in the graph, which may help to improve node representation. To this end, there are works that use graph partition to group similar nodes together and then perform further node representation, however, they are usually two-step approaches, i.e., first using a trained model to get nodes' embedding and then using a graph partition algorithm to obtain the node set, followed by further encoding and decoding, which means the partition algorithm is not task-specific. Therefore, this paper proposes a plug-and-play plug-in model based on clustering and its training and testing process. The algorithm first uses a graph partitioning algorithm to obtain an initial partition, creates cluster nodes and corresponding edges for each cluster and adds them to the original graph to obtain an augmented graph. After integrating the original model into this model, the augmented graph can be encoded to obtain the enhanced node representations and the updated graph partitioning. This model integrates node feature encoding and graph partitioning into one model, which can perform both representation learning and partitioning learning for different tasks, and thanks to the plug-in format, this model can be applied to most existing models to enhance their node representation effects and thus improve the metrics of the corresponding tasks to a certain extent.

### Problems

In brief, this work focuses on the following problems:

1. In large-scale graphs, **scalability** is important so many works use sampling to reduce models' complexity, which makes models miss **global information** in the graphs. 
2. When graphs are too large, people often use mini-batch training. However, **in a certain batch, nodes' receptive field is limited in it**, which means no matter what the model is, the nodes always only get local information when training.

### Proposed Method

To solve the problems above, this work designs an algorithm called **ClusterALL**. It is an inductive and **plug-in** model, which means it must run with other models, and this work aims to use ClusterALL to improve the effect of other models in node property prediction tasks.

ClusterALL uses clustering to let nodes get global info even they are in a certain batch and the model has scalability.

![ClusterALL structure](https://raw.githubusercontent.com/ObsisMc/undergraduate_thesis/049c313bf4739bac001ae252911d17a00951108f/README.assets/ClusterALL%20structure.svg?token=ARFJJWN2PVBKYWK7AKJPV73EPMNPK)

 Method (see the figure above):

- Train

  We use training 

  1. Pre-processing:

     1. firstly runs graph partition algorithm (like Metis) to get clusters.
     2. adds additional nodes, called cluster center nodes (CCnodes), into the original graph and connects them with original nodes according to the cluster result.

  2. Runs model:

     we have a deep learning model called RAC and CCnodes are registered into it so that CCnodes can be trained under global receptive field. RAC also learn how to partition the original graph and then updates clusters in the graph. **Other models are put in RAC.**

- Test

  1. adds testing subgraph into training subgraph.
  2. loads trained cluster result, then run RAC.
  3. gets predicted node properties and if needed, outputs cluster result of testing data.



RAC's structure is following:

![ablated RAC structure](https://raw.githubusercontent.com/ObsisMc/undergraduate_thesis/049c313bf4739bac001ae252911d17a00951108f/README.assets/RAC%20structure.svg?token=ARFJJWI325W2XBMSXIZMIELEPMNSE)

The attention module can learn which nodes belong to which clusters. **Encoder is where other models is put into.**

### Dataset

Dataset is [OGB](https://ogb.stanford.edu/) and this work uses [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) and [ogbn-proteins](https://ogb.stanford.edu/docs/nodeprop/#ogbn-proteins). They both have a large graph that has over 10w nodes and 100w edges.

### Experiments

Please see report.



## Project Structure

- `AbstractClusteror.py`

Our algorithm, an interface. You must modify your model and code before using our interface. The following classes are important:

1. class `AbstractClusteror`

   It is RAC in the report and must be implemented

2. class `AbstractClusterDataset`, `AbstractClusterLoader`

   They are for pre-processing and you just need to use them and don't need to implement them.

3. class `ClusterOptimizer`

   It is about training strategy. 

- directory `nodeformer` and `ogb_models`

  They are examples of how to use ClusterALL

  - `nodeformer`

    It is about **Nodeformer** model, please see [qitianwu/NodeFormer](https://github.com/qitianwu/NodeFormer) to learn about it. 

    - To train nodeformer with ClusterALL, you need to run `main_nodeformer`.

    - `NodeformerCluster.py` has implemented interface of ClusterALL.

  - `ogb_models`

    It is about **MLP**, **GCN** and **GraphSAGE** model. It has `arxiv` and `proteins` directories which corresponds to different datasets.

    - `gnn.py` and `mlp.py` are from OGB ([code of arxiv](https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/arxiv), [code of proteins](https://github.com/snap-stanford/ogb/tree/master/examples/nodeproppred/proteins)).

    - To train GCN and GraphSAGE, please run `main_gnn.py`; to train MLP, please run `main_mlp.py`.
    - `GNNCluster.py` and `MLPCluster.py` have implemented interface of ClusterALL.

- directory `script`

  It has training scripts to train nodeformer and OGB's models. You can use them to have a try.

- `analysis.py` & `analysis_utils.py`

  It is used for analyze ClusterALL and isn't related to model training and testing.

  - `analysis_utils.py` can plot figures in report. You can download data from [baiduyun](https://pan.baidu.com/s/1QrUeFbN_MC72h_MT4r8SQQ?pwd=2hj5), put them in directory `analysis_data` and try to plot some.

## Build 

The project mainly needs

- PyTorch (1.12.1)
- torch-geometric (2.3.0)
- ogb (1.3.5)

You can use `envirnoments.yml` to install dependencies. 

## Q&A

If has any questions, welcome to create a issue.