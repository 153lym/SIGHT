# SIGHT

SIGHT: Synergizing Implicit Neural Representations and Graph Convolution to Decouple Gene-Spatial Heterogeneity for Spatially Resolved Transcriptomics


## Overview:

We introduce SIGHT, a computational framework that synergizes the topological strengths of Graph Convolutional Networks (GCNs) with the continuous signal modeling capabilities of Implicit Neural Representations (INRs). Instead of restricting information flow to fixed adjacency graphs, SIGHT leverages Sinusoidal Representation Networks (SIRENs) to model spatial coordinates as continuous functions, allowing for the precise capture of complex, high-frequency spatial details that are often lost in discrete grid representations. By explicitly decoupling gene-driven identity from spatially-driven geometric constraints and recombining them via a dual-path reconstruction mechanism, SIGHT effectively resolves the fundamental trade-off between domain smoothness and boundary sharpness.



## Run environment:
 
SIGHT is implemented in the pytorch framework. The detailed running environment can be found in file [SIGHT.yaml](SIGHT.yaml).
In the experiments, the GPU we used was NVIDIA RTX A6000.

## Run

### SRT raw dataSet download
The DLPFC dataset with 10x Visium platform is accessible within the spatialLIBD package (http://spatial.libd.org/spatialLIBD). The mouse liver dataset with 10x Visium platform is collected from https://www.livercellatlas.org/. The HER2 Positive Breast Tumors dataset with ST platform is collected from https://github.com/almaan/her2st. The mouse medial prefrontal cortex dataset with STARmap platform is collected from http://clarityresourcecenter.org/. The mouse hypothalamic preoptic region with MERFISH platform is collected from https://datadryad.org/stash/dataset/doi:10.5061/dryad.8t8s248. The mouse somatosensory cortex profiled dataset with osmFISH platform is collected from http://linnarssonlab.org/osmFISH/availability/. The Human Lymph Node dataset with 10x Visium platform is collected from https://www.10xgenomics.com/datasets/human-lymph-node-1-standard-1-1-0.

### Run SIGHT

We give example on the DLPFC datasets with 10x Visium platform. 
We provide the example code in [main.py](main.py).

`python main.py`

For different SRT data, it is necessary to set the range of alpha and gamma parameters to find the optimal set of parameter values. 
During the experiment, we set the range of [0, 10] for both. The optimal parameters for each data can be found in file `best_param.txt`.

### Results
The clustering results for all data are saved in `results/`, for reference.

## Citation:
**This repository contains the source code for the paper:**

xxxxxx
