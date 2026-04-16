# SIGHT

SIGHT: Synergizing continuous physical geometry and transcriptomic semantic to decode tissue architecture

## Overview:

Here, we present SIGHT, a dual-stream deep learning framework designed to synergistically integrate microenviron-mental topology-based semantics with physically continuous spatial geometry. SIGHT utilizes graph convolutional networks to encode microenvironmental topology-based semantics, while employing sinusoidal representation net-works to directly model spatial coordinates as continuous fields. Fortified by a multi-scale contrastive learning strategy, SIGHT successfully resolves the trade-off between semantic and spatial. We demonstrate SIGHT’s robustness and su-perior performance across diverse platforms, effectively identifying fine-grained spatial domains, denoising data, and inferring cellular trajectories to reveal deeper biological insights into tumor heterogeneity and tissue architecture.



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
During the experiment, we use grid search to search for the best parameters. The optimal parameters for each data can be found in file `best_param.txt`.

### Results
The clustering results for all data are saved in `results/`, for reference.

## Citation:
**This repository contains the source code for the paper:**

SIGHT: Synergizing continuous physical geometry and transcriptomic semantic to decode tissue architecture
