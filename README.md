# Knowledge Distillation to Mixture of Experts

This repository is for **Graph Knowledge Distillation to Mixture of Experts** project.
It contains code for knowledge distillation from GNN to MLP, MoE and RbM.


## Preparing datasets
To run experiments for dataset used in the paper, please download from the following links and put them under `data/` (see below for instructions on organizing the datasets).

- *DGL data* (`cora`, `citeseer`, `pubmed`) are automatically downloaded.

- *CPF data* (`a-computer`, and `a-photo`): Download the '.npz' files from [here](https://github.com/BUPT-GAMMA/CPF/tree/master/data/npz).

- *OGB data* (`ogbn-arxiv` and `ogbn-products`): Datasets will be automatically downloaded when running the `load_data` function in `dataloader.py`. More details [here](https://ogb.stanford.edu/).


## How to run distillation to MLP
All the code was tested on Python 3.8.13
```bash
python main_distill.py -d <DATASET_TYPE> -t <TEACHER_TYPE> -m <RUN_MODE> -s <STUDENT_TYPE>  --config <PATH_TO_CONFIG> [--reliable_sampling] [--positional_encoding] [--similarity_distill] [--adv_augment] [--label_propagation] [--gpu_id <GPU_ID>] [--seed <SEED>] [--batch_size <SIZE>]
```
<DATASET_TYPE> can be *cora*, *citeseer*, *pubmed*, *amazon-com*, *amazon-photo*, *academic-cs*, *academic-physics*, *ogbn-arxiv* or *ogbn-products*.<br>
<TEACHER_TYPE> is either *gcn* or *sage*.<br>
<RUN_MODE> is either *inductive* or *transductive*.<br>
<STUDENT_TYPE> is one of *mlp*, *moe* or *rbm*.<br>
<PATH_TO_CONFIG> is one of the run configs (see **config** fonder).<br>
<GPU_ID> is an id of a gpu. If negative, will run on cpu.<br>
<SEED> fixes the seed of a random generator. If negative, will run with a rnadom seed.


### Our setput allows to emulate main baseline configurations
To run NOSMOG configuration use:
```bash
python main_distill.py -d <DATASET_TYPE> -m <RUN_MODE> --config <PATH_TO_CONFIG> -t sage -s mlp --positional_encoding --similarity_distill --adv_augment --batch_size 4096 [--gpu_id <GPU_ID>] [--seed <SEED>]
```

To run KRD configuration use:
```bash
python main_distill.py -d <DATASET_TYPE> -m <RUN_MODE> --config <PATH_TO_CONFIG> -t sage -s mlp --reliable_sampling [--gpu_id <GPU_ID>] [--seed <SEED>]
```

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@article{rumiantsev2024graph,
  title={Graph Knowledge Distillation to Mixture of Experts},
  author={Pavel Rumiantsev and Mark Coates},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=vzZ3pbNRvh}
}
```

## Acknowledgements

1. NOSMOG: Learning Noise-robust and Structure-aware MLPs on Graphs ([ArXiv](https://arxiv.org/pdf/2208.10010.pdf), [Code](https://github.com/meettyj/NOSMOG/))
2. Quantifying the Knowledge in GNNs for Reliable Distillation into MLPs ([ArXiv](https://arxiv.org/pdf/2306.05628.pdf), [Code](https://github.com/LirongWu/KRD/))
3. Classifying Nodes in Graphs without GNNs ([ArXiv](https://arxiv.org/pdf/2402.05934.pdf), [Code](https://github.com/dani3lwinter/CoHOp))
