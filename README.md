# DAG-GNN

Code for DAG-GNN work

## Getting Started

### Prerequisites

```
Python 3.7
PyTorch >1.0
```


## How to Run 

Synthetic data experiments 

### Synthetic Experiments

CHOICE = linear, nonlinear_1, or nonlinear_2, corresponding to the experiments in the paper

```
python train.py --graph_linear_type=<CHOICE>
```


## Cite

If you make use of this code in your own work, please cite our paper:

```
@inproceedings{yu2019dag,
  title={DAG-GNN: DAG Structure Learning with Graph Neural Networks},
  author={Yue Yu, Jie Chen, Tian Gao, and Mo Yu},
  booktitle={Proceedings of the 36th International Conference on Machine Learning},
  year={2019}
}
```


## Acknowledgments
Our work and code benefit from two existing works, which we are very grateful.

* DAG NOTEAR https://github.com/xunzheng/notears
* Neural relational inference for interacting systems https://github.com/ethanfetaya/nri



