# [ECCV-2024] Dual-Decoupling Learning and Metric-Adaptive Thresholding for Semi-Supervised Multi-Label Learning
The implementation for the paper ["Dual-Decoupling Learning and Metric-Adaptive Thresholding for Semi-Supervised Multi-Label Learning"](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06917.pdf) (ECCV 2024). 

## Preparing Data
See the ``README.md`` file in the ``data`` directory for instructions on downloading and preparing the datasets.

## Training Model
To train a model, we have provided a running example in the ``run.sh`` script, for each dataset on each ``lb_ratio``. 

To reproduce the results from the paper, simply execute the following command after finishing data preprocessing above:
```
bash run.sh
```
Of course, readers can also modify the parameters as needed.
