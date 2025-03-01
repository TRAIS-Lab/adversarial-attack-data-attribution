

# Official Implementation of ICLR 2025 Paper: *Adversarial Attacks on Data Attribution*

This repository contains the official implementation of the paper, structured as follows:

## Repository Structure

- **`/models`** – Contains model implementations.
- **`/utils`** – Includes scripts for data preprocessing.

## Training
- **`train_model.py`** – Script for training models.

## Attack Methods
- **`outlier_attack.py`** – Implements the outlier attack method.
- **`shadow_attack.py`** – Implements the shadow attack method.

## Evaluation
- **`eval_inf.py`** – Evaluates the model using Influence Functions.
- **`eval_trak.py`** – Evaluates the model using TRAK.

## Result Comparison
- **`result_compare.py`** – Compares results based on important counts.

## Text Generation Task Pipeline
**`/text-gen`** - Contains Code for Text Generation Task



## Citation

If you find this repo helpful for your research, please consider citing our paper below.

```latex
@article{wang2024adversarial,
  title={Adversarial Attacks on Data Attribution},
  author={Wang, Xinhe and Hu, Pingbang and Deng, Junwei and Ma, Jiaqi W},
  journal={arXiv preprint arXiv:2409.05657},
  year={2024}
}
```
