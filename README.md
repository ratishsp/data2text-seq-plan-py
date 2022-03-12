# data2text-seq-plan-py [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/data-to-text-generation-with-variational/data-to-text-generation-on-mlb-dataset-2)](https://paperswithcode.com/sota/data-to-text-generation-on-mlb-dataset-2?p=data-to-text-generation-with-variational)
This repo contains code for [Data-to-text Generation with Variational Sequential Planning](https://arxiv.org/abs/2202.13756) (Ratish Puduppully and Yao Fu and Mirella Lapata;  In Transactions of the Association for Computational Linguistics (TACL)); this code is based on an earlier (version 0.9.2) fork of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

## Citation
```
@article{puduppully-2021-seq-plan,
  author    = {Ratish Puduppully and Yao Fu and Mirella Lapata},
  title     = {Data-to-text Generation with Variational Sequential Planning},
  journal = {Transactions of the Association for Computational Linguistics (to appear)},
  url       = {https://arxiv.org/abs/2202.13756},
  year      = {2022}
}
```

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```


## Code Details
The steps for training and inference for the MLB dataset are given in [README_MLB](README_MLB.md).

## Model
The links for the models are [MLB](https://huggingface.co/ratishsp/SeqPlan-MLB), [RotoWire](https://huggingface.co/ratishsp/SeqPlan-RotoWire) and [German RotoWire](https://huggingface.co/ratishsp/SeqPlan-GermanRotoWire).

## Model Outputs
The model outputs are at [MLB](https://huggingface.co/datasets/GEM-submissions/ratishsp__seqplan__1646397329/raw/main/submission.json) and [German-RotoWire](https://huggingface.co/datasets/GEM-submissions/ratishsp__seqplan__1646397829/blob/main/submission.json).

## Acknowledgements
Part of the code is based on the [Sequential Knowledge Transformer repo](https://github.com/bckim92/sequential-knowledge-transformer).

