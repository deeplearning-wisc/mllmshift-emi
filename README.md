# [Understanding Multimodal LLMs Under Distribution Shifts: An Information-Theoretic Approach (ICML'25)](https://arxiv.org/abs/2502.00577v2)
> by [Changdae Oh](https://changdaeoh.github.io/)<sup>1</sup>, [Zhen Fang](https://fang-zhen.github.io/)<sup>2</sup>, [Shawn Im](https://shawn-im.github.io/)<sup>1</sup>, [Xuefeng Du](https://d12306.github.io/)<sup>1</sup>, and [Yixuan Li](https://pages.cs.wisc.edu/~sharonli/)<sup>1</sup>.
> <br/>
> <sup>1</sup>[University of Wisconsin--Madison](https://www.wisc.edu/), <sup>2</sup>[University of Technology Sydney](https://www.uts.edu.au/)

[![Paper](https://img.shields.io/badge/arXiv-2502.00577-orange)](https://arxiv.org/abs/2502.00577v2)
[![Poster](https://img.shields.io/badge/ICML2025-Poster-teal)](https://icml.cc/virtual/2025/poster/44373)
[![Slide](https://img.shields.io/badge/ICLR2025ws-Slide-royalblue)](https://drive.google.com/file/d/1lh1WohIQ0HbX5PQJHsergRJkUj_P5Gtn/view?usp=sharing)
[![Dataset](https://img.shields.io/badge/HFdataset-SyntheticShift-yellow)](https://huggingface.co/datasets/changdae/llavabench-shift-synthetic-v1)
[![Dataset](https://img.shields.io/badge/HFdataset-NaturalShift-yellow)](https://huggingface.co/datasets/changdae/llavabench-shift-natural-v1)


# Overview 
_In this repository, we highlight our proposals with corresponding code and instructions to reproduce our experiments and for potential broad usage._

### Research Highlight
* We presented _**effective mutual information (EMI)**_, $\text{EMI}(P_{XY};P_{\theta}):=I(P_{X}\otimes P_{\theta})-I(P_{XY})$, as a new theory-grounded metric to assess the quality of outputs from an MLLM given input query. 
  * Our theoretical analysis reveals the connection between EMI and LLM-judge-based pair-wise preference score, such as relative preference score or win rate.
* Based on EMI, we proposed _**effective mutual information difference (EMID)**_, $\text{EMID}(P_{XY},Q_{XY};P_{\theta}):=\text{EMI}(P_{XY};P_{\theta})-\text{EMI}(Q_{XY};P_{\theta})$, as an information-theoretic measure of MLLM robustness under distribution shifts.
  * We then provided theoretical upper bound of EMID, which is constructed by $D_{\rm JS}(P_{X_v}||Q_{X_v})$, $D_{\rm JS}(P_{X_t}||Q_{X_t})$, $D_{\rm JS}(P_{Y_{\theta}}||P_{Y})$, and $D_{\rm JS}(Q_{Y_{\theta}}||Q_{Y})$ terms, to characterize performance gap of MLLM under distribution shifts.
* On 61 types of distribution shifts, we validated that empirical EMI estimates have strong correlation with relative preference scores, and EMID upper bound estimates consistently correlated with EMID estimates.

### Procedure
> Our project was built on top of `LLaVA codebase`, and we only provide the pipeline for EMI, EMID, UpperBound computations here, so you can leverage more information about MLLM training and inference from LLaVA [paper](https://arxiv.org/abs/2304.08485) and [repository](https://github.com/haotian-liu/LLaVA/tree/main). 

* Basic information
  * EMI consumes a `(image_query:PILImage, text_query:str, model_response:str, GT_response:str)` tuple as an input to access the quality of a model response.
  * EMID and EMID UB consumes a pair of two tuples from different data distributions to measure the robustness of model response qualities across different input distributions.
  * We compute all the above quantities on top of embeddings from pre-trained encoder models such as CLIP-VIT and RoBERTa to bypass non-trivial MI modeling on raw input space.


* EMID and its upper bound estimation on a pair of two datasets, e.g., one of in-distribtuion (ID) and one of out-of-distribution (OOD)
  1. Do inference on all datasets of your interest to gather responses $Y_{\theta}$ of your models given input queries.
  2. Get embedding vectors $\tilde{X}_{v}$, $\tilde{X}_{t}$, $\tilde{Y}_{\theta}$, and $\tilde{Y}_{gt}$ for the `(image_query, text_query, model_response, GT_response)` tuples with pre-trained vision and text encoders. If you don't have ground truth (GT) responses for datasets, get them by querying a reference model, e.g., GPT-4o.
  3. (Optional) Construct an embedding-pair dataset $\{(\tilde{X},\tilde{Y})\}$, and train a neural MI estimator on it.
  4. You can compute EMI and EMID by feeding embedding tuples into the (pre-)trained MI estimator.
  5. You can also compute EMID UB on top of embedding tuples with RJSD estimator (See `JSD_cov()` function in `main.py`)

# Environment
Exactly the same as the env of llava-v1.5 with `datasets==3.5.0` installation.
``` linux
conda create -n mllmshift-emi python=3.10 -y
conda activate mllmshift-emi

git clone https://github.com/deeplearning-wisc/mllmshift-emi.git
cd mllmshift-emi

pip install --upgrade pip
pip install -e .
pip install datasets==3.5.0
```

# Data Preparation
* To test new models on the llava-bench shift benchmarks, you need to prepare model responses' on all kinds of distribution shifts scenarios (28 for natural, 35 for synthetic).
* You can access our two types of benchmarks through Hugging Face dataset hub in public, [`llavabench-shift-synthetic-v1`](https://huggingface.co/datasets/changdae/llavabench-shift-synthetic-v1) and [`llavabench-shift-natural-v1`](https://huggingface.co/datasets/changdae/llavabench-shift-natural-v1), that contain image query, text query, and gt response (gpt4).
* Refer to the [document for evaluation from LLaVA repository](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) to get **model responses** by doing inference with your MLLMs.

# Run
* We provide a **pre-trained weight for CLUB MI estimator** at `estimator_ckpt/CLUB_global.pt`, so you don't need to re-train MI estimator from scratch.
  * In contrast to that used in our paper, this estimator was trained on a pooled dataset of synthetic and natural shifts dataset with >10K samples, whereas we previously used two separate MI estimators for synthetic and natural shifts.
    * So the replication results would be slightly different with numbers in the paper.
  * [`CAUTION!`] if your downstream tasks are significantly distinct from the llava-bench family of datasets, you may need to retrain it.
* For easy reproduction, we also provide the responses generated from `llava-v1.5-13b` and `llava-v1.6-vicuna-13b` models under the path `data/{DATA_SPLIT_NAME}-{MODEL_NAME}.jsonl`.

```linux
unzip data.zip
python main.py --model_name llava-v1.5-13b --shift_type SYNTHETIC
python main.py --model_name llava-v1.5-13b --shift_type NATURAL
python main.py --model_name llava-v1.6-vicuna-13b --shift_type SYNTHETIC
python main.py --model_name llava-v1.6-vicuna-13b --shift_type NATURAL
```

* After running the above programs, you will find organized results at `results/*.json`.


# Citation
If this repository was useful to your works, please consider to cite our paper!
```
@inproceedings{
oh2025understanding,
title={Understanding Multimodal LLMs Under Distribution Shifts: An Information-Theoretic Approach},
author={Oh, Changdae and Fang, Zhen and Im, Shawn and Du, Xuefeng and Li, Yixuan},
booktitle={International Conference on Machine Learning},
year={2025},
}
```

# Acknowledgement
* We appreciate the amazing work with a fully open codebase from the [`LLaVA`](https://github.com/haotian-liu/LLaVA) authors that enables us to initiate our project.
* We are also sincerely thankful for the authors of [`CLUB`](https://github.com/Linear95/CLUB) and [`RepresentationJSD`](https://github.com/uk-cliplab/representationJSD/tree/main) that allow us to build a reliable estimation framework for the mutual information and Jensen-Shannon divergence.
