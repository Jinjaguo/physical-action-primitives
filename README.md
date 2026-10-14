# Physical Action Primitives

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org)
[![HF Collection](https://img.shields.io/badge/%F0%9F%A4%97-Models%20%26%20Datasets-yellow)](https://huggingface.co/collections/jinjaguo/physical-action-primitives-pap-6acf000c390e5c2872dec03b)

Official code for **Beyond Verbs and Trajectories: Physical Action Primitives through Local Interaction Laws**.

Jinjia Guo\*, Xinyu Wang\*, Bowen Sun, Dongqi Zuo, Jing Du
University of Florida · University of California, Berkeley (\*equal contribution)

Action primitives provide a natural way for robots to reuse experience, yet a basic question remains unresolved: **what should a reusable primitive preserve?** Similar action labels, trajectories, and nominal effects do not establish compatible physical responses to changed interventions. We introduce **Physical Action Primitives (PAPs)**, which organize reusable action knowledge by **local intervention–response structure**. A PAP retains this structure as an explicit, persistent physical response operator that can be **queried** for predicted outcomes and **invoked** to select interventions for desired effects, while remaining fixed as downstream execution is substituted or corrected from measurement.

Independent physical audits show that PAP organization produces more consistent response neighborhoods than semantic, trajectory, latent-skill, and nominal-effect organization, reducing audit discrepancy area by **26.7%** relative to semantic organization and **19.1%** relative to trajectory organization. A nonlinear PAP construction achieves closely matched organization performance, supporting the organizing principle across affine and nonlinear parameterizations. Retained PAPs predict responses beyond their identification probes, support repeated invocation under changing requests and contexts, and carry source-dependent information that measurably affects control.

## Repository layout

| Package (`src/physical_action_primitives/`) | Contents |
| --- | --- |
| `corpus/` | Collection and checks of the pretraining interaction corpus: fresh episodes, branches from shared parent states, and a historical supplement. |
| `backbone/` | The 50M / 200M / 800M physical backbone: model, data pipeline, normalization, and training. |
| `construction/` | PAP construction: frozen-backbone feature extraction, response heads (PAP-Affine, PAP-Nonlinear, PAP-Endpoint, PAP-ActionSeq), Query/Invoke interfaces, and model selection. |
| `source_exchange/` | Controlled exchange of retained source conditions and the context-only comparison. |
| `execution/` | Execution substitution and measured calibration. |
| `organization/` | Physical-organization audit comparing PAP, semantic, trajectory (DMP), SPiRL, and endpoint organizers. |
| `query_invoke/` | Prospective Query/Invoke evaluation across PushT, PokeCube, and RollBall. |
| `simulation/` | ManiSkill3 task, controller, and rigid-response utilities. |
| `configs/` | Training and evaluation configurations. |
| `tests/` | CPU tests for the backbone and the PAP contracts. |

## Installation

```bash
git clone https://github.com/Jinjaguo/physical-action-primitives.git
cd physical-action-primitives
conda create -n pap python=3.11 -y && conda activate pap
# Install a PyTorch build that matches your CUDA version: https://pytorch.org/get-started/locally/
pip install -e '.[simulation,analysis,test]'
```

Simulation and training use Linux, an NVIDIA GPU, PyTorch, and [ManiSkill3](https://github.com/haosulab/ManiSkill) / SAPIEN. CPU-only use (loading checkpoints, running the PAP contract tests) needs only NumPy, SciPy, h5py, and PyTorch. The SPiRL comparator uses the upstream implementation at a pinned commit:

```bash
bash scripts/install_spirl.sh
```

All data, checkpoints, and results live under one directory selected by `PAP_ROOT` (default: the current directory):

```bash
export PAP_ROOT=/path/to/workspace
```

```
$PAP_ROOT/
├── data/corpus/{fresh,historical,branched}/   pretraining interaction corpus
├── data/manifests/                            corpus, split, and normalization manifests
├── data/organization/                         physical-organization study data
├── data/query_invoke/                         Query/Invoke study data
├── checkpoints/backbone/{50m,200m,800m}/      pretrained backbones
├── checkpoints/pap/                           PAP response models
└── results/                                   evaluation outputs
```

## Downloads

Models and data are hosted on Hugging Face in the [PAP collection](https://huggingface.co/collections/jinjaguo/physical-action-primitives-pap-6acf000c390e5c2872dec03b). The repository layouts mirror `$PAP_ROOT`, so a single download command places files where the code expects them:

| Repository | Contents | Size |
| --- | --- | --- |
| [`jinjaguo/pap-backbone`](https://huggingface.co/jinjaguo/pap-backbone) | 50M / 200M / 800M backbones | 4.2 GB |
| [`jinjaguo/pap-response-models`](https://huggingface.co/jinjaguo/pap-response-models) | PAP response models (affine, nonlinear, endpoint, action-sequence), organization models | 0.1 GB |
| [`jinjaguo/pap-corpus`](https://huggingface.co/datasets/jinjaguo/pap-corpus) | Pretraining corpus and PPO teacher policies | 267 GB |
| [`jinjaguo/pap-study-data`](https://huggingface.co/datasets/jinjaguo/pap-study-data) | Data and results of the evaluation studies | 1.7 GB |
| [`jinjaguo/pap-legacy-backup`](https://huggingface.co/jinjaguo/pap-legacy-backup) | Original-format archive (not loadable by this code) | 17 GB |

```bash
pip install -U huggingface_hub
hf download jinjaguo/pap-backbone --local-dir $PAP_ROOT
hf download jinjaguo/pap-response-models --local-dir $PAP_ROOT
hf download jinjaguo/pap-corpus --repo-type dataset --local-dir $PAP_ROOT   # only needed to retrain the backbone
```

## Usage

Every stage is a module with a command line, run from anywhere with `python -m`.

```bash
# 1. Collect the pretraining corpus (see corpus/ for the shard-level entry points)
python -m physical_action_primitives.corpus.collect_fresh --help

# 2. Train the backbone
python -m physical_action_primitives.backbone.train --config configs/backbone/train_800m.json

# 3. Build PAPs: extract frozen-backbone features, train and select response models
python -m physical_action_primitives.construction.extract_features --help
python -m physical_action_primitives.construction.train --help
python -m physical_action_primitives.construction.select_models --help

# 4. Evaluate
python -m physical_action_primitives.organization.evaluate --help
python -m physical_action_primitives.query_invoke.evaluate --help
python -m physical_action_primitives.source_exchange.evaluate --help
python -m physical_action_primitives.execution.evaluate --help
```

Recorded configurations are in `configs/`.

## Citation

```bibtex
@inproceedings{guo2027pap,
  title     = {Beyond Verbs and Trajectories: Physical Action Primitives through Local Interaction Laws},
  author    = {Guo, Jinjia and Wang, Xinyu and Sun, Bowen and Zuo, Dongqi and Du, Jing},
  year      = {2027}
}
```

## License

Code is released under the [MIT License](LICENSE). Released model weights and datasets are covered by the licenses stated on their Hugging Face pages.
