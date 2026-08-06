# The Binding Effect: Multi-Dimensional Gender Bias in Instruction TTS

Code and prompt sets for the paper
**"The Binding Effect: Analysis of How Multi-Dimensional Cues Form Gender Bias
in Instruction TTS"** (Interspeech 2026).

Kuan-Yu Chen, Yi-Cheng Lin, Po-Chung Hsieh, Huang-Cheng Chou, Chih-Fan Hsu,
Jeng-Lin Li, Hung-yi Lee, Jian-Jiun Ding
National Taiwan University · AI Research Center, Inventec Corporation

[![arXiv](https://img.shields.io/badge/arXiv-2603.20743-b31b1b.svg)](https://arxiv.org/abs/2603.20743)

---

## Overview

Bias evaluations in Instruction TTS (ITTS) usually test **one attribute at a
time**, ignoring that real prompts combine many social cues. We model a prompt
as a composition of three theoretically grounded axes and study how they
interact to shape the perceived **gender** of synthesized speech:

- **Social Status** — Weberian stratification, realized via Social Dominance
  Orientation (SDO) descriptors (high / low status).
- **Career** — socially structured occupational roles (female-/mixed-/male-leaning).
- **Persona** — Big Five dispositional traits (Openness, Conscientiousness,
  Extraversion, Agreeableness, Neuroticism).

The **Binding Effect** is the phenomenon where these cues interact
non-additively: e.g. a female-leaning occupation combined with high-status and
a male-leaning persona can flip the perceived gender toward male. The framework
quantifies this with an **interaction term** measured in log-odds space.

**Research questions**

- **RQ1** — How do compositional interactions among Social Status, Career, and
  Persona modulate latent gender associations vs. univariate baselines?
- **RQ2** — Which dimensions dominate acoustic gender realization, and do some
  attributes systematically override others?
- **RQ3** — Are the observed patterns consistent with the semantic priors of
  pretrained text encoders, the training-data distributions, or both?

The pipeline is **two-stage**: (1) measure univariate gender priors per
descriptor, then (2) compose descriptors across axes and quantify the
interaction term relative to the additive baseline. Gender is estimated with a
wav2vec 2.0 age/gender classifier over 100 samples per prompt.

## Evaluated ITTS models

| Alias | Model | Backbone | Text encoder |
|-------|-------|----------|--------------|
| `voxinstruct` | VoxInstruct | LLaMA (AR + NAR) | mT5-base |
| `promptttspp` | PromptTTS++ | Diffusion + MDN | BERT |
| `parler-mini` | Parler-TTS Mini | AudioLM | Flan-T5-large |
| `parler-large` | Parler-TTS Large | AudioLM | Flan-T5-large |

## Repository contents

```
generate_wav.py        Unified TTS generation for the 4 model backends
analyze_gender.py      Gender detection + bias statistics over generated WAVs
descriptions/          Prompt sets (JSON) for each axis / composition
  ├── descriptions_status_bias.json    Social Status (SDO)
  ├── description_career_bias.json     Career / occupation
  ├── descriptions_persona_bias.json   Persona (Big Five)
  ├── descriptions_two_axis.json       Bi-dimensional compositions
  └── descriptions_multi_axis.json     Tri-dimensional compositions
run_all_models.sh / run_experiments.sh          Batch generation runners
analyze_all_models.sh / analyze_experiments.sh  Batch analysis runners
requirements.txt       Python dependencies
setup.txt              Step-by-step environment setup
```

> **Not included (by design):** the TTS model backends and their checkpoints
> (`models/`), and generated audio / analysis outputs. These are large and are
> excluded via `.gitignore`. The TTS backends are installed separately (below);
> the gender-detection model is auto-downloaded on first run.

### Prompt (JSON) format

Each entry pairs a style **description** with a neutral **prompt_text**:

```json
{
  "id": "0001",
  "description": "Speaks with imaginative phrasing and vivid curiosity...",
  "trait": "Openness",
  "keywords": "curious",
  "prompt_text": "Hey, how are you doing today?"
}
```

## Installation

See [setup.txt](setup.txt) for the full walkthrough. In brief:

```bash
conda create -n BindingBias python=3.9.18 -y
conda activate BindingBias

# Point CUDA_HOME at your local CUDA 11.8 install (see setup.txt)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip==24.0
pip install -r requirements.txt
pip install 'transformers<4.46'
```

**TTS backends** are installed per model and may need separate environments:

- **Parler-TTS** — `pip install git+https://github.com/huggingface/parler-tts.git`
- **PromptTTS++** — install from its official repository (provides the
  `promptttspp` package used by `generate_wav.py`).
- **VoxInstruct** — install from its official repository (provides the
  `utils.utils` / `utils.extract_hubert` modules and `fairseq`). This model
  typically runs in its own conda env (`voxinstruct`), as reflected in the
  batch scripts.

## Usage

### 1. Generate speech

```bash
python generate_wav.py \
    --model parler-large \
    --json descriptions/descriptions_persona_bias.json \
    --output results/parler_large/
```

`--model` ∈ {`parler-large`, `parler-mini`, `promptttspp`, `voxinstruct`}.
Output WAVs are named by id (`0001.wav`, `0002.wav`, …).
Use `--no-skip` to regenerate existing files (default skips them).

### 2. Analyze gender & bias

```bash
python analyze_gender.py \
    --wav_path results/parler_large/ \
    --json descriptions/descriptions_persona_bias.json \
    --output analysis/parler_large/
```

When `--output` is a directory, this writes:

- `detection_results.csv` — per-utterance predictions (id, wav, predicted
  gender, male/female scores, trait, keywords)
- `overall_gender_distribution.csv` — overall female/male distribution
- `gender_by_trait.csv` — female/male ratio per trait
- `gender_by_keyword.csv` — female/male ratio per keyword

The gender detector uses an `audonnx` wav2vec 2.0 age/gender model that is
downloaded and cached automatically on first use.

### Batch runners

```bash
# Override paths via env vars if desired; defaults resolve to this repo.
OUTPUT_BASE=./ITTS_audios bash run_all_models.sh      # generate for all models
WAV_BASE=./ITTS_audios    bash analyze_all_models.sh  # analyze all models
```

`run_experiments.sh` / `analyze_experiments.sh` run a fixed set of
per-model experiment files instead of every JSON in `descriptions/`.

## Dependencies

Python 3.9, PyTorch 2.3 (CUDA 11.8), `transformers<4.46`, plus audio/analysis
packages (`soundfile`, `pyworld`, `vocos`, `audonnx`, `audeer`, `pandas`,
`scikit-learn`, `hydra-core`, `peft`, …). See [requirements.txt](requirements.txt)
and [setup.txt](setup.txt) for the complete, version-pinned list.

## Citation

```bibtex
@inproceedings{chen2026binding,
  title     = {The Binding Effect: Analysis of How Multi-Dimensional Cues Form
               Gender Bias in Instruction TTS},
  author    = {Chen, Kuan-Yu and Lin, Yi-Cheng and Hsieh, Po-Chung and
               Chou, Huang-Cheng and Hsu, Chih-Fan and Li, Jeng-Lin and
               Lee, Hung-yi and Ding, Jian-Jiun},
  booktitle = {Interspeech},
  year      = {2026},
  note      = {arXiv:2603.20743}
}
```

## Responsible use

This project studies gender bias to help **diagnose and mitigate** it. Gender is
operationalized as a binary purely for tractable, macroscopic measurement; this
is a modeling simplification, not a statement about gender identity. The prompt
sets and analysis are intended for bias auditing and research only.
