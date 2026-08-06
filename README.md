<div align="center">

# The Binding Effect: Multi-Dimensional Gender Bias in Instruction TTS

Official code and prompt sets for **"The Binding Effect: Analysis of How Multi-Dimensional Cues Form Gender Bias in Instruction TTS"** (Interspeech 2026).

**Kuan-Yu Chen, Yi-Cheng Lin, Po-Chung Hsieh, Huang-Cheng Chou,**  
**Chih-Fan Hsu, Jeng-Lin Li, Hung-yi Lee, Jian-Jiun Ding**  
*National Taiwan University · AI Research Center, Inventec Corporation*

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2603.20743-b31b1b.svg)](https://arxiv.org/abs/2603.20743)
[![Interspeech 2026](https://img.shields.io/badge/Interspeech-2026-blue.svg)](https://arxiv.org/abs/2603.20743)

</div>

---

## 📌 Overview

Bias evaluations in Instruction TTS (ITTS) usually test **one attribute at a time**, ignoring that real prompts combine many social cues. We model a prompt as a composition of three theoretically grounded axes and study how they interact to shape the perceived **gender** of synthesized speech:

* **Social Status** — Weberian stratification, realized via Social Dominance Orientation (SDO) descriptors (*high / low status*).
* **Career** — Socially structured occupational roles (*female- / mixed- / male-leaning*).
* **Persona** — Big Five dispositional traits (*Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism*).

> [!NOTE]
> **The Binding Effect:**  
> The phenomenon where multi-dimensional social cues interact **non-additively**. For example, combining a female-leaning occupation with high status and a male-leaning persona can flip the perceived gender toward male. Our framework quantifies this with an **interaction term** measured in log-odds space.

### ❓ Research Questions
* **RQ1:** How do compositional interactions among Social Status, Career, and Persona modulate latent gender associations vs. univariate baselines?
* **RQ2:** Which dimensions dominate acoustic gender realization, and do some attributes systematically override others?
* **RQ3:** Are the observed patterns consistent with the semantic priors of pretrained text encoders, training-data distributions, or both?

---

## 🤖 Evaluated ITTS Models

The framework evaluates four representative ITTS backends:

| Alias | Model | Backbone Architecture | Text Encoder |
|-------|-------|----------------------|--------------|
| `voxinstruct` | **VoxInstruct** | LLaMA (AR + NAR) | `mT5-base` |
| `promptttspp` | **PromptTTS++** | Diffusion + MDN | `BERT` |
| `parler-mini` | **Parler-TTS Mini** | AudioLM | `Flan-T5-large` |
| `parler-large` | **Parler-TTS Large** | AudioLM | `Flan-T5-large` |

---

## 📁 Repository Layout

```text
.
├── generate_wav.py                # Unified TTS generation for the 4 model backends
├── analyze_gender.py               # Gender detection + bias statistics over generated WAVs
│
├── descriptions/                  # Prompt sets (JSON) for each axis / composition
│   ├── descriptions_status_bias.json     # Social Status (SDO)
│   ├── description_career_bias.json      # Career / occupation
│   ├── descriptions_persona_bias.json     # Persona (Big Five)
│   ├── descriptions_two_axis.json        # Bi-dimensional compositions
│   └── descriptions_multi_axis.json      # Tri-dimensional compositions
│
├── run_all_models.sh              # Batch generation runner
├── run_experiments.sh             # Per-model experiment generator runner
├── analyze_all_models.sh          # Batch analysis runner
├── analyze_experiments.sh         # Per-model experiment analysis runner
│
├── requirements.txt               # Python dependencies
└── setup.txt                      # Step-by-step environment setup guide

```

> [!TIP]
> **Not Included (by design):** TTS model checkpoints (`models/`) and generated audio/analysis outputs are excluded via `.gitignore` due to size. TTS backends are installed separately; the `wav2vec 2.0` gender detector is auto-downloaded on first run.

### 📄 Prompt (JSON) Format Example

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

---

## 🛠️ Installation

Refer to [setup.txt](https://www.google.com/search?q=setup.txt) for a complete walkthrough. Quick setup:

```bash
# 1. Create and activate Conda environment
conda create -n BindingBias python=3.9.18 -y
conda activate BindingBias

# 2. Install PyTorch with CUDA 11.8 support
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install --upgrade pip==24.0

# 3. Install core dependencies
pip install -r requirements.txt
pip install 'transformers<4.46'

```

### 🧩 TTS Backend Environments

Each TTS backend may require separate environments or packages:

* **Parler-TTS:** `pip install git+https://github.com/huggingface/parler-tts.git`
* **PromptTTS++:** Install from official repository (provides `promptttspp` package).
* **VoxInstruct:** Install from official repository (provides `utils.utils`, `utils.extract_hubert`, and `fairseq`). Typically runs in a dedicated `voxinstruct` conda env.

---

## 🚀 Usage

### 1. Generate Speech

```bash
python generate_wav.py \
    --model parler-large \
    --json descriptions/descriptions_persona_bias.json \
    --output results/parler_large/

```

* `--model` choices: `parler-large`, `parler-mini`, `promptttspp`, `voxinstruct`.
* Output WAVs are named by entry ID (`0001.wav`, `0002.wav`, ...).
* Use `--no-skip` to force regeneration of existing files.

### 2. Analyze Gender & Bias

```bash
python analyze_gender.py \
    --wav_path results/parler_large/ \
    --json descriptions/descriptions_persona_bias.json \
    --output analysis/parler_large/

```

When `--output` is a directory, the script generates:

* `detection_results.csv` — Per-utterance predictions (*id, wav, predicted gender, male/female scores, trait, keywords*).
* `overall_gender_distribution.csv` — Overall female/male ratio.
* `gender_by_trait.csv` — Female/male ratio per trait.
* `gender_by_keyword.csv` — Female/male ratio per keyword.

### ⚡ Batch Execution

```bash
# Override paths via environment variables if needed
OUTPUT_BASE=./ITTS_audios bash run_all_models.sh      # Generate speech for all models
WAV_BASE=./ITTS_audios    bash analyze_all_models.sh  # Analyze bias across all models

```

---

## ⚖️ Responsible Use

> [!IMPORTANT]
> This project studies gender bias to help **diagnose and mitigate** structural bias in generative audio models.
> Gender is operationalized as a binary categorization strictly for tractable, macroscopic measurement. This is a technical modeling simplification, not a statement on gender identity. All prompt sets and analysis tools are intended solely for bias auditing and academic research.

---

## 📖 Citation

If you find this research or code useful, please cite our paper:

```bibtex
@inproceedings{chen2026binding,
  title     = {The Binding Effect: Analysis of How Multi-Dimensional Cues Form Gender Bias in Instruction TTS},
  author    = {Chen, Kuan-Yu and Lin, Yi-Cheng and Hsieh, Po-Chung and Chou, Huang-Cheng and Hsu, Chih-Fan and Li, Jeng-Lin and Lee, Hung-yi and Ding, Jian-Jiun},
  booktitle = {Interspeech},
  year      = {2026},
  note      = {arXiv:2603.20743}
}

```
