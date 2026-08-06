<div align="center">

# The Binding Effect: Multi-Dimensional Gender Bias in Instruction TTS

Official code and available prompt sets for **"The Binding Effect: Analysis of How Multi-Dimensional Cues Form Gender Bias in Instruction TTS"** (Interspeech 2026).

**Authors:** Kuan-Yu Chen, Yi-Cheng Lin, Po-Chung Hsieh, Huang-Cheng Chou, Chih-Fan Hsu, Jeng-Lin Li, Hung-yi Lee, and Jian-Jiun Ding

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2603.20743-b31b1b.svg)](https://arxiv.org/abs/2603.20743)
[![Interspeech 2026](https://img.shields.io/badge/Interspeech-2026-blue.svg)](https://arxiv.org/abs/2603.20743)

</div>

---

## 📌 Overview

Bias evaluations in Instruction TTS (ITTS) usually test one attribute at a time, although real prompts combine social cues. This project studies how three theoretically grounded axes interact to shape the perceived gender of synthesized speech:

* **Social Status**: Weberian stratification represented with high/low Social Dominance Orientation descriptors.
* **Career**: Socially structured occupational roles with female-, mixed-, or male-leaning priors.
* **Persona**: Big Five traits: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism.

> [!NOTE]
> **The Binding Effect:**  
> The **Binding Effect** is the non-additive interaction of these cues. The paper-primary probability is `female / (female + male + child)`, matching the event `D(y) = Female` over successful classifier outcomes. Unknown/failed outcomes have no `D(y)` result and are excluded but reported. Adult-only `female / (female + male)` is also reported separately.

---

## 💡 Method

The paper uses two stages:
1. **Stage 1** estimates the empirical female probability for each isolated descriptor.
2. **Stage 2** evaluates bi- and tri-dimensional combinations against those baselines.

For female probability `P(x)`, `L(x) = ln(P(x) / (1 - P(x)))`. Pairwise interaction is `I(w1,w2) = L(x_bi) - L(x_uni1) - L(x_uni2)`. Three-way interaction subtracts all univariate and pairwise effects from `L(x_multi)` as in paper equations 5 and 6. At an all-female or all-non-female boundary, the implementation applies the binomial Haldane-Anscombe correction `(female + 0.5) / (non-female + 0.5)`; interior proportions are unchanged.

The paper states 10,000 iterations with random-label shuffling but does not publish the exact algorithm. A pooled shuffle tests equality of condition probabilities, not the required additive-logit null `I = 0`. `analyze_interactions.py` therefore fits independent binomial condition probabilities constrained to `I = 0`, then performs 10,000 seeded random binary-label draws at the original condition sizes. It uses a two-sided statistic and the `(extreme + 1) / (iterations + 1)` Monte Carlo correction. This choice is explicitly a constrained-null randomization approximation to the unavailable paper implementation. Reports retain the paper's moderate (`p < 0.05`, `|I| > 1.0`) and strong (`p < 0.01`, `|I| > 2.8`) categories.

Optional `semantic_bias.py` implements equation 7, the female-minus-male anchor cosine bias `Delta`, and configured group comparisons with Cohen's d. It fails with an installation instruction when `sentence-transformers` is unavailable.

---

## 📊 Repository Status

> [!IMPORTANT]
> **Prompt Availability & Reproduction Gap:**  
> The generation, classifier, interaction-statistics, semantic-analysis, configuration preflight, and prompt-audit paths are runnable. Model weights, external TTS implementations, generated audio, paper outputs, and the full human-verified prompt collection are not bundled.
> 
> The paper specifies 13,300 prompts: 6,900 univariate and 6,400 compositional. This repository contains 5,900: 3,900 univariate and 2,000 compositional. The absent 7,400 human-verified prompts are not reconstructed or fabricated, so exact paper reproduction is currently unavailable. `prompt_manifest.json` records exact per-file count mismatches, unexpected JSON files, hashes, and the shortfall. Because no authoritative paper prompt hashes were published, count equality alone would not assert content identity.
> 
> The old `*_experiment.json` files referenced by the original experiment scripts are absent. Those scripts now state this and delegate to the actual files in `descriptions/` rather than silently skipping nonexistent inputs.

---

## 📁 Structure

```text
.
├── generate_wav.py             # Unified generation and model preflight
├── analyze_gender.py            # wav2vec 2.0 classifier and descriptive summaries
├── analyze_interactions.py      # Equations 5/6 and permutation significance
├── binding_stats.py            # Dependency-free statistical core
├── semantic_bias.py            # Optional embedding Delta and Cohen's d
├── prompt_audit.py             # Prompt schema/count/hash audit
├── prompt_manifest.json        # Audit of the currently distributed prompts
├── model_config.example.json    # External model/source configuration example
│
├── descriptions/               # Available status, career, persona, bi-, and tri-axis JSON
├── run_all_models.sh            # Generate every available JSON with all four models
├── analyze_all_models.sh        # Analyze every configured model and available JSON
└── tests/                      # Offline synthetic statistics and schema tests

```

Each prompt item has `id`, `description`, `trait`, `keywords`, and gender-neutral `prompt_text` fields.

---

## 🛠️ Setup

See [`setup.txt`](https://www.google.com/search?q=setup.txt). A minimal analysis setup is:

```bash
conda create -n BindingBias python=3.9 -y
conda activate BindingBias
pip install torch==2.3.0 torchaudio==2.3.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt

```

> [!TIP]
> Heavy TTS imports are lazy, so `--help`, prompt audit, statistics tests, and generation `--check` do not load model code.

---

## 📦 External Assets

* **Parler-TTS:** Defaults to the public IDs `parler-tts/parler-tts-mini-v1` and `parler-tts/parler-tts-large-v1`. Install [Parler-TTS](https://github.com/huggingface/parler-tts); downloads follow Hugging Face cache settings.
* **PromptTTS++ & VoxInstruct:** Require separately obtained official source checkouts and checkpoint assets. They are supported through `--backend-path` plus `--model-id`, environment variables shown by preflight errors, or [`model_config.example.json`](https://www.google.com/search?q=model_config.example.json). No weights are redistributed here.
* **Classifier:** The acoustic classifier is AudEERING's [wav2vec2 age/gender model](https://zenodo.org/record/7761387) and downloads on first real analysis run.

---

## 🚀 Usage

### 1. Validate Preflight

Validate without loading or downloading a model:

```bash
python generate_wav.py --model parler-mini --check
python generate_wav.py --model promptttspp --config model_config.json --check

```

### 2. Generate and Classify Speech

```bash
python generate_wav.py --model parler-mini \
  --json descriptions/descriptions_persona_bias.json --output results/parler-mini/persona
python analyze_gender.py --wav_path results/parler-mini/persona \
  --json descriptions/descriptions_persona_bias.json --output analysis/parler-mini/persona

```

Outputs include `detection_results.csv`, overall distribution, and trait/keyword summaries. Existing valid WAVs are skipped unless `--no-skip` is passed.

### 3. Analyze Interactions

For interactions, provide a JSON spec whose order-2 entries contain exactly three conditions in `[joint, uni1, uni2]` order, or whose order-3 entries contain seven in `[triple, pair12, pair13, pair23, uni1, uni2, uni3]` order. Every condition has `name` and a `csv` path to classifier output:

```bash
python analyze_interactions.py --spec interactions.json --output analysis/interactions.csv
python semantic_bias.py --spec semantic.json --output analysis/semantic.json

```

### 4. Batch Execution

Batch generation and analysis use all four configured models:

```bash
CONFIG_PATH=./model_config.json OUTPUT_BASE=./ITTS_audios bash run_all_models.sh
WAV_BASE=./ITTS_audios ANALYSIS_BASE=./analysis bash analyze_all_models.sh

```

Generation exits nonzero if any item fails. Batch scripts continue to report all incomplete model/dataset combinations, then exit nonzero if any generation, missing WAV set, or analysis failed.

---

## 🔬 Reproducibility and Paper Alignment

Run the prompt audit and offline tests with no model or network:

```bash
python prompt_audit.py --output prompt_manifest.json
python prompt_audit.py --strict-paper  # intentionally fails while 7,400 prompts are absent
python -m unittest discover -v

```

The implementation aligns with paper equations 5-7, uses 10,000 constrained-null randomizations as documented above, validates complete named interaction condition sets, includes `child` as a non-female paper-primary classifier outcome, and reports adult-only and unknown outcomes separately. Reproducing paper tables additionally requires the missing human-verified prompts, original model/checkpoint versions, random samples, generated waveforms, classifier audit data, and the authors' exact permutation implementation; these are unresolved external blockers.

---

## 📖 Citation

```bibtex
@inproceedings{chen2026binding,
  title     = {The Binding Effect: Analysis of How Multi-Dimensional Cues Form Gender Bias in Instruction TTS},
  author    = {Chen, Kuan-Yu and Lin, Yi-Cheng and Hsieh, Po-Chung and Chou, Huang-Cheng and Hsu, Chih-Fan and Li, Jeng-Lin and Lee, Hung-yi and Ding, Jian-Jiun},
  booktitle = {Interspeech},
  year      = {2026},
  note      = {arXiv:2603.20743}
}

```

---

## 📜 Licenses and Acknowledgements

The arXiv paper is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). No source-code license file is currently included, so reuse of this repository's code is not granted beyond applicable law until the authors add one. External models, checkpoints, datasets, and backends retain their own licenses and terms.

This work is intended for bias diagnosis and mitigation research. Binary acoustic classification is a measurement simplification, not a statement about gender identity. Acknowledgements and funding details are provided in the paper.
