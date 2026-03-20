# BindingBias: The Binding Effect in Instruction TTS

An experimental platform for studying how multi-dimensional cues form gender bias in instruction-based TTS models, integrating multiple advanced TTS models and bias analysis tools.

**Project Name**: BindingBias (The Binding Effect: Multi-Dimensional Gender Bias in Instruction TTS)
**Paper**: *The Binding Effect: Analysis of How Multi-Dimensional Cues Form Gender Bias in Instruction TTS*

## 📌 Quick Start

```bash
# 1. Check installation status
python test_installation.py

# 2. Generate speech
python generate_wav.py --model parler-large --json descriptions/descriptions_persona_bias.json --output results/test/

# 3. Analyze gender
python analyze_gender.py --wav_path results/test/ --json descriptions/descriptions_persona_bias.json --output analysis/test/
```

**Detailed Guide**: See [USAGE.md](USAGE.md) | **Project Summary**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## Environment Setup

### Quick Start

```bash
# 1. Create and activate environment
conda create -n BindingBias python=3.9.18 -y
conda activate BindingBias

# 2. Set CUDA environment variables (see setup.txt for details)


# 3. Install PyTorch and dependencies
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip==24.0
pip install -r requirements.txt

# 4. Install special packages
pip install git+https://github.com/facebookresearch/fairseq.git
pip install git+https://github.com/huggingface/parler-tts.git
pip install encodec
pip install 'transformers<4.46'
```

**For detailed setup instructions, see**: [setup.txt](setup.txt)

## Directory Structure

```
ITTS/  (BindingBias)
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.txt                         # Environment setup guide
│
├── generate_wav.py                   # Unified WAV generation script ⭐
├── analyze_gender.py                 # Gender analysis script ⭐
├── utils.py                          # Utility functions
│
├── descriptions/                     # JSON description files (with IDs)
│   ├── description_career_bias*.json         # Career bias data 
│   ├── descriptions_multi_axis.json          # Multi-axis bias 
│   ├── descriptions_persona_bias.json        # Personality trait bias 
│   ├── descriptions_status_bias.json            # Social Status bias
│   ├── descriptions_two_axis.json            # Two-axis bias 
│
├── models/                           # TTS model implementations
│   ├── parler-tts/                  # Parler-TTS (large & mini)
│   │   ├── run_exp.py              # Large model generation script
│   │   └── run_exp_mini.py         # Mini model generation script
│   ├── promptttspp/                 # PromptTTS++
│   │   └── gen.py                  # Generation script
│   └── VoxInstruct/                 # VoxInstruct
│       └── inference.py            # Inference script
│
├── experiments/                      # Experimental tools
│   └── gender_detect/               # Gender detection model
│       └── (gender detection models)
│
└── results/                         # Output results
    ├── parler_large/               # Parler Large WAV files
    ├── parler_mini/                # Parler Mini WAV files
    ├── promptttspp/                # PromptTTS++ WAV files
    ├── voxinstruct/                # VoxInstruct WAV files
    └── analysis/                   # Analysis CSV results
```

## Project Description

### TTS Models

#### 1. Parler-TTS
- **Path**: `models/parler-tts/`
- **Description**: Multilingual TTS model based on natural language descriptions
- **Features**: Supports multiple voice styles and languages
- **Usage**: Generates natural speech from detailed style descriptions

#### 2. PromptTTS++
-  **Path**: `models/promptttspp/`
- **Description**: Advanced prompt-based TTS system
- **Features**: Supports multi-dimensional control (emotion, age, gender, etc.)
- **Usage**: Precise control of voice style and features using prompts

#### 3. VoxInstruct
- **Path**: `models/voxinstruct/`
- **Description**: Instruction-driven speech synthesis model (based on fairseq)
- **Features**: Controls synthesis through natural language instructions
- **Usage**: Natural language instruction-based speech generation

### Bias Research Datasets

All JSON files contain the following standard fields:
- `id`: Unique identifier (format: 0001, 0002, ...)
- `description`: TTS prompt description
- `trait`: Personality trait or bias type
- `keywords`: Related keywords
- `prompt_text`: Text to synthesize

### Experimental Tools

#### Gender Detection & Bias Analysis
- **Path**: `experiments/gender_detect/`
- **Features**:
  - Analyze gender bias in TTS models
  - Multi-axis analysis (gender, career, personality traits, etc.)
  - Bias quantification and statistical analysis
  - Bias mitigation method research

### Utility Scripts

#### 1. generate_wav.py ⭐ (Main Generation Script)

Unified TTS speech generation script supporting 4 models:

**Supported Models:**
- `parler-large`: Parler-TTS Large model
- `parler-mini`: Parler-TTS Mini model
- `promptttspp`: PromptTTS++ (refer to models/promptttspp/gen.py)
- `voxinstruct`: VoxInstruct (refer to models/VoxInstruct/inference.py)

**Usage:**
```bash
# Parler-TTS Large
python generate_wav.py \
    --model parler-large \
    --json descriptions/descriptions_persona_bias.json \
    --output results/parler_large/

# Parler-TTS Mini
python generate_wav.py \
    --model parler-mini \
    --json descriptions/descriptions_multi_axis.json \
    --output results/parler_mini/

# VoxInstruct (use original script)
cd models/VoxInstruct
python inference.py
```

**Parameters:**
- `--model`: Model name (parler-large, parler-mini, promptttspp, voxinstruct)
- `--json`: Input JSON file path
- `--output`: Output WAV file directory
- `--skip-existing`: Skip existing files (default)
- `--no-skip`: Regenerate all files

**Output Format:**
- WAV files named by ID: 0001.wav, 0002.wav, 0003.wav, ...

#### 2. analyze_gender.py ⭐ (Gender Analysis Script)

Analyzes generated WAV files, detects gender, and computes statistics.
Calls the gender detection model from `experiments/gender_detect/`.

**Usage:**
```bash
# Analyze WAV files only
python analyze_gender.py \
    --wav_path results/parler_large/ \
    --output analysis/parler_large.csv

# Analyze with JSON metadata
python analyze_gender.py \
    --wav_path results/parler_mini/ \
    --json descriptions/descriptions_persona_bias.json \
    --output analysis/parler_mini/

# Generate multiple statistics CSVs
python analyze_gender.py \
    --wav_path results/voxinstruct/ \
    --json descriptions/descriptions_multi_axis.json \
    --output analysis/voxinstruct/
```

**Parameters:**
- `--wav_path`: WAV file directory
- `--json`: Original JSON file (optional, for matching trait and keywords)
- `--output`: Output CSV file or directory

**Output Files:**

If output is a single CSV file:
- `detection_results.csv`: Complete detection results

If output is a directory, generates:
- `detection_results.csv`: Complete detection results (id, wav_file, predicted_gender, male_score, female_score, trait, keywords)
- `overall_gender_distribution.csv`: Overall gender distribution
- `gender_by_trait.csv`: Gender distribution by trait and female/male ratio
- `gender_by_keyword.csv`: Gender distribution by keyword and female/male ratio

**統計指標：**
- Female/Male count and percentage
- Female/Male ratio per trait
- Female/Male ratio per keyword


## Dependencies

### Core Framework
- Python 3.9.18
- PyTorch 2.3.0 (CUDA 11.8)
- transformers <4.46
- fairseq (from GitHub)

### Audio Processing
- soundfile, pysptk, pyworld, vocos
- audb, audinterface, audmetric, audonnx, audplot
- nnmnkwii, faster_whisper

### Machine Learning
- numpy==1.26.4, scipy, pandas, scikit-learn
- hydra-core, omegaconf, peft
- tensorboard

### Others
- parler-tts (from GitHub)
- encodec, sentencepiece, protobuf
- gradio, notebook

For detailed list, see [requirements.txt](requirements.txt)