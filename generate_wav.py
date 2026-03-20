#!/usr/bin/env python3
"""
Unified TTS Generation Script for BindingBias
The Binding Effect: Multi-Dimensional Gender Bias in Instruction TTS

Supports 4 TTS models:
  - parler-large: Parler-TTS Large model
  - parler-mini: Parler-TTS Mini model  
  - promptttspp: PromptTTS++
  - voxinstruct: VoxInstruct

Usage:
  python generate_wav.py --model parler-large --json descriptions/descriptions_persona_bias.json --output results/parler_large/
  python generate_wav.py --model parler-mini --json descriptions/descriptions_multi_axis.json --output results/parler_mini/
  python generate_wav.py --model promptttspp --json descriptions/descriptions_two_axis.json --output results/promptttspp/
  python generate_wav.py --model voxinstruct --json descriptions/descriptions_sdo_bias.json --output results/voxinstruct/

Input JSON format:
  [
    {
      "id": "0001",
      "description": "Style/voice description...",
      "trait": "trait type",
      "keywords": "keywords",
      "prompt_text": "Text to synthesize"
    },
    ...
  ]

Output:
  - WAV files named by ID: 0001.wav, 0002.wav, etc.
"""

import os
import sys
import json
import argparse
import re
import torch
import numpy as np
import soundfile as sf
import soundfile as snd
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add model paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "models" / "parler-tts"))
sys.path.insert(0, str(SCRIPT_DIR / "models" / "promptttspp"))
sys.path.insert(0, str(SCRIPT_DIR / "models" / "VoxInstruct"))


# ============================================================
# Model Configurations
# ============================================================

MODEL_CONFIGS = {
    "parler-large": {
        "model_dir": "",
        "max_new_tokens": 8000,
        "temperature": 0.8,
    },
    "parler-mini": {
        "model_dir": "",
        "max_new_tokens": 2048,
        "temperature": 0.8,
    },
    "promptttspp": {
        "model_dir": "",
        "noise_scale": 0.5,
    },
    "voxinstruct": {
        "model_dir": "",
        "max_length": 1000,
        "temperature": 1.0,
    }
}


# ============================================================
# Helper Functions
# ============================================================

def _strip_control_chars(s: str) -> str:
    """Remove control characters from string"""
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", s)


def _is_valid_wav(path: str, min_seconds: float = 0.08) -> bool:
    """Check if WAV file exists and has valid audio"""
    if not os.path.isfile(path):
        return False
    try:
        info = snd.info(path)
        if info.samplerate is None or info.frames is None or info.frames <= 0:
            return False
        dur = info.frames / float(info.samplerate)
        return dur >= min_seconds
    except Exception:
        return False


# ============================================================
# Parler-TTS Generator
# ============================================================

class ParlerTTSGenerator:
    def __init__(self, model_name: str, output_dir: str, skip_existing: bool = True):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get config
        self.config = MODEL_CONFIGS[model_name]
        self.model_dir = self.config["model_dir"]
        
        print(f"[INFO] Loading {model_name} from {self.model_dir}")
        print(f"[INFO] Using device: {self.device}")
        
        # Load model
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device)
        
        try:
            self.model.set_default_attn_implementation("sdpa")
        except Exception:
            pass
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        
        # Resize embeddings if needed
        emb = self.model.get_input_embeddings()
        emb_vocab = int(emb.num_embeddings)
        tk_vocab = getattr(self.tokenizer, "vocab_size", None)
        if tk_vocab and tk_vocab > emb_vocab:
            print(f"[INFO] Resizing embeddings: {emb_vocab} -> {tk_vocab}")
            self.model.resize_token_embeddings(tk_vocab)
        
        print(f"[INFO] {model_name} loaded successfully")
    
    def safe_tokenize(self, text: str, is_description: bool = False):
        """Clean and tokenize text, clip out-of-range IDs"""
        text = (text or "").strip()
        text = _strip_control_chars(text)
        
        tok = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            return_attention_mask=True,
            truncation=True,
            max_length=getattr(self.tokenizer, "model_max_length", 4096),
        )
        
        ids = tok.input_ids
        attn = tok.attention_mask
        vocab_size = int(self.model.get_input_embeddings().num_embeddings)
        
        # Clip out-of-range tokens to UNK
        unk_id = getattr(self.tokenizer, "unk_token_id", 0)
        mask_bad = (ids < 0) | (ids >= vocab_size)
        ids = torch.where(mask_bad, torch.tensor(unk_id, dtype=ids.dtype), ids)
        
        return ids.to(self.device), attn.to(self.device)
    
    def generate_single(self, item: Dict) -> bool:
        """Generate a single WAV file from JSON item"""
        item_id = item.get("id", "unknown")
        output_path = self.output_dir / f"{item_id}.wav"
        
        # Skip if exists
        if self.skip_existing and _is_valid_wav(str(output_path)):
            return True
        
        description = item.get("description", "")
        prompt_text = item.get("prompt_text", "")
        
        if not description or not prompt_text:
            print(f"[WARN] Skipping {item_id}: missing description or prompt_text")
            return False
        
        try:
            # Tokenize
            desc_ids, desc_attn = self.safe_tokenize(description, is_description=True)
            prompt_ids, prompt_attn = self.safe_tokenize(prompt_text, is_description=False)
            
            # Generate
            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=desc_ids,
                    attention_mask=desc_attn,
                    prompt_input_ids=prompt_ids,
                    prompt_attention_mask=prompt_attn,
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"],
                    do_sample=True,
                )
            
            # Save - convert to float32 for soundfile compatibility
            audio_arr = generation.cpu().numpy().squeeze()
            if audio_arr.dtype == np.float16:
                audio_arr = audio_arr.astype(np.float32)
            sf.write(str(output_path), audio_arr, self.model.config.sampling_rate)
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to generate {item_id}: {e}")
            return False
    
    def batch_generate(self, data: List[Dict]):
        """Generate WAV files for all items in data"""
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for item in tqdm(data, desc=f"Generating with {self.model_name}"):
            item_id = item.get("id", "unknown")
            output_path = self.output_dir / f"{item_id}.wav"
            
            if self.skip_existing and _is_valid_wav(str(output_path)):
                skip_count += 1
                continue
            
            if self.generate_single(item):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n[SUMMARY]")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Failed:  {fail_count}")
        print(f"  Total:   {len(data)}")


# ============================================================
# PromptTTS++ Generator
# ============================================================

class PromptTTSPPGenerator:
    def __init__(self, model_name: str, output_dir: str, skip_existing: bool = True):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = MODEL_CONFIGS[model_name]
        model_dir = self.config['model_dir']
        
        print(f"[INFO] Loading {model_name} from {model_dir}")
        print(f"[INFO] Using device: {self.device}")
        
        # Import dependencies
        import hydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        import nltk
        from g2p_en import G2p
        import torchaudio
        
        # Add promptttspp to path
        promptttspp_dir = SCRIPT_DIR / "models" / "promptttspp"
        if str(promptttspp_dir) not in sys.path:
            sys.path.insert(0, str(promptttspp_dir))
        
        from promptttspp.text.eng import symbols, text_to_sequence
        from promptttspp.utils.model import lowpass_filter
        
        # Download NLTK data
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        
        # Initialize G2P and utilities
        self.g2p = G2p()
        self.symbols = symbols
        self.text_to_sequence = text_to_sequence
        self.lowpass_filter = lowpass_filter
        
        # Load configs using hydra - use initialize_config_dir for absolute paths
        from hydra import initialize_config_dir, compose
        config_dir = str(promptttspp_dir / "egs" / "proposed" / "bin" / "conf")
        initialize_config_dir(config_dir=config_dir, version_base=None)
        cfg = compose(config_name="demo")
        
        # Override checkpoint paths
        cfg.model_ckpt_path = os.path.join(model_dir, "checkpoint", "proposed", "last.ckpt")
        cfg.vocoder_ckpt_path = os.path.join(model_dir, "checkpoint", "bigvgan_f0_full", "last.ckpt")
        cfg.mel_stats_file = os.path.join(model_dir, "checkpoint", "stats.yaml")
        
        # Load model and vocoder using the helper function from original code
        self.model, self.vocoder = self._load_model(
            cfg.model, cfg.model_ckpt_path, 
            cfg.vocoder, cfg.vocoder_ckpt_path
        )
        
        # Load mel transform and stats
        self.to_mel = instantiate(cfg.transforms)
        self.sample_rate = self.to_mel.sample_rate
        self.mel_stats = OmegaConf.load(cfg.mel_stats_file)
        
        print(f"[INFO] {model_name} loaded successfully")
    
    def _load_model(self, model_cfg, model_ckpt_path, vocoder_cfg, vocoder_ckpt_path):
        """Load model and vocoder (from original main.py)"""
        from hydra.utils import instantiate
        
        model = instantiate(model_cfg)
        model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu")["model"])
        model = model.to(self.device).eval()
        
        vocoder = instantiate(vocoder_cfg)
        vocoder.load_state_dict(torch.load(vocoder_ckpt_path, map_location="cpu")["generator"])
        vocoder = vocoder.to(self.device).eval()
        
        return model, vocoder
    
    @torch.no_grad()
    def _synthesize_single(self, content_prompt: str, style_prompt: str):
        """Core synthesis function (from original main.py)"""
        # Convert text to phonemes
        phonemes = self.g2p(content_prompt)
        phonemes = [p if p not in [",", "."] else "sil" for p in phonemes]
        phonemes = [p for p in phonemes if p in self.symbols]
        phoneme_ids = self.text_to_sequence(" ".join(phonemes))
        phoneme_ids = torch.LongTensor(phoneme_ids)[None, :].to(self.device)
        
        # Generate mel-spectrogram with style prompt
        dec, log_cf0, vuv = self.model.infer(
            phoneme_ids,
            style_prompt=style_prompt,
            use_max=True,
            noise_scale=self.config.get('noise_scale', 0.5),
            return_f0=True,
        )
        
        # Post-process f0
        modfs = int(1.0 / (10 * 0.001))
        log_cf0 = self.lowpass_filter(log_cf0, modfs, cutoff=20)
        f0 = log_cf0.exp()
        f0[vuv < 0.5] = 0
        
        # Denormalize mel-spectrogram
        dec = dec * self.mel_stats["std"] + self.mel_stats["mean"]
        
        # Vocoder to waveform
        wav = self.vocoder(dec, f0).squeeze(1).cpu()
        return wav
    
    def generate_single(self, item: Dict) -> bool:
        """Generate a single WAV file from JSON item"""
        item_id = item.get("id", "unknown")
        output_path = self.output_dir / f"{item_id}.wav"
        
        # Skip if exists
        if self.skip_existing and _is_valid_wav(str(output_path)):
            return True
        
        description = item.get("description", "")
        prompt_text = item.get("prompt_text", "")
        
        if not description or not prompt_text:
            print(f"[WARN] Skipping {item_id}: missing description or prompt_text")
            return False
        
        try:
            # Synthesize
            wav = self._synthesize_single(prompt_text, description)
            
            # Save wav file
            import torchaudio
            torchaudio.save(str(output_path), wav, sample_rate=self.sample_rate)
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to generate {item_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def batch_generate(self, data: List[Dict]):
        """Generate WAV files for all items in data"""
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for item in tqdm(data, desc=f"Generating with {self.model_name}"):
            item_id = item.get("id", "unknown")
            output_path = self.output_dir / f"{item_id}.wav"
            
            if self.skip_existing and _is_valid_wav(str(output_path)):
                skip_count += 1
                continue
            
            if self.generate_single(item):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n[SUMMARY]")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Failed:  {fail_count}")
        print(f"  Total:   {len(data)}")


# ============================================================
# VoxInstruct Generator
# ============================================================

class VoxInstructGenerator:
    def __init__(self, model_name: str, output_dir: str, skip_existing: bool = True):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.config = MODEL_CONFIGS[model_name]
        model_dir = self.config['model_dir']
        
        print(f"[INFO] Loading {model_name} from {model_dir}")
        print(f"[INFO] Using device: {self.device}")
        
        # Import VoxInstruct dependencies
        from model.ar import VoxInstructAR
        from model.nar import VoxInstructNAR
        from utils.utils import get_config_from_file, sequence_mask, to_device, save_wav, top_k_top_p_filtering, convert_audio
        from utils.extract_hubert import HubertWithKmeans
        from transformers import AutoTokenizer
        from vocos import Vocos
        from encodec import EncodecModel
        import torchaudio
        
        # Save utility functions
        self.sequence_mask = sequence_mask
        self.to_device = to_device
        self.save_wav = save_wav
        self.top_k_top_p_filtering = top_k_top_p_filtering
        self.convert_audio = convert_audio
        
        # Load AR config and fix paths
        ar_config_path = os.path.join(SCRIPT_DIR, "models", "VoxInstruct", "configs", "train_ar.yaml")
        ar_hp = get_config_from_file(ar_config_path).hparams
        self.ar_hp = ar_hp
        
        # Fix relative paths to absolute paths
        checkpoint_dir = os.path.join(model_dir, "voxinstruct-sft-checkpoint")
        ar_hp.mt5_path = os.path.join(model_dir, "google-mt5-base-checkpoint")
        ar_hp.vocos_path = os.path.join(model_dir, "vocos-encodec-24khz")
        ar_hp.encodec_path = os.path.join(model_dir, "encodec-checkpoint")
        ar_hp.hubert_path = os.path.join(model_dir, "hubert-base-checkpoint")
        
        # Load AR model
        self.ar_model = VoxInstructAR(hp=ar_hp).to(self.device)
        ar_ckpt = torch.load(os.path.join(checkpoint_dir, "ar_1800k.pyt"), map_location=self.device)
        self.ar_model.load_state_dict(ar_ckpt['model'], strict=True)
        self.ar_model.to(torch.bfloat16).eval()
        
        # Load NAR config and fix paths
        nar_config_path = os.path.join(SCRIPT_DIR, "models", "VoxInstruct", "configs", "train_nar.yaml")
        nar_hp = get_config_from_file(nar_config_path).hparams
        self.nar_hp = nar_hp
        
        # Fix relative paths to absolute paths
        nar_hp.mt5_path = os.path.join(model_dir, "google-mt5-base-checkpoint")
        nar_hp.vocos_path = ar_hp.vocos_path
        
        # Load NAR model
        self.nar_model = VoxInstructNAR(hp=nar_hp).to(self.device)
        nar_ckpt = torch.load(os.path.join(checkpoint_dir, "nar_1800k.pyt"), map_location=self.device)
        self.nar_model.load_state_dict(nar_ckpt['model'], strict=True)
        self.nar_model.to(torch.bfloat16).eval()
        
        # Load Vocos vocoder
        vocos_path = ar_hp.vocos_path
        self.vocos = Vocos.from_hparams(f"{vocos_path}/config.yaml")
        state_dict = torch.load(f"{vocos_path}/pytorch_model.bin", map_location="cpu")
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in self.vocos.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
        self.vocos.load_state_dict(state_dict)
        self.vocos.to(self.device).eval()
        
        # Load Encodec - use default pretrained if local not available
        encodec_path = ar_hp.encodec_path
        if os.path.exists(encodec_path) and os.path.isdir(encodec_path):
            self.encodec = EncodecModel.encodec_model_24khz(pretrained=True, repository=Path(encodec_path)).to(self.device)
        else:
            print(f"[INFO] Local encodec not found, using default pretrained model")
            self.encodec = EncodecModel.encodec_model_24khz(pretrained=True).to(self.device)
        self.encodec.overlap = 0
        self.encodec.set_target_bandwidth(bandwidth=6.0)
        
        # Load HuBERT
        hubert_path = ar_hp.hubert_path
        self.hubert = HubertWithKmeans(
            checkpoint_path=f'{hubert_path}/hubert_base_ls960.pt',
            kmeans_path=f'{hubert_path}/hubert_base_ls960_L9_km500.bin',
            target_sample_hz=16000,
            seq_len_multiple_of=320
        )
        self.hubert.eval()
        self.hubert.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(ar_hp.mt5_path, local_files_only=True)
        
        # CFG parameters (from original inference)
        self.cfg_st_on_text = 1.5
        self.cfg_at_on_text = 3.0
        self.cfg_at_on_st = 1.5
        self.nar_iter_steps = 8
        
        print(f"[INFO] {model_name} loaded successfully")
    
    def generate_single(self, item: Dict) -> bool:
        """Generate a single WAV file from JSON item"""
        item_id = item.get("id", "unknown")
        output_path = self.output_dir / f"{item_id}.wav"
        
        # Skip if exists
        if self.skip_existing and _is_valid_wav(str(output_path)):
            return True
        
        description = item.get("description", "")
        prompt_text = item.get("prompt_text", "")
        
        if not description or not prompt_text:
            print(f"[WARN] Skipping {item_id}: missing description or prompt_text")
            return False
        
        try:
            # Prepare text instruction
            text = f"{description}. \"{prompt_text}\""
            text = text.strip().capitalize()
            
            # Tokenize text
            text_id = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()
            actual_len = text_id.shape[0]
            if actual_len >= self.ar_hp.max_text_len:
                text_id = text_id[:self.ar_hp.max_text_len]
                text_id[self.ar_hp.max_text_len - 1] = 1  # <eos> for mt5
                actual_len = self.ar_hp.max_text_len
            
            # Pad text_id to max_text_len
            import torch.nn.functional as F
            text_id = F.pad(text_id, (0, self.ar_hp.max_text_len - actual_len), value=0)
            
            text_ids = text_id.unsqueeze(0).to(self.device)
            text_id_lens = torch.tensor([actual_len], device=self.device)
            text_attn_mask = self.sequence_mask(text_id_lens, max_len=self.ar_hp.max_text_len, device=self.device)
            
            # Initialize sequences (no audio prompt, lang_id=1 for English)
            lang_id = 1 + 1  # offset
            seqs = torch.tensor([[self.ar_hp.bos_id, lang_id]], device=self.device)
            segment_ids = torch.tensor([[1, 1]], device=self.device)
            
            # AR inference
            pred_st_flag = True
            past_key_values_base = None
            text_encode = None
            free_text_encode = torch.zeros([1, self.ar_hp.max_text_len, self.ar_hp.hidden_dim], device=self.device).to(torch.bfloat16)
            
            for j in range(self.config['max_length']):
                ar_outputs_base, text_encode = self.ar_model.predict(
                    input_ids=seqs,
                    segment_ids=segment_ids,
                    text_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                    past_key_values=past_key_values_base,
                    text_encode=text_encode,
                )
                cond_logits = ar_outputs_base['logits']
                past_key_values_base = ar_outputs_base['past_key_values']
                
                if pred_st_flag:
                    logits = cond_logits
                    logits[:, :, self.ar_hp.bos_id] = -1e5
                    logits[:, :, self.ar_hp.st_token_num + 1:self.ar_hp.eos_id] = -1e5
                    logits[:, :, self.ar_hp.eos_id + 1] = -1e5
                    filtered_logits = self.top_k_top_p_filtering(logits[0, -1, :], top_k=5, top_p=0.95, temperature=self.config.get('temperature', 0.8))
                else:
                    logits = cond_logits
                    logits[:, :, self.ar_hp.bos_id] = -1e5
                    logits[:, :, 0:self.ar_hp.st_token_num + 1] = -1e5
                    logits[:, :, self.ar_hp.eos_id] = -1e5
                    filtered_logits = self.top_k_top_p_filtering(logits[0, -1, :], top_k=50, top_p=0.95, temperature=self.config.get('temperature', 0.8))
                
                probs = filtered_logits.softmax(dim=-1)
                samples = torch.multinomial(probs, 1).unsqueeze(1).to(self.device)
                
                seqs = torch.cat([seqs, samples], dim=1)
                
                if pred_st_flag:
                    segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 1], dim=1)
                else:
                    segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 2], dim=1)
                
                # Switch from ST to AT
                if samples.item() == self.ar_hp.eos_id:
                    pred_st_flag = False
                    st_len = (segment_ids == 1).sum(dim=1)
                    past_key_values_base = None
                
                if samples.item() == self.ar_hp.eos_id + 1:
                    break
                elif j == self.config['max_length'] - 1:
                    samples[:, :] = self.ar_hp.eos_id + 1
                    seqs = torch.cat([seqs, samples], dim=1)
                    segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 2], dim=1)
                    break
            
            # NAR inference
            b, t = seqs.shape
            full_seqs = torch.stack([seqs] * self.ar_hp.at_res_num, dim=1)
            layer_index = torch.ones(size=[b,], device=self.device)
            
            for layer_idx in range(1, self.ar_hp.at_res_num):
                full_seqs = self.nar_model.predict(
                    full_seqs,
                    0,  # at_prompt_len (no audio prompt)
                    segment_ids,
                    text_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                    layer_index=layer_idx,
                    iter_step=self.nar_iter_steps,
                )
            
            # Extract acoustic tokens
            st_len_val = st_len.item()
            full_seq = full_seqs[:, :, st_len_val:]
            full_seq = (full_seq - self.ar_hp.st_token_num - self.ar_hp.lang_num - 1)[:, :, :-1].clamp(0, 1023)
            
            # Decode with Vocos
            features = self.vocos.codes_to_features(full_seq[0])
            bandwidth_id = torch.tensor([2], device=self.device)
            wav = self.vocos.decode(features, bandwidth_id=bandwidth_id)
            wav = wav.cpu().squeeze().numpy()
            
            # Save
            import torchaudio
            torchaudio.save(str(output_path), torch.from_numpy(wav).unsqueeze(0), sample_rate=24000)
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to generate {item_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def batch_generate(self, data: List[Dict]):
        """Generate WAV files for all items in data"""
        success_count = 0
        skip_count = 0
        fail_count = 0
        
        for item in tqdm(data, desc=f"Generating with {self.model_name}"):
            item_id = item.get("id", "unknown")
            output_path = self.output_dir / f"{item_id}.wav"
            
            if self.skip_existing and _is_valid_wav(str(output_path)):
                skip_count += 1
                continue
            
            if self.generate_single(item):
                success_count += 1
            else:
                fail_count += 1
        
        print(f"\n[SUMMARY]")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Failed:  {fail_count}")
        print(f"  Total:   {len(data)}")


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified TTS Generation Script for CoP_bias',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parler-TTS Large
  python generate_wav.py --model parler-large --json descriptions/descriptions_sdo_bias.json --output 
  
  # Parler-TTS Mini
  python generate_wav.py --model parler-mini --json descriptions/descriptions_sdo_bias.json  --output 
  
  # PromptTTS++
  python generate_wav.py --model promptttspp --json descriptions/descriptions_sdo_bias.json --output
  
  # VoxInstruct
  python generate_wav.py --model voxinstruct --json descriptions/descriptions_sdo_bias.json --output
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       choices=['parler-large', 'parler-mini', 'promptttspp', 'voxinstruct'],
                       help='TTS model to use')
    parser.add_argument('--json', type=str, required=True,
                       help='Input JSON file with generation data')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for generated WAV files')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip existing WAV files (default: True)')
    parser.add_argument('--no-skip', action='store_false', dest='skip_existing',
                       help='Regenerate all files even if they exist')
    
    args = parser.parse_args()
    
    # Load JSON data
    print(f"[INFO] Loading data from {args.json}")
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"[ERROR] JSON must contain a list of items")
        sys.exit(1)
    
    print(f"[INFO] Loaded {len(data)} items")
    
    # Create generator
    if args.model in ['parler-large', 'parler-mini']:
        generator = ParlerTTSGenerator(args.model, args.output, args.skip_existing)
    elif args.model == 'promptttspp':
        generator = PromptTTSPPGenerator(args.model, args.output, args.skip_existing)
    elif args.model == 'voxinstruct':
        generator = VoxInstructGenerator(args.model, args.output, args.skip_existing)
    else:
        print(f"[ERROR] Unknown model: {args.model}")
        sys.exit(1)
    
    # Generate
    generator.batch_generate(data)
    
    print(f"\n[INFO] Generation complete!")
    print(f"[INFO] Output directory: {args.output}")


if __name__ == "__main__":
    main()
