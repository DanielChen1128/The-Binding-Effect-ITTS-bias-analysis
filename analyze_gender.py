#!/usr/bin/env python3
"""
Gender Analysis Script for BindingBias
The Binding Effect: Multi-Dimensional Gender Bias in Instruction TTS

Analyzes WAV files to detect gender and compute statistics by trait and keywords.
Uses the gender detection model from experiments/gender_detect/

Usage:
  python analyze_gender.py --wav_path results/parler_large/ --output analysis/parler_large_analysis.csv
  python analyze_gender.py --wav_path results/parler_mini/ --json descriptions/descriptions_persona_bias.json --output analysis/

Input:
  - wav_path: Directory containing WAV files named by ID (0001.wav, 0002.wav, ...)
  - json (optional): Original JSON file to match with metadata (trait, keywords)
  - output: Output CSV file path or directory

Output CSV columns:
  - id: File ID
  - wav_file: WAV filename
  - predicted_gender: Detected gender (male/female)
  - male_score: Confidence score for male
  - female_score: Confidence score for female
  - trait: Trait from JSON (if provided)
  - keywords: Keywords from JSON (if provided)
  
Summary statistics:
  - Overall gender distribution (count, percentage)
  - Gender distribution per trait
  - Gender distribution per keyword
  - Female/Male ratio per trait
  - Female/Male ratio per keyword
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Add gender detection path
SCRIPT_DIR = Path(__file__).parent
GENDER_DETECT_DIR = SCRIPT_DIR / "experiments" / "gender_detect"
sys.path.insert(0, str(GENDER_DETECT_DIR))


# ============================================================
# Gender Detection using audonnx
# ============================================================

class GenderDetector:
    def __init__(self):
        """Initialize gender detection model"""
        try:
            import audonnx
            import audeer
            import torch
            
            print("[INFO] Loading gender detection model...")
            
            # Download and cache model
            model_url = 'https://zenodo.org/record/7761387/files/w2v2-L-robust-6-age-gender.25c844af-1.1.1.zip'
            model_dir = SCRIPT_DIR / 'models' / 'gender_detector'
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_file = model_dir / 'model.onnx'
            
            # Download if not exists
            if not model_file.exists():
                print("[INFO] Downloading gender detection model (first time only)...")
                cache_dir = model_dir / 'cache'
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                archive_path = audeer.download_url(model_url, cache_dir, verbose=True)
                audeer.extract_archive(archive_path, model_dir)
                print("[INFO] Model downloaded and extracted")
            
            # Load model
            self.model = audonnx.load(str(model_dir))
            self.sampling_rate = 16000
            
            print("[INFO] Gender detection model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load gender detection model: {e}")
            print("[INFO] Please install: pip install audonnx audeer torch")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def detect_gender(self, wav_path: str) -> Tuple[str, float, float]:
        """
        Detect gender from WAV file
        
        Returns:
            (predicted_gender, male_score, female_score)
        """
        try:
            import soundfile as sf
            import torch
            import torch.nn.functional as F
            
            # Read audio file
            signal, sr = sf.read(wav_path)
            
            # Resample if needed
            if sr != self.sampling_rate:
                import librosa
                signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sampling_rate)
            
            # Convert to float32 mono
            if signal.ndim > 1:
                signal = signal.mean(axis=1)
            signal = signal.astype(np.float32)
            
            # Get model output
            output = self.model(signal, self.sampling_rate)
            
            # Extract gender logits: [female, male, child]
            gender_logits = output['logits_gender'][0]
            
            # Convert to probabilities
            probs = F.softmax(torch.tensor(gender_logits), dim=-1).numpy()
            
            # Extract scores (index 0=female, 1=male, 2=child)
            female_score = float(probs[0])
            male_score = float(probs[1])
            # child_score = float(probs[2])  # Not used for now
            
            # Predict gender based on higher score
            predicted_gender = 'female' if female_score > male_score else 'male'
            
            return predicted_gender, male_score, female_score
            
        except Exception as e:
            print(f"[WARN] Failed to detect gender for {wav_path}: {e}")
            return 'unknown', 0.0, 0.0
    
    def batch_detect(self, wav_files: List[str]) -> List[Dict]:
        """
        Detect gender for multiple WAV files
        
        Returns:
            List of dicts with: {id, wav_file, predicted_gender, male_score, female_score}
        """
        results = []
        
        for wav_file in tqdm(wav_files, desc="Detecting gender"):
            file_id = Path(wav_file).stem  # Get ID from filename
            
            predicted_gender, male_score, female_score = self.detect_gender(wav_file)
            
            results.append({
                'id': file_id,
                'wav_file': os.path.basename(wav_file),
                'predicted_gender': predicted_gender,
                'male_score': male_score,
                'female_score': female_score,
            })
        
        return results


# ============================================================
# Analysis Functions
# ============================================================

def load_json_metadata(json_path: str) -> pd.DataFrame:
    """Load JSON file and convert to DataFrame"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Ensure 'id' column exists
    if 'id' not in df.columns:
        print("[WARN] JSON does not have 'id' column, using index")
        df['id'] = [f"{i:04d}" for i in range(1, len(df) + 1)]
    
    return df[['id', 'trait', 'keywords', 'description', 'prompt_text']]


def merge_with_metadata(results: List[Dict], metadata_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge detection results with metadata"""
    results_df = pd.DataFrame(results)
    
    if metadata_df is not None:
        # Merge on 'id'
        merged_df = results_df.merge(metadata_df, on='id', how='left')
    else:
        merged_df = results_df
        merged_df['trait'] = 'N/A'
        merged_df['keywords'] = 'N/A'
    
    return merged_df


def compute_statistics(df: pd.DataFrame) -> Dict:
    """Compute gender distribution statistics"""
    stats = {}
    
    # Overall distribution
    overall_counts = df['predicted_gender'].value_counts()
    overall_pct = df['predicted_gender'].value_counts(normalize=True) * 100
    
    stats['overall'] = {
        'counts': overall_counts.to_dict(),
        'percentages': overall_pct.to_dict(),
        'total': len(df)
    }
    
    # Distribution by trait
    if 'trait' in df.columns:
        trait_stats = {}
        for trait in df['trait'].unique():
            trait_df = df[df['trait'] == trait]
            trait_counts = trait_df['predicted_gender'].value_counts()
            trait_pct = trait_df['predicted_gender'].value_counts(normalize=True) * 100
            
            # Calculate female/male ratio
            female_count = trait_counts.get('female', 0)
            male_count = trait_counts.get('male', 0)
            ratio = female_count / male_count if male_count > 0 else float('inf')
            
            trait_stats[trait] = {
                'counts': trait_counts.to_dict(),
                'percentages': trait_pct.to_dict(),
                'female_male_ratio': ratio,
                'total': len(trait_df)
            }
        
        stats['by_trait'] = trait_stats
    
    # Distribution by keywords
    if 'keywords' in df.columns:
        keyword_stats = {}
        for keyword in df['keywords'].unique():
            kw_df = df[df['keywords'] == keyword]
            kw_counts = kw_df['predicted_gender'].value_counts()
            kw_pct = kw_df['predicted_gender'].value_counts(normalize=True) * 100
            
            # Calculate female/male ratio
            female_count = kw_counts.get('female', 0)
            male_count = kw_counts.get('male', 0)
            ratio = female_count / male_count if male_count > 0 else float('inf')
            
            keyword_stats[keyword] = {
                'counts': kw_counts.to_dict(),
                'percentages': kw_pct.to_dict(),
                'female_male_ratio': ratio,
                'total': len(kw_df)
            }
        
        stats['by_keyword'] = keyword_stats
    
    return stats


def save_statistics(stats: Dict, output_dir: Path):
    """Save statistics to separate CSV files"""
    
    # Overall statistics
    overall_df = pd.DataFrame({
        'Gender': list(stats['overall']['counts'].keys()),
        'Count': list(stats['overall']['counts'].values()),
        'Percentage': [stats['overall']['percentages'].get(g, 0) for g in stats['overall']['counts'].keys()]
    })
    overall_csv = output_dir / 'overall_gender_distribution.csv'
    overall_df.to_csv(overall_csv, index=False)
    print(f"[INFO] Saved overall statistics to {overall_csv}")
    
    # By trait statistics
    if 'by_trait' in stats:
        trait_rows = []
        for trait, trait_data in stats['by_trait'].items():
            row = {
                'trait': trait,
                'female_count': trait_data['counts'].get('female', 0),
                'male_count': trait_data['counts'].get('male', 0),
                'female_pct': trait_data['percentages'].get('female', 0),
                'male_pct': trait_data['percentages'].get('male', 0),
                'female_male_ratio': trait_data['female_male_ratio'],
                'total': trait_data['total']
            }
            trait_rows.append(row)
        
        trait_df = pd.DataFrame(trait_rows)
        trait_csv = output_dir / 'gender_by_trait.csv'
        trait_df.to_csv(trait_csv, index=False)
        print(f"[INFO] Saved trait statistics to {trait_csv}")
    
    # By keyword statistics
    if 'by_keyword' in stats:
        keyword_rows = []
        for keyword, kw_data in stats['by_keyword'].items():
            row = {
                'keyword': keyword,
                'female_count': kw_data['counts'].get('female', 0),
                'male_count': kw_data['counts'].get('male', 0),
                'female_pct': kw_data['percentages'].get('female', 0),
                'male_pct': kw_data['percentages'].get('male', 0),
                'female_male_ratio': kw_data['female_male_ratio'],
                'total': kw_data['total']
            }
            keyword_rows.append(row)
        
        keyword_df = pd.DataFrame(keyword_rows)
        keyword_csv = output_dir / 'gender_by_keyword.csv'
        keyword_df.to_csv(keyword_csv, index=False)
        print(f"[INFO] Saved keyword statistics to {keyword_csv}")


def print_summary(stats: Dict):
    """Print summary statistics to console"""
    print("\n" + "="*60)
    print("GENDER ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall
    print("\n[Overall Gender Distribution]")
    print(f"Total samples: {stats['overall']['total']}")
    for gender, count in stats['overall']['counts'].items():
        pct = stats['overall']['percentages'].get(gender, 0)
        print(f"  {gender.capitalize()}: {count} ({pct:.2f}%)")
    
    # By trait
    if 'by_trait' in stats:
        print("\n[Gender Distribution by Trait]")
        for trait, trait_data in sorted(stats['by_trait'].items()):
            print(f"\n  {trait}:")
            print(f"    Total: {trait_data['total']}")
            for gender, count in trait_data['counts'].items():
                pct = trait_data['percentages'].get(gender, 0)
                print(f"    {gender.capitalize()}: {count} ({pct:.2f}%)")
            print(f"    Female/Male Ratio: {trait_data['female_male_ratio']:.3f}")
    
    print("\n" + "="*60)


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Gender Analysis Script for CoP_bias',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze WAV files only
  python analyze_gender.py --wav_path results/parler_large/ --output analysis/parler_large.csv
  
  # Analyze with metadata from JSON
  python analyze_gender.py --wav_path results/parler_mini/ --json descriptions/descriptions_persona_bias.json --output analysis/
  
  # Specify output directory for multiple CSV files
  python analyze_gender.py --wav_path results/voxinstruct/ --json descriptions/descriptions_multi_axis.json --output analysis/voxinstruct/
        """
    )
    
    parser.add_argument('--wav_path', type=str, required=True,
                       help='Directory containing WAV files')
    parser.add_argument('--json', type=str, default=None,
                       help='JSON file with metadata (optional)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file or directory')
    
    args = parser.parse_args()
    
    # Get WAV files
    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        print(f"[ERROR] WAV path does not exist: {wav_path}")
        sys.exit(1)
    
    wav_files = sorted(wav_path.glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No WAV files found in {wav_path}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(wav_files)} WAV files")
    
    # Load metadata if provided
    metadata_df = None
    if args.json:
        print(f"[INFO] Loading metadata from {args.json}")
        metadata_df = load_json_metadata(args.json)
        print(f"[INFO] Loaded metadata for {len(metadata_df)} items")
    
    # Initialize detector
    detector = GenderDetector()
    
    # Detect gender
    print("[INFO] Starting gender detection...")
    results = detector.batch_detect([str(f) for f in wav_files])
    
    # Merge with metadata
    df = merge_with_metadata(results, metadata_df)
    
    # Determine output path
    output_path = Path(args.output)
    if output_path.suffix == '.csv':
        # Single CSV file
        output_csv = output_path
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Saved results to {output_csv}")
    else:
        # Directory for multiple CSV files
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        output_csv = output_dir / 'detection_results.csv'
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Saved results to {output_csv}")
        
        # Compute and save statistics
        print("[INFO] Computing statistics...")
        stats = compute_statistics(df)
        save_statistics(stats, output_dir)
        
        # Print summary
        print_summary(stats)
    
    print(f"\n[INFO] Analysis complete!")


if __name__ == "__main__":
    main()
