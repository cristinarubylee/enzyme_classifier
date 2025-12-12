"""
Enzyme Classification Predictor
Predicts EC main class for protein sequences using trained ESM2 model.

Usage:
    python predict.py --sequence "MKTAYIAK..."
    python predict.py --fasta sequences.fasta --output predictions.csv
    python predict.py --fasta sequences.fasta --top-k 3
"""

import sys
import yaml
import torch
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging
from peft import PeftModel
from tqdm import tqdm


# EC class names
EC_CLASSES = [
    "Oxidoreductases",   # EC 1
    "Transferases",      # EC 2
    "Hydrolases",        # EC 3
    "Lyases",            # EC 4
    "Isomerases",        # EC 5
    "Ligases",           # EC 6
    "Translocases"       # EC 7
]

CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def print_header():
    print("\n" + "=" * 70)
    print("  ESM2 Enzyme Classifier (LoRA Fine-Tuned)")
    print("=" * 70 + "\n")


def print_separator():
    """Print a separator line."""
    print("-" * 70)


class EnzymeClassifier:
    """Enzyme EC class predictor using trained ESM2 model."""
    
    def __init__(self, model_path: str, config_path: str = "config.yaml", verbose: bool = True):
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.verbose:
            print(f"Device: {self.device.upper()}")
            if self.device == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            print()
        
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        
        if self.verbose:
            print("Loading model components...")
        
        # Load tokenizer
        if self.verbose:
            print("  [1/3] Loading tokenizer...", end=" ", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg['model']['name'], trust_remote_code=True
        )
        if self.verbose:
            print("Done")
        
        # Suppress warnings for base model loading
        logging.set_verbosity_error()
        
        # Load base model
        if self.verbose:
            print("  [2/3] Loading base model...", end=" ", flush=True)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg['model']['name'],
            num_labels=self.cfg['model']['num_labels'],
            trust_remote_code=True
        )
        if self.verbose:
            print("Done")
        
        logging.set_verbosity_warning()
        
        # Load LoRA adapter
        if self.verbose:
            print("  [3/3] Loading LoRA weights...", end=" ", flush=True)
        self.model = PeftModel.from_pretrained(base_model, model_path).to(self.device)
        self.model.eval()
        if self.verbose:
            print("Done")
            print("\nModel loaded successfully\n")
    
    def validate_sequence(self, sequence: str) -> Tuple[bool, str]:
        """Validate and clean protein sequence."""
        if not sequence:
            return False, "Empty sequence"
        
        seq = sequence.upper().strip()
        valid_aa = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ*.')
        invalid_chars = set(seq) - valid_aa
        
        if invalid_chars:
            return False, f"Invalid characters: {invalid_chars}"
        if len(seq) < 10:
            return False, f"Sequence too short ({len(seq)} aa, minimum 10)"
        if len(seq) > 20000:
            return False, f"Sequence too long ({len(seq)} aa, maximum 20000)"
        
        return True, seq
    
    def predict_single(self, sequence: str, return_all_probs: bool = False) -> Dict:
        """Predict EC class for a single sequence."""
        is_valid, result = self.validate_sequence(sequence)
        if not is_valid:
            return {'error': result, 'valid': False}
        
        cleaned_seq = result
        inputs = self.tokenizer(
            cleaned_seq, 
            return_tensors="pt", 
            padding=False, 
            truncation=True,
            max_length=1024  # ESM2 max length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_label = torch.max(probs, dim=-1)
        
        result = {
            'valid': True,
            'sequence_length': len(cleaned_seq),
            'predicted_class': pred_label.item(),
            'predicted_class_name': EC_CLASSES[pred_label.item()],
            'confidence': confidence.item(),
        }
        
        if return_all_probs:
            result['all_probabilities'] = {
                EC_CLASSES[i]: float(probs[0, i]) for i in range(len(EC_CLASSES))
            }
        
        return result
    
    def predict_batch(self, sequences: List[str], batch_size: int = 8,
                     return_all_probs: bool = False, show_progress: bool = True) -> List[Dict]:
        """Predict EC class for multiple sequences."""
        results = []
        
        # Create progress bar
        num_batches = (len(sequences) + batch_size - 1) // batch_size
        iterator = range(0, len(sequences), batch_size)
        
        if show_progress and self.verbose:
            iterator = tqdm(iterator, total=num_batches, desc="Processing batches", 
                          unit="batch", ncols=80)
        
        for i in iterator:
            batch_seqs = sequences[i:i+batch_size]
            valid_seqs, batch_results = [], []
            
            for seq in batch_seqs:
                is_valid, result = self.validate_sequence(seq)
                if is_valid:
                    valid_seqs.append(result)
                    batch_results.append({'valid': True})
                else:
                    batch_results.append({'valid': False, 'error': result})
            
            if valid_seqs:
                inputs = self.tokenizer(
                    valid_seqs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=1024
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                    confidence, pred_labels = torch.max(probs, dim=-1)
                
                valid_idx = 0
                for res in batch_results:
                    if res['valid']:
                        pred_label = pred_labels[valid_idx].item()
                        res.update({
                            'sequence_length': len(valid_seqs[valid_idx]),
                            'predicted_class': pred_label,
                            'predicted_class_name': EC_CLASSES[pred_label],
                            'confidence': confidence[valid_idx].item(),
                        })
                        
                        if return_all_probs:
                            res['all_probabilities'] = {
                                EC_CLASSES[k]: float(probs[valid_idx, k])
                                for k in range(len(EC_CLASSES))
                            }
                        valid_idx += 1
            
            results.extend(batch_results)
        
        return results
    
    def predict_from_fasta(self, fasta_path: str, batch_size: int = 8,
                          return_all_probs: bool = False) -> pd.DataFrame:
        """Predict EC class for sequences in a FASTA file."""
        if self.verbose:
            print(f"Reading FASTA file: {fasta_path}")
        
        sequences, seq_ids = [], []
        
        with open(fasta_path) as f:
            current_seq, current_id = [], None
            
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_id:
                        sequences.append(''.join(current_seq))
                        seq_ids.append(current_id)
                    current_id = line[1:].strip()
                    current_seq = []
                else:
                    current_seq.append(line)
            
            if current_id:
                sequences.append(''.join(current_seq))
                seq_ids.append(current_id)
        
        if self.verbose:
            print(f"Found {len(sequences)} sequences\n")
        
        results = self.predict_batch(sequences, batch_size, return_all_probs, show_progress=True)
        
        if self.verbose:
            print()  # Extra line after progress bar
        
        df_data = []
        for seq_id, result in zip(seq_ids, results):
            row = {'sequence_id': seq_id}
            
            if result['valid']:
                row.update({
                    'predicted_class': result['predicted_class'] + 1,  # Convert to EC number
                    'predicted_class_name': result['predicted_class_name'],
                    'confidence': result['confidence'],
                    'sequence_length': result['sequence_length'],
                    'status': 'success'
                })
                
                if return_all_probs:
                    for class_name, prob in result['all_probabilities'].items():
                        row[f'prob_{class_name}'] = prob
            else:
                row.update({
                    'status': 'failed',
                    'error': result['error']
                })
            
            df_data.append(row)
        
        return pd.DataFrame(df_data)


def format_confidence_bar(confidence: float, width: int = 20) -> str:
    """Create a visual confidence bar."""
    filled = int(confidence * width)
    empty = width - filled
    return "[" + "=" * filled + " " * empty + "]"


def main():
    parser = argparse.ArgumentParser(
        description='Predict EC main class for protein sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py -s "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLALN"
  python predict.py -f sequences.fasta -o predictions.csv
  python predict.py -f sequences.fasta -k 3
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--sequence', '-s', help='Single protein sequence')
    input_group.add_argument('--fasta', '-f', help='FASTA file with sequences')
    
    parser.add_argument('--model-path', '-m', default=None,
                       help='Path to trained model (default: from config.yaml)')
    parser.add_argument('--batch-size', '-b', type=int, default=8,
                       help='Batch size for prediction (default: 8)')
    parser.add_argument('--top-k', '-k', type=int, default=None,
                       help='Show top K class probabilities')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Print header
    if not args.quiet:
        print_header()
    
    # Load config to get model path if not specified
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        model_path = args.model_path or cfg['paths']['final_model']
        
        if not args.quiet:
            print(f"Model path: {model_path}\n")
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        classifier = EnzymeClassifier(model_path, CONFIG_PATH, verbose=not args.quiet)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.sequence:
        if not args.quiet:
            print("Running prediction...")
            print_separator()
        
        result = classifier.predict_single(args.sequence, return_all_probs=(args.top_k is not None))
        
        if not result['valid']:
            print(f"\nError: {result['error']}")
            sys.exit(1)
        
        # Print results
        print("\nRESULTS")
        print_separator()
        print(f"Sequence length:     {result['sequence_length']} amino acids")
        print()
        print(f"Predicted class:     EC {result['predicted_class'] + 1} - {result['predicted_class_name']}")
        print(f"Confidence:          {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"                     {format_confidence_bar(result['confidence'])}")
        
        if args.top_k:
            print(f"\nTop {args.top_k} predictions:")
            print_separator()
            probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
            for i, (class_name, prob) in enumerate(probs[:args.top_k], 1):
                class_num = EC_CLASSES.index(class_name) + 1
                bar = format_confidence_bar(prob, width=15)
                print(f"  {i}. EC {class_num} - {class_name:20s} {prob:.4f} ({prob*100:5.2f}%) {bar}")
        
        print("\n" + "="*70 + "\n")
    
    elif args.fasta:
        try:
            df = classifier.predict_from_fasta(
                args.fasta, batch_size=args.batch_size, return_all_probs=(args.top_k is not None)
            )
            
            # Print summary
            if not args.quiet:
                print("SUMMARY")
                print_separator()
                print(f"Total sequences:     {len(df)}")
                print(f"Successful:          {(df['status'] == 'success').sum()}")
                print(f"Failed:              {(df['status'] == 'failed').sum()}")
                
                if (df['status'] == 'success').any():
                    successful_df = df[df['status'] == 'success']
                    print(f"\nMean confidence:     {successful_df['confidence'].mean():.4f}")
                    print(f"Min confidence:      {successful_df['confidence'].min():.4f}")
                    print(f"Max confidence:      {successful_df['confidence'].max():.4f}")
                    
                    print("\nClass distribution:")
                    for class_name, count in successful_df['predicted_class_name'].value_counts().sort_index().items():
                        class_num = EC_CLASSES.index(class_name) + 1
                        pct = (count / len(successful_df)) * 100
                        print(f"  EC {class_num} - {class_name:20s} {count:5d} ({pct:5.1f}%)")
            
            if args.output:
                df.to_csv(args.output, index=False)
                if not args.quiet:
                    print(f"\nResults saved to: {args.output}")
            else:
                print("\nPREVIEW (first 10 rows):")
                print_separator()
                preview_cols = ['sequence_id', 'predicted_class_name', 'confidence', 'status']
                print(df[preview_cols].head(10).to_string(index=False))
                if len(df) > 10:
                    print(f"\n... and {len(df) - 10} more rows")
                    print("\nUse --output to save full results to CSV")
            
            if not args.quiet:
                print("\n" + "="*70 + "\n")
        
        except FileNotFoundError:
            print(f"Error: FASTA file not found: {args.fasta}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()