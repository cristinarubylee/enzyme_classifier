# Enzyme Classification with ESM2 and LoRA

A deep learning system for classifying enzymes into their EC (Enzyme Commission) main classes using protein language models developed for Cornell University's CS 4701 course (Practicum in Artificial Intelligence).

This project fine-tunes Meta AI's ESM2 (Evolutionary Scale Modeling 2) using LoRA (Low-Rank Adaptation) for efficient parameter training on enzyme sequence data from UniProt.

## Project Overview

Enzymes are biological catalysts classified by the International Union of Biochemistry and Molecular Biology (IUBMB) into 7 main classes based on the chemical reactions they catalyze. This project builds a classifier that predicts an enzyme's main EC class (1-7) from its amino acid sequence.

**EC Main Classes:**
1. **Oxidoreductases** - Catalyze oxidation-reduction reactions
2. **Transferases** - Transfer functional groups between molecules
3. **Hydrolases** - Catalyze hydrolysis reactions
4. **Lyases** - Break bonds by means other than hydrolysis
5. **Isomerases** - Catalyze geometric or structural changes within a molecule
6. **Ligases** - Join two molecules with covalent bonds
7. **Translocases** - Catalyze the movement of ions or molecules across membranes

## Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM
- ~10GB disk space for data and models

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/enzyme-classifier.git
cd enzyme-classifier
```

2. **Create virtual environment:**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download and prepare data**
   
   **Option A (Recommended - Notebook):**
   1. Open `notebooks/01_data_exploration.ipynb`
   2. Run all cells (`Run → Run All Cells`)
   3. This will automatically:
      - Download enzyme sequences from UniProt
      - Clean and validate the data
      - Perform exploratory data analysis
      - Balance the dataset
      - Create train/val/test splits
      - Tokenize sequences for model training
   4. Confirm that the data was saved to `data/raw/`, `data/processed/`, and `data/tokenized/`
   
   **Option B (Manual Download):**
   1. Download directly from UniProt:  
      ```
      https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&fields=accession,sequence,ec&query=(reviewed:true)%20AND%20(ec:*)
      ```
   2. Save the file to `data/raw/uniprot_raw.csv`
   3. Open `notebooks/01_data_exploration.ipynb` and run from the "Data Cleaning" section onwards

5. **Train the model**
   
   Open `notebooks/02_training.ipynb` and run all cells. This will:
   - Load the tokenized datasets
   - Initialize ESM2 model with LoRA adapters
   - Train for the specified number of epochs
   - Save checkpoints and the final model
   - Log metrics to Weights & Biases (if enabled)

6. **Evaluate the model**
   
   Open `notebooks/03_evaluation.ipynb` and run all cells. This will:
   - Generate predictions on validation and test sets
   - Calculate comprehensive metrics
   - Create confusion matrices and visualizations
   - Perform error analysis
   - Save all results to `results/`

## Usage

### Jupyter Notebook Pipeline (Recommended)

The complete ML pipeline is implemented in three Jupyter notebooks:

1. **Data Preparation:** `notebooks/01_data_exploration.ipynb`
   - Downloads and cleans data from UniProt
   - Performs exploratory data analysis
   - Balances dataset and creates splits
   - Tokenizes sequences

2. **Model Training:** `notebooks/02_training.ipynb`
   - Trains ESM2 with LoRA fine-tuning
   - Supports resuming from checkpoints
   - Tracks experiments with W&B
   - Saves best model based on macro F1-score

3. **Evaluation:** `notebooks/03_evaluation.ipynb`
   - Comprehensive performance analysis
   - Confusion matrices and per-class metrics
   - Error analysis and visualization
   - Generates detailed reports

### Command-Line Inference

Use the trained model to classify new protein sequences:

```bash
# Single sequence prediction
python predict.py -s "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQ..."

# Batch prediction from FASTA file
python predict.py -f sequences.fasta -o predictions.csv

# Show top-k predictions with probabilities
python predict.py -s "MKTAYIAK..." -k 3

# Specify custom model path
python predict.py -f sequences.fasta -m models/experiments/my_model

# Quiet mode (suppress progress messages)
python predict.py -f sequences.fasta -o predictions.csv -q
```

**Example output:**
```
======================================================================
  Enzyme Classification Predictor - CS 4701
======================================================================

RESULTS
----------------------------------------------------------------------
Sequence length:     342 amino acids

Predicted class:     EC 3 - Hydrolases
Confidence:          0.9234 (92.34%)
                     [==================  ]

Top 3 predictions:
----------------------------------------------------------------------
  1. EC 3 - Hydrolases         0.9234 (92.34%) [===============     ]
  2. EC 2 - Transferases       0.0521 ( 5.21%) [=               ]
  3. EC 1 - Oxidoreductases    0.0189 ( 1.89%) [                ]

======================================================================
```

## Project Structure

```
enzyme-classifier/
├── config.yaml                 # Central configuration file
├── predict.py                  # Command-line inference script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── notebooks/                  # Jupyter notebooks for complete pipeline
│   ├── 01_data_exploration.ipynb   # Data download, cleaning, and EDA
│   ├── 02_training.ipynb           # Model training with LoRA
│   └── 03_evaluation.ipynb         # Model evaluation and analysis
│
├── data/                       # Data directory (created by notebooks)
│   ├── raw/                    # Raw and cleaned CSV data
│   │   ├── uniprot_raw.csv
│   │   └── uniprot_cleaned.csv
│   ├── processed/              # Train/val/test splits
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── tokenized/              # Tokenized datasets (HuggingFace format)
│       ├── train_dataset/
│       ├── val_dataset/
│       └── test_dataset/
│
├── models/                     # Model directory (created by training)
│   ├── checkpoints/            # Training checkpoints (by epoch)
│   │   ├── checkpoint-1000/
│   │   ├── checkpoint-2000/
│   │   └── ...
│   └── final_model/            # Best trained model (LoRA weights)
│
└── results/                    # Results directory (created by evaluation)
    ├── predictions/            # Prediction CSVs with probabilities
    │   ├── validation_predictions.csv
    │   ├── test_predictions.csv
    │   └── misclassified_samples.csv
    ├── figures/                # Confusion matrices and plots
    │   ├── confusion_matrix_test.png
    │   └── confusion_matrix_val.png
    ├── metrics/                # Detailed classification reports
    │   ├── overall_metrics.csv
    │   ├── perclass_metrics_test.csv
    │   ├── classification_report_test.txt
    │   └── error_patterns.csv
    └── evaluation_summary.txt  # Comprehensive summary report
```

## Configuration

All hyperparameters and settings are centralized in `config.yaml`:

```yaml
# Model architecture
model:
  name: "facebook/esm2_t12_35M_UR50D"
  num_labels: 7
  lora:
    r: 8              # LoRA rank (adapter dimension)
    alpha: 16         # LoRA alpha (scaling factor)
    dropout: 0.1      # LoRA dropout rate
    target_modules: ["query", "value"]  # Attention layers to adapt

# Training settings
training:
  epochs: 3
  batch_size: 8
  learning_rate: 2e-4
  grad_accum_steps: 1
  fp16: true          # Mixed precision training

# Data balancing targets (samples per class)
data:
  targets:
    0: 30000  # EC 1 - Oxidoreductases
    1: 30000  # EC 2 - Transferases
    2: 30000  # EC 3 - Hydrolases
    3: 30000  # EC 4 - Lyases
    4: 30000  # EC 5 - Isomerases
    5: 30000  # EC 6 - Ligases
    6: 30000  # EC 7 - Translocases
  split:
    train: 0.70
    val: 0.15
    test: 0.15

# Weights & Biases tracking
wandb:
  enabled: true
  project: "enzyme-classification-esm2"
  run_name: "esm2_lora_baseline"
```

You can modify these settings directly in `config.yaml` or override them in the notebook cells for experimentation.

## Experiments

### Resume Training

Continue training from the last checkpoint:

```python
# In notebooks/02_training.ipynb
RESUME_TRAINING = True   # Continue from last checkpoint
EPOCHS = 5               # Train for 5 total epochs
```

### Hyperparameter Tuning

Create new experiments with different settings:

```python
# In notebooks/02_training.ipynb
EXPERIMENT_NAME = "high_lr_experiment"
LEARNING_RATE = 5e-4    # Higher learning rate
LORA_R = 16             # Larger LoRA rank
LORA_ALPHA = 32         # Corresponding alpha
BATCH_SIZE = 4          # Smaller batch size
```

Each experiment creates its own checkpoint directory and W&B run for easy comparison.

### Custom Data Balancing

Adjust the class distribution in `config.yaml`:

```yaml
data:
  targets:
    0: 50000  # More oxidoreductases
    1: 20000  # Fewer transferases
    2: 30000  # Standard hydrolases
    # ... etc
```

### Track Experiments

- **Weights & Biases:** Automatically tracks training metrics, loss curves, and validation performance
- **Results Directory:** All predictions and metrics saved to `results/` for later analysis
- **Checkpoints:** Model checkpoints saved after each epoch in `models/checkpoints/`

## Technical Details

### Model Architecture

- **Base Model:** ESM2-t12-35M (35 million parameters)
  - Pre-trained on 250 million protein sequences
  - 12 transformer layers, 480 hidden dimensions
  - Specialized tokenizer for protein sequences
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
  - Adds small trainable adapter matrices to attention layers
  - Only ~300k trainable parameters (<1% of base model)
- **Input:** Protein sequences up to 1024 amino acids
- **Output:** 7-class probability distribution over EC main classes


## Acknowledgments

- **Meta AI** for developing and open-sourcing the ESM2 protein language model
- **Hugging Face** for the Transformers and PEFT (Parameter-Efficient Fine-Tuning) libraries
- **UniProt** for providing the comprehensive enzyme sequence database