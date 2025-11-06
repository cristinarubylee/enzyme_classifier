## Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/enzyme-classifier.git
cd enzyme-classifier
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download data**

Option A (notebook):
1. Open `notebooks/01_data_download.ipynb`
2. Run all cells (`Run → Run All`)
3. Confirm that the data was saved to `data/raw/`

Option B (manual):
1. Download from:  
   `https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&fields=accession,sequence,ec&query=(reviewed:true)%20AND%20(ec:*)`
2. Move the downloaded file to `data/raw/`


5. **Run preprocessing:**


## Usage

### Training
```bash
python src/model_training.py --config configs/lora_config.yaml
```

### Evaluation
```bash
python src/evaluation.py --model models/checkpoints/best
```

### Inference


## Experiments

Track experiments in `results/experiments.md` or use Weights & Biases.


## Data

Dataset source: UniProt
- Training samples: 
- Validation samples:
- Test samples:
- Classes: 7 EC main categories

## Acknowledgments

- ESM2 model by Meta AI
- Hugging Face Transformers & PEFT libraries