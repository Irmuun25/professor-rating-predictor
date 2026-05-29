# Predicting Professor Review Ratings

A natural language processing project that fine-tunes a DistilBERT transformer model to predict 1-5 star ratings purely from the text of student reviews. 

**Author**: Munkh-Irmuun Munkhbat

## Overview
This repository contains a complete pipeline for fetching, processing, and classifying professor reviews. The model is designed to determine if transformer-based sentiment analysis can accurately predict numerical star ratings based on subjective text feedback. 

## Data Source
The dataset consists of 1,250 reviews collected dynamically using the PlanetTerp API. The data covers five Computer Science and Mathematics professors at the University of Maryland: (You can change the names in ipynb file.)
* Nelson Padua-Perez
* Fawzi Emad
* Justin Wyss-Gallifent
* Jason Filippou
* Larry Herman

## Model Architecture and Training
* **Base Model**: `distilbert-base-uncased`
* **Task**: 5-class Sequence Classification (1-5 stars)
* **Frameworks**: PyTorch & Hugging Face Transformers
* **Training Details**: Trained for 3 epochs with a batch size of 16 and a learning rate of 2e-5 using the AdamW optimizer
* **Data Split**: 80/20 train/validation split (999 training reviews, 250 validation reviews).

## Key Results
The fine-tuned model demonstrates strong sentiment detection capabilities, particularly on extreme ratings. 
* **Overall Exact Accuracy**: 66%
* **Within ±1 Star Accuracy**: 86%
* **F1 Score**: 0.58
* **Recall Highs**: 94% recall on 5-star reviews and 88% recall on 1-star reviews.

The model rarely confuses extreme ratings but faces some challenges distinguishing subtle linguistic differences in mid-range (2-4 star) reviews, partially due to class imbalances in the source data.

```markdown
## Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/professor-rating-predictor.git](https://github.com/yourusername/professor-rating-predictor.git)
   cd professor-rating-predictor

```

2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Run the Jupyter Notebook to fetch the data, train the model, and view the evaluation matrices.

---

### `requirements.txt`

This file lists all the dependencies required to run your setup and training blocks successfully.

```text
transformers>=4.0.0
torch>=2.0.0
datasets
pandas
scikit-learn
matplotlib
seaborn
requests
tqdm
numpy
```
### `.gitignore`

This file ensures that temporary files, caches, and potentially heavy downloaded model weights are not accidentally committed to your repository.

```text
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Jupyter Notebook
.ipynb_checkpoints

# Virtual Environments
venv/
env/
.env/

# Data and Models
*.csv
*.pt
*.bin
*.safetensors
/results/
/logs/

# IDE settings
.vscode/
.idea/

```
