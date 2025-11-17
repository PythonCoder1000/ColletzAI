# Collatz Conjecture Neural Network Predictor

A deep learning project that predicts the number of steps and maximum value reached in the Collatz conjecture sequence using various neural network architectures.

## Features

- **Multiple Model Architectures**: MLP, LSTM, Hybrid (MLP+LSTM), and Transformer models
- **Fine-tuning Capabilities**: Learning rate scheduling, multiple optimizers, gradient clipping
- **Apple Silicon Support**: Automatically uses MPS (Metal Performance Shaders) on M4 chip
- **Comprehensive Training**: Early stopping, validation tracking, model checkpointing

## Project Structure

```
.
├── src/
│   ├── models.py      # Model architectures (MLP, LSTM, Hybrid, Transformer)
│   ├── train.py       # Training script with fine-tuning options
│   ├── test.py        # Model testing and prediction interface
│   └── generate.py    # Data generation for Collatz sequences
├── data/
│   └── data.txt       # Training data (value, steps, max_value)
├── models/            # Saved model checkpoints (gitignored)
└── requirements.txt   # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "Christian Jin (During the washington trip)"
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

Train a model with default settings (Hybrid model):
```bash
python src/train.py
```

The training script supports multiple model types. Edit `src/train.py` to change:
- `model_type`: 'mlp', 'lstm', 'hybrid', or 'transformer'
- `optimizer_type`: 'adam', 'adamw', or 'sgd'
- `scheduler_type`: 'reduce_on_plateau', 'cosine', 'step', or 'warmup_cosine'

### Testing a Model

Run the interactive testing interface:
```bash
python src/test.py
```

### Generating Data

Generate Collatz sequence data:
```bash
python src/generate.py
```

## Model Architectures

1. **MLP**: Multi-layer perceptron with batch normalization
2. **LSTM**: Long Short-Term Memory network with bidirectional option
3. **Hybrid**: Combines MLP and LSTM branches for enhanced performance
4. **Transformer**: Transformer-based encoder model

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm

See `requirements.txt` for specific versions.

## License

[Add your license here]

## Author

[Your name]

