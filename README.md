# Maze Solver: Learning Pathfinding with Sequence Models

A deep learning project implementing and comparing **RNN with Bahdanau Attention** and **Transformer** architectures for autonomous maze pathfinding. This project explores how modern sequence-to-sequence models can learn to navigate mazes by predicting coordinate sequences.

<div align="center">
  <img alt="Maze Solving Example" src="https://via.placeholder.com/500x300?text=Maze+Solving+Visualization" width="500"/>
</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Architectures](#architectures)
- [Installation](#installation)
- [Usage](#usage)
  - [Training RNN Model](#training-rnn-model)
  - [Training Transformer Model](#training-transformer-model)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Comparative Analysis](#comparative-analysis)
- [Methodology](#methodology)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [References](#references)

---

## 🎯 Overview

This project tackles the **maze pathfinding problem** using sequence-to-sequence models. Given a maze represented as an adjacency list, the models predict a valid path from origin to target as a sequence of coordinate tokens.

**Problem Statement:** Learn to map maze descriptions (encoded as adjacency lists) to valid paths (coordinate sequences) using neural networks.

**Key Challenge:** The model must understand spatial relationships, wall placements, and generate coherent sequences that form valid paths without revisiting positions or hitting walls.

### Why This Problem?

- **Demonstrates sequence modeling**: Models must learn sequential dependencies
- **Tests spatial reasoning**: Understanding 2D coordinate systems and adjacency
- **Compares architectures**: Direct comparison between RNN's sequential bias and Transformer's parallel attention
- **Real-world relevance**: Pathfinding is fundamental to robotics, game AI, and navigation systems

---

## 📁 Project Structure

```
maze-solver/
├── model.py                    # RNN Encoder-Decoder with Bahdanau Attention
├── transformer_model.py        # Transformer-based Seq2Seq model
├── train_rnn.py               # Training script for RNN model
├── train_transformer.py       # Training script for Transformer model
├── eval.py                    # Evaluation & inference utilities
├── report.pdf                 # Detailed project report with results
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Code Organization

- **`model.py`**: Contains core RNN architecture
  - `EncoderRNN`: Vanilla RNN encoder with pack_padded_sequence for efficiency
  - `Attention`: Bahdanau attention mechanism
  - `DecoderRNN`: RNN decoder with attention
  - `Seq2Seq`: Complete end-to-end seq2seq wrapper

- **`transformer_model.py`**: Contains Transformer architecture
  - `PositionalEncoding`: Sinusoidal positional encoding
  - `TransformerSeq2Seq`: Full Transformer model with masking

- **`train_rnn.py`**: Complete RNN training pipeline
  - Data loading and preprocessing
  - Training loop with validation
  - Performance tracking and visualization
  - Maze path visualization

- **`train_transformer.py`**: Complete Transformer training pipeline
  - Parallel architecture with OneCycleLR scheduler
  - Efficient batch processing
  - Metrics computation

- **`eval.py`**: Inference and evaluation
  - Model loading
  - Inference on new samples
  - Metrics computation (Token Accuracy, Sequence Accuracy, F1 Score)

---

## ⭐ Key Features

✅ **Two State-of-the-Art Architectures**
- RNN with Bahdanau Attention for sequential processing
- Transformer with multi-head self-attention for parallel processing

✅ **Comprehensive Training Pipeline**
- Data preprocessing and tokenization
- Teacher forcing for stable training
- Learning rate scheduling (OneCycleLR for Transformer)
- Early stopping and checkpoint management

✅ **Robust Evaluation**
- Token-level accuracy (individual coordinate predictions)
- Sequence-level accuracy (complete valid paths)
- F1 scores for imbalanced data
- Attention weight visualization

✅ **Production-Ready Code**
- Padding-aware computations with `pack_padded_sequence`
- GPU support with automatic device detection
- Reproducible results with seeded randomness
- Clean, documented codebase

✅ **Visualization & Analysis**
- Training/validation/test loss curves
- Accuracy and F1 score tracking
- Maze path visualizations (predicted vs. ground truth)
- Comparative analysis between architectures

---

## 🏗️ Architectures

### 1. RNN with Bahdanau Attention

**Architecture Overview:**
```
Input (Maze) → Embedding → Encoder RNN → [Encoder Outputs, Hidden State]
                                              ↓
                                           Attention Module
                                              ↓
                           [Context Vector] + Decoder Input → Decoder RNN → Output
```

**Key Components:**
- **Encoder**: 2-layer vanilla RNN (not LSTM/GRU) with 512 hidden units
- **Attention**: Dynamic context vector based on decoder state and encoder outputs
- **Decoder**: Generates output sequence token-by-token with teacher forcing

**Mathematical Formulation:**
```
Energy(i,j) = tanh(W_enc * h_j + W_dec * s_{i-1})
Attention(i,j) = softmax(V^T * Energy(i,j))
Context_i = Σ Attention(i,j) * h_j
```

**Hyperparameters:**
- Embedding Dimension: 128
- Hidden Dimension: 512
- Layers: 2
- Dropout: 0.1
- Teacher Forcing Ratio: 0.5

**Advantages:**
- Sequential processing forces coherent path generation
- Attention mechanism provides interpretability
- Memory efficient for variable-length sequences

### 2. Transformer Architecture

**Architecture Overview:**
```
Input → Embedding + Positional Encoding → Encoder Stack → Cross-Attention
                                                               ↓
                              Target + PE → Decoder Stack → Output
```

**Key Components:**
- **Positional Encoding**: Sinusoidal positional encodings to preserve sequence order
- **Encoder**: 6 transformer layers with multi-head self-attention
- **Decoder**: 6 transformer layers with masked self-attention and cross-attention
- **Masking**: Causal masking for autoregressive generation

**Hyperparameters:**
- Model Dimension (d_model): 128
- Attention Heads: 8 (each head: 16 dimensions)
- Feed-forward Dimension: 512
- Encoder Layers: 6
- Decoder Layers: 6
- Dropout: 0.1

**Advantages:**
- Parallel processing of entire sequences
- Direct attention to all maze positions
- Better for capturing global dependencies
- Superior token-level accuracy

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/maze-solver.git
cd maze-solver
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies Include:
- `torch>=2.0.0` - Deep learning framework
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.4.0` - Visualization
- `scikit-learn>=0.24.0` - Metrics

---

## 🚀 Usage

### Training RNN Model

```bash
python train_rnn.py
```

**Key Flags/Configuration (in code):**
```python
# Set in train_rnn.py
EMBEDDING_DIM = 128
HIDDEN_DIM = 512
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5
```

**Output:**
- Trained model checkpoint: `rnn_model_final.pt`
- Training logs: CSV with loss/accuracy metrics
- Visualizations: Training curves and maze predictions

**Training Time:** ~15-30 minutes (depends on dataset size and hardware)

### Training Transformer Model

```bash
python train_transformer.py
```

**Key Configuration:**
```python
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 35
```

**Output:**
- Trained model checkpoint: `transformer_model_final.pt`
- Training logs with OneCycleLR scheduler
- Performance curves and validation results

**Training Time:** ~10-20 minutes (faster due to parallel processing)

### Evaluation

```bash
python eval.py --model rnn --checkpoint rnn_model_final.pt --test-data test_mazes.txt
```

**Inference Example:**
```python
from eval import evaluate_model

# Load model and vocabulary
model, vocab = load_rnn_model('rnn_model_final.pt')

# Generate prediction
maze_input = "<ADJLIST_START> (0,0) <--> (0,1) ... <ADJLIST_END> ..."
predicted_path = model.generate(maze_input, vocab)
print(predicted_path)  # Output: (0,0) (0,1) (1,1) ... (5,5)
```

---

## 📊 Results

### RNN with Bahdanau Attention

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Token Accuracy** | 94.68% | 73.55% | 72.71% |
| **Sequence Accuracy** | 70.63% | 62.18% | 61.38% |
| **F1 Score** | 0.947 | 0.736 | 0.730 |
| **Final Loss** | - | - | 1.562 |

### Transformer

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Token Accuracy** | 83.27% | 81.14% | 81.14% |
| **Sequence Accuracy** | 32.32% | 29.82% | 29.51% |
| **F1 Score** | 0.826 | 0.801 | 0.802 |
| **Final Loss** | - | - | 0.559 |

### Key Observations

1. **RNN Excels at Path Coherence**: Higher sequence accuracy (61.38% vs 29.51%) indicates RNN better maintains valid path sequences throughout generation.

2. **Transformer Better at Token Prediction**: Higher token accuracy (81.14% vs 72.71%) shows superior individual coordinate prediction.

3. **Loss Trade-off**: Transformer achieves lower loss but this doesn't translate to better sequence-level performance, revealing the gap between token accuracy and path validity.

4. **Learning Dynamics**: RNN shows stable progression with gradual improvement, while Transformer converges faster but plateaus.

---

## 🔬 Comparative Analysis

### Architecture Strengths

| Aspect | RNN | Transformer |
|--------|-----|-------------|
| **Sequential Coherence** | ✅ Strong (61% seq acc) | ⚠️ Weak (29% seq acc) |
| **Token Prediction** | ⚠️ 72% | ✅ 81% |
| **Training Speed** | ⚠️ Slower | ✅ Faster |
| **Memory Efficiency** | ✅ Better with padding | ⚠️ Higher (full attention) |
| **Long-term Dependencies** | ⚠️ Limited | ✅ Direct connections |
| **Interpretability** | ✅ Attention weights | ✅ Attention heads |

### Why RNN Wins at Path Coherence

The RNN's sequential nature and teacher forcing ratio of 0.5 enforces learning valid transitions. The bottleneck of a fixed context vector from the encoder forces the model to commit to a globally coherent plan early.

### Why Transformer Wins at Token Accuracy

Parallel processing and multi-head attention allow simultaneous consideration of maze structure from multiple perspectives. Each head can specialize: one tracking position, another tracking adjacency constraints.

---

## 🔧 Methodology

### Data Format

Mazes are represented as structured text with three key components:

```
<ADJLIST_START>
(0,0) <--> (0,1) ;
(0,1) <--> (0,2) ;
(0,2) <--> (1,2) ;
...
<ADJLIST_END>

<ORIGIN_START> (0,0) <ORIGIN_END>

<TARGET_START> (5,5) <TARGET_END>

<PATH_START> (0,0) (0,1) (0,2) (1,2) ... (5,5) <PATH_END>
```

### Data Processing

1. **Tokenization**: Split into coordinate tokens, special tokens (SOS, EOS, PAD), and operators (<-->)
2. **Vocabulary Building**: Create token-to-index mappings (49 tokens for RNN, 50 for Transformer)
3. **Padding**: Pad sequences to max length with <PAD> token
4. **Train/Val/Test Split**: 70% / 15% / 15%

### Training Strategy

- **RNN**: Lower learning rate (0.001), teacher forcing, dropout 0.1
- **Transformer**: OneCycleLR scheduler, higher batch size, faster convergence
- **Loss Function**: Cross-entropy loss (ignoring padding tokens)
- **Metrics**: Token accuracy, Sequence accuracy, F1 score, Precision/Recall

### Evaluation Metrics

**Token Accuracy**: Percentage of individually correct coordinate predictions
```
Token Acc = (Correct Tokens) / (Total Non-Pad Tokens)
```

**Sequence Accuracy**: Percentage of completely valid end-to-end paths
```
Seq Acc = (Completely Valid Paths) / (Total Paths)
```

**F1 Score**: Harmonic mean of precision and recall
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

---

## 💾 Checkpoints & Pre-trained Models

Pre-trained models coming soon! You can train from scratch using the provided scripts.

**To save custom checkpoints:**
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'hyperparams': config
}, 'checkpoint_epoch_10.pt')
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Error
```python
# Reduce batch size in training script
BATCH_SIZE = 16  # Instead of 32/64
```

### Slow Training
```python
# Ensure GPU is being used
import torch
print(torch.cuda.is_available())  # Should be True
```

### Low Accuracy
- Increase training epochs
- Adjust learning rate
- Verify data preprocessing
- Check vocabulary matches between training and eval

---

## 📈 Extending the Project

### Possible Extensions

1. **Hybrid Architecture**: Combine RNN's sequence coherence with Transformer's accuracy
2. **Beam Search**: Implement k-beam search for better path generation
3. **Larger Mazes**: Test on 10x10, 20x20 grids
4. **Constraint Learning**: Explicitly teach path validity constraints
5. **Curriculum Learning**: Start with small mazes, gradually increase complexity
6. **LSTM/GRU Variants**: Compare with gate-based RNNs
7. **Multi-Task Learning**: Predict path AND wall configuration simultaneously

### Custom Dataset

To train on your own maze data:

```python
def create_custom_dataset(maze_files):
    """Load custom maze data in the same format"""
    sequences = []
    for file in maze_files:
        with open(file) as f:
            sequences.append(f.read())
    return sequences

# Then tokenize and create PyTorch dataset
```

---

## 📚 Requirements

See `requirements.txt`:
```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📄 Project Report

For detailed analysis, mathematical formulations, and comprehensive results, see [`report.pdf`](./report.pdf).

**Report Includes:**
- Theoretical frameworks for both architectures
- Implementation details and design choices
- Complete result tables and graphs
- Maze visualization comparisons
- Performance analysis and insights

---

## 🎓 References

### Key Papers

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - https://arxiv.org/abs/1706.03762

2. **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2015)
   - https://arxiv.org/abs/1409.0473

3. **Sequence to Sequence Learning with Neural Networks** (Sutskever et al., 2014)
   - https://arxiv.org/abs/1409.3215

### Learning Resources

- PyTorch Documentation: https://pytorch.org/docs/
- Attention Mechanism Explained: https://jalammar.github.io/attention-is-all-you-need/
- Transformer Architecture: https://jalammar.github.io/illustrated-transformer/

---

## 👥 Authors

- **Divy Shaileshbhai Dudhat** (2025BSY7560)
- **Chavda Uday** (2025AIB3001)

**Course**: COL 7341 - Machine Learning Assignment IV  
**Institution**: Indian Institute of Technology Delhi

---

## 📝 License

This project is provided as-is for educational purposes. Feel free to use, modify, and distribute for academic work.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⭐ Show Your Support

If this project helped you learn about sequence models, please give it a star! ⭐

---

## 📧 Questions & Support

For questions or issues:
- Open an issue on GitHub
- Check the `report.pdf` for detailed documentation
- Review code comments for implementation details

---

<div align="center">

**Made with ❤️ for the ML community**

*Last Updated: February 2025*

</div>
