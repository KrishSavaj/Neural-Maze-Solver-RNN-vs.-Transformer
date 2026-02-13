# Quick Start Training Guide

Get up and running with the Maze Solver in 5 minutes!

## 🚀 Quick Setup (5 minutes)

```bash
# 1. Clone and navigate
git clone https://github.com/yourusername/maze-solver.git
cd maze-solver

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 🎯 Train Models

### Option 1: Train RNN (Recommended for beginners)

```bash
python train_rnn.py
```

**Expected Output:**
- Console logs showing epoch progress
- Final test loss: ~1.56
- Test accuracy: ~72.7%
- Duration: 15-30 minutes

### Option 2: Train Transformer (Faster)

```bash
python train_transformer.py
```

**Expected Output:**
- Final test loss: ~0.56
- Test accuracy: ~81%
- Duration: 10-20 minutes

### Option 3: Train Both (Recommended)

```bash
# In two separate terminals
python train_rnn.py       # Terminal 1
python train_transformer.py  # Terminal 2
```

## 📊 Evaluate & Compare

```bash
python eval.py
```

This will:
- Load both trained models
- Run inference on test set
- Generate performance comparison
- Visualize predictions

## 🔧 Configuration (Optional)

Edit hyperparameters directly in the training scripts:

**In `train_rnn.py` (lines ~20-35):**
```python
EMBEDDING_DIM = 128       # Increase for more capacity
HIDDEN_DIM = 512
BATCH_SIZE = 32           # Reduce if OOM error
EPOCHS = 20
LEARNING_RATE = 0.001     # Decrease for stability
TEACHER_FORCING_RATIO = 0.5
```

**In `train_transformer.py` (lines ~20-35):**
```python
D_MODEL = 128             # Embedding dimension
NHEAD = 8                 # Attention heads
NUM_LAYERS = 6            # Increase for more depth
BATCH_SIZE = 64
EPOCHS = 35
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Reduce `BATCH_SIZE` to 16 or 8 |
| **Very slow training** | Check GPU: `torch.cuda.is_available()` |
| **Model not improving** | Increase `EPOCHS` or adjust `LEARNING_RATE` |
| **Import errors** | Run `pip install -r requirements.txt` again |

## 📈 Expected Results

### RNN Performance
```
Epoch 1:  Train Loss: 2.45, Val Loss: 2.10, Val Acc: 35%
Epoch 5:  Train Loss: 1.80, Val Loss: 1.50, Val Acc: 55%
Epoch 10: Train Loss: 1.20, Val Loss: 1.35, Val Acc: 68%
Epoch 20: Train Loss: 0.80, Val Loss: 1.56, Val Acc: 73%
```

### Transformer Performance
```
Epoch 1:  Train Loss: 2.10, Val Loss: 1.85, Val Acc: 42%
Epoch 5:  Train Loss: 1.20, Val Loss: 1.05, Val Acc: 65%
Epoch 10: Train Loss: 0.75, Val Loss: 0.72, Val Acc: 78%
Epoch 35: Train Loss: 0.45, Val Loss: 0.56, Val Acc: 81%
```

## 🎓 Next Steps

1. **Compare Results**: Check which model works best for your use case
   - Need path coherence? → Use RNN
   - Need accuracy? → Use Transformer

2. **Visualize Predictions**: Check `eval.py` for maze visualizations

3. **Read Full Report**: See `report.pdf` for detailed analysis

4. **Extend the Project**: Try the ideas in README's "Extending the Project" section

## 📚 File Outputs

After training, you'll get:

```
├── rnn_model_final.pt              # Trained RNN model
├── transformer_model_final.pt      # Trained Transformer model
├── rnn_training_log.csv           # RNN metrics per epoch
├── transformer_training_log.csv   # Transformer metrics per epoch
├── rnn_metrics.json               # RNN final metrics
├── transformer_metrics.json       # Transformer final metrics
└── plots/                         # Visualizations
    ├── rnn_loss.png
    ├── rnn_accuracy.png
    ├── transformer_loss.png
    ├── transformer_accuracy.png
    └── maze_predictions.png
```

## 💡 Tips for Best Results

1. **Run on GPU**: Ensure CUDA is available (`nvidia-smi`)
2. **Full Training**: Don't interrupt training mid-way (checkpoints available in code)
3. **Fresh Start**: Delete old `.pt` files before retraining
4. **Monitor Memory**: Watch GPU memory usage: `watch -n 1 nvidia-smi`
5. **Patience**: Transformers take ~35 epochs, RNNs need ~20 epochs

## 🔍 Debugging

Print model architecture:
```python
from model import Seq2Seq, EncoderRNN, DecoderRNN, Attention

encoder = EncoderRNN(vocab_size, 128, 512, 2, 0.1, pad_idx=0)
print(encoder)  # Shows all layers
```

Check training progress:
```python
# Look at training_log.csv
import pandas as pd
df = pd.read_csv('rnn_training_log.csv')
df.tail(10)  # Last 10 epochs
```

Inspect model predictions:
```python
# Run individual inference in eval.py
predicted_path = model.inference(maze_input)
print(predicted_path)
```

## 📞 Need Help?

- Check README.md for detailed documentation
- Review report.pdf for theory and methodology
- Check code comments for implementation details
- Open an issue on GitHub

---

**Happy Training! 🎉**
