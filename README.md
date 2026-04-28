# 🌐 English → Spanish Transformer Translator

A production-ready neural machine translation system powered by a custom Transformer architecture, built from scratch using PyTorch.

## 🚀 Features
- **Transformer from Scratch**: Implementation of the Transformer architecture (Encoder-Decoder) using the `rui.torch.transformer` engine.
- **No External APIs**: Performs translation entirely locally. No Google Translate API or HuggingFace transformers used.
- **Premium UI**: A modern, dark-themed Streamlit application with glassmorphism and real-time translation history.
- **Serialized Pipeline**: Full persistence of vectorizers, model weights, and hyperparameters for easy deployment.
- **Jupyter Ready**: Includes `translater.ipynb` for full visibility into the training process.

## 🛠 Tech Stack
- **Core**: Python, PyTorch, NumPy
- **Interface**: Streamlit
- **Data**: `spa.txt` (118k English-Spanish pairs)
- **Architecture**:
  - `d_emb`: 128
  - `n_layers`: 2
  - `n_heads`: 8
  - `d_ff`: 512

## 🏃‍♂️ How to Run

### 1. Web Application
Launch the interactive translator:
```bash
streamlit run app.py
```

### 2. Command Line Inference
Translate a single sentence directly:
```bash
python inference.py "Hello, how are you today?"
```

### 3. Training & Exploration
Open `translater.ipynb` in Jupyter or VS Code to see the data processing, model training, and accuracy evaluations.

## 📂 Project Structure
- `rui/`: Core package containing the Transformer and TextVectorizer implementations.
- `app.py`: Streamlit web interface logic and custom CSS.
- `inference.py`: Model loading and greedy decoding logic.
- `models/`: Saved model weights (`.pt`), vectorizers (`.pkl`), and configuration (`.json`).
- `spa.txt`: The English-Spanish dataset.

---
Built as part of an advanced AI coding project.
