# 🚀 GPT-Style Byte Pair Encoding (BPE) Tokenizer From Scratch in Pure Python

[![Python Version](https://shields.io)](https://python.org)
[![License: MIT](https://shields.io)](https://opensource.org)
[![Colab](https://shields.io)](https://google.com)

The ultimate production-grade, highly optimized **Byte Pair Encoding (BPE) Tokenizer** built completely from scratch. This repository mirrors the exact subword tokenization mechanics used by modern state-of-the-art Large Language Models (**LLMs**) such as **GPT-4, Llama 3, Mistral, and BERT**. 

Engineered natively to handle complex multilingual datasets, this pipeline flawlessly processes **English text, Hindi Unicode formatting (Complex Scripts), and special character matrices** without token corruption or script deformation.

---

## ⚡ Test This Code Instantly on Your Computer or Google Colab!

This entire pipeline is self-contained and optimized to run with **zero external dependencies** if needed, making it fully compatible with alternative Python runtimes (like **PyPy JIT**).

### 🖥️ Option 1: Run Locally on Your Computer
1. Clone this repository:
   ```bash
   git clone https://github.com
   cd Byte-Pair-Encoding-BPE-GPT-Style
   ```
2. Run the main processing loop:
   ```bash
   python Decoder.py
   ```

### ☁️ Option 2: Run on Google Colab Notebook
You can tokenize datasets in the cloud instantly. Open a blank **Google Colab Notebook** and run:
```python
# 1. Clone the repository directly into your Colab workspace
!git clone https://github.com
%cd Byte-Pair-Encoding-BPE-GPT-Style

# 2. Run the sample script to see the production-grade tokenization matrix
!python Decoder.py
```

---

## 💎 Elite Features & Architecture

* **GPT-Standard Tokenization:** Custom iterative pair-merging calculations built from first principles.
* **Flawless Multilingual Support:** Seamlessly transitions between English characters and dense Hindi subwords (`à¤¾`, `à¥\x87`) without data loss.
* **Full Special Token Matrix:** Native implementation of sequence barriers (`<BOS>`, `<EOS>`) and dynamic multi-element padding alignment (`<PAD>`).
* **Hugging Face Style Output:** Out-of-the-box dictionary compiler containing both `input_ids` and structural `attention_mask` tensors.
* **Secure Mojibake Recovery:** Employs precise `latin-1` byte-stitching layers to perfectly reverse text-compression deformations back into clean UTF-8 text strings.

---

## 📦 Professional Tokenizer Output Format

When text passes through the compiled `tokenizer()` function, it automatically generates a machine-ready dictionary ready to be pushed directly into PyTorch networks:

```python
{
    'input_ids': array([276,  72, 101, 108, 112, 277, 275, 275], dtype=int32),
    'attention_mask': array([  1,   1,   1,   1,   1,   1,   0,   0], dtype=int32)
}
```
* **`276`** = Beginning of Sequence (`<BOS>`) — *Injected seamlessly at Index 0.*
* **`277`** = End of Sequence (`<EOS>`) — *Injected at the text termination boundary.*
* **`275`** = Padding Token (`<PAD>`) — *Monitored by the `0` masking value so the attention layer entirely ignores it.*

---

## 🛠️ Project Component Structure

```text
BPE/
├── Eecoder.py      # BPE subword rank calculator & string packager
├── Decoder.py      # Secure bytes-stitching text reconstruction engine
├── BPE.py          # Main formatting compiler (Pads, Matrix, & Masks)
├── vocab.json      # Complete base vocabulary character mapping (0-255 + merges)
├── mearg.json      # Prioritized subword merge ranking registry
└── README.md       # Project specifications & SEO Engine
```

---

## 🚀 Step-by-Step Deep Learning Integration (PyTorch)

To scale this tokenizer into a high-performance deep learning pipeline, simply wrap it in a standard PyTorch custom `Dataset` class:

```python
import torch
from torch.utils.data import Dataset
from tokanizer import tokenizer

class MultilingualTextDataset(Dataset):
    def __init__(self, dataset_strings, max_sequence_length=16):
        self.data = dataset_strings
        self.max_len = max_sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Triggers the fast tokenizer pipeline
        tokenized_package = tokenizer(self.data[idx], padd=True, max_length=self.max_len, Truncate=True)
        
        return {
            'input_ids': torch.tensor(tokenized_package['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokenized_package['attention_mask'], dtype=torch.long)
        }
```

---

## 🤝 Contributing & Support

If this repository helped you understand how modern LLM tokenizers work from scratch, please **give it a ⭐ Star**! Feel free to fork the repository, open issues, or submit pull requests to accelerate execution speeds even further.

*Maintained with LOVE by [Codaaman](https://github.com)* ~AK-47
