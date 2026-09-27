import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =========================
# Project paths
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "model"
TOKENIZER_DIR = ROOT_DIR / "tokenizer"
DATA_DIR = ROOT_DIR / "data"

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(TOKENIZER_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))


from transformer import MyAI
from tokenizer import Tokenizer
from dataset import AIDataset


# =========================
# Settings
# =========================

BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EPOCHS = 10
BLOCK_SIZE = 256

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================
# Files
# =========================

DATASET_FILE = DATA_DIR / "conversations.jsonl"
VOCAB_FILE = DATA_DIR / "vocab.json"
MODEL_FILE = ROOT_DIR / "my_ai.pt"


# =========================
# Load training texts
# =========================

texts = []

with open(
    DATASET_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        item = json.loads(line)

        for message in item.get(
            "messages",
            []
        ):

            content = message.get(
                "content",
                ""
            )

            if content.strip():
                texts.append(content)


if not texts:
    raise RuntimeError(
        "No training text found."
    )


# =========================
# Tokenizer
# =========================

tokenizer = Tokenizer()

tokenizer.build_vocab(texts)

tokenizer.save(VOCAB_FILE)

print(
    "Vocabulary size:",
    len(tokenizer)
)


# =========================
# Dataset
# =========================

dataset = AIDataset(
    DATASET_FILE,
    tokenizer,
    block_size=BLOCK_SIZE
)


if len(dataset) == 0:
    raise RuntimeError(
        "Dataset contains no valid training samples."
    )


# =========================
# Collate
# =========================

def collate_fn(batch):

    inputs = [
        item[0]
        for item in batch
    ]

    targets = [
        item[1]
        for item in batch
    ]

    inputs = torch.nn.utils.rnn.pad_sequence(
        inputs,
        batch_first=True,
        padding_value=tokenizer.vocab["<PAD>"]
    )

    targets = torch.nn.utils.rnn.pad_sequence(
        targets,
        batch_first=True,
        padding_value=tokenizer.vocab["<PAD>"]
    )

    return inputs, targets


# =========================
# DataLoader
# =========================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# =========================
# Model
# =========================

model = MyAI(
    vocab_size=len(tokenizer)
).to(DEVICE)


# =========================
# Optimizer
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Information
# =========================

print()
print("==============================")
print("MY-AI Training")
print("==============================")
print("Device:", DEVICE)
print("Training samples:", len(dataset))
print("Vocabulary:", len(tokenizer))
print("Epochs:", EPOCHS)
print("==============================")
print()


# =========================
# Training
# =========================

model.train()


for epoch in range(EPOCHS):

    total_loss = 0.0
    batches = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            y.reshape(-1),
            ignore_index=tokenizer.vocab["<PAD>"]
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    average_loss = (
        total_loss / batches
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
        f" - Loss: {average_loss:.4f}"
    )


# =========================
# Save model
# =========================

print()
print("Saving model...")

torch.save(
    model.state_dict(),
    MODEL_FILE
)


# =========================
# Verify files
# =========================

if not MODEL_FILE.exists():
    raise RuntimeError(
        "Model file was not created."
    )

if MODEL_FILE.stat().st_size == 0:
    raise RuntimeError(
        "Model file is empty."
    )

if not VOCAB_FILE.exists():
    raise RuntimeError(
        "Vocabulary file was not created."
    )


print()
print("==============================")
print("Training complete!")
print("==============================")

print(
    "Model file:",
    MODEL_FILE
)

print(
    "Model size:",
    MODEL_FILE.stat().st_size,
    "bytes"
)

print(
    "Vocabulary file:",
    VOCAB_FILE
)
