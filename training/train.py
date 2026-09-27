import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =========================
# Project paths
# =========================

ROOT_DIR = (
    Path(__file__).resolve().parent.parent
)

MODEL_DIR = ROOT_DIR / "model"
TOKENIZER_DIR = ROOT_DIR / "tokenizer"
DATA_DIR = ROOT_DIR / "data"


sys.path.insert(
    0,
    str(MODEL_DIR)
)

sys.path.insert(
    0,
    str(TOKENIZER_DIR)
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent)
)


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
# Dataset file
# =========================

DATASET_FILE = (
    DATA_DIR / "conversations.jsonl"
)

VOCAB_FILE = (
    DATA_DIR / "vocab.json"
)

MODEL_FILE = (
    ROOT_DIR / "my_ai.pt"
)


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

                texts.append(
                    content
                )


# =========================
# Tokenizer
# =========================

tokenizer = Tokenizer()

tokenizer.build_vocab(
    texts
)

tokenizer.save(
    VOCAB_FILE
)


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
# Collate function
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

    inputs = (
        torch.nn.utils.rnn.pad_sequence(
            inputs,
            batch_first=True,
            padding_value=(
                tokenizer.vocab["<PAD>"]
            )
        )
    )

    targets = (
        torch.nn.utils.rnn.pad_sequence(
            targets,
            batch_first=True,
            padding_value=(
                tokenizer.vocab["<PAD>"]
            )
        )
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
