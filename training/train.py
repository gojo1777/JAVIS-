import json
import sys

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader


sys.path.append("../model")
sys.path.append("../tokenizer")


from transformer import MyAI
from tokenizer import Tokenizer
from dataset import AIDataset


# =========================
# Settings
# =========================

BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EPOCHS = 10

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================
# Load texts
# =========================

texts = []


with open(
    "../data/conversations.jsonl",
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        item = json.loads(line)

        for message in item["messages"]:

            texts.append(
                message["content"]
            )


# =========================
# Tokenizer
# =========================

tokenizer = Tokenizer()

tokenizer.build_vocab(texts)

tokenizer.save(
    "../data/vocab.json"
)


print(
    "Vocabulary size:",
    len(tokenizer.vocab)
)


# =========================
# Dataset
# =========================

dataset = AIDataset(
    "../data/conversations.jsonl",
    tokenizer,
    block_size=256
)


def collate_fn(batch):

    inputs = [
        x for x, y in batch
    ]

    targets = [
        y for x, y in batch
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
    vocab_size=len(tokenizer.vocab)
).to(DEVICE)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# Training
# =========================

print(
    "Device:",
    DEVICE
)

print(
    "Training samples:",
    len(dataset)
)

print(
    "Starting training..."
)


for epoch in range(EPOCHS):

    total_loss = 0.0

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

            ignore_index=(
                tokenizer.vocab["<PAD>"]
            )
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {total_loss:.4f}"
    )


# =========================
# Save model
# =========================

torch.save(
    model.state_dict(),
    "../my_ai.pt"
)


print(
    "Training complete!"
)

print(
    "Model saved as my_ai.pt"
)
