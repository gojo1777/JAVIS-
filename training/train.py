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


# -------------------------
# Settings
# -------------------------

BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Load training texts
# -------------------------

texts = []

with open(
    "../data/conversations.jsonl",
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        item = json.loads(line)

        for message in item["messages"]:
            texts.append(message["content"])


# -------------------------
# Build tokenizer
# -------------------------

tokenizer = Tokenizer()

tokenizer.build_vocab(texts)

tokenizer.save(
    "../data/vocab.json"
)

print("Vocabulary size:", len(tokenizer.vocab))


# -------------------------
# Dataset
# -------------------------

dataset = AIDataset(
    "../data/conversations.jsonl",
    tokenizer,
    block_size=256
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda batch: (
        torch.nn.utils.rnn.pad_sequence(
            [x for x, y in batch],
            batch_first=True,
            padding_value=tokenizer.vocab["<PAD>"]
        ),
        torch.nn.utils.rnn.pad_sequence(
            [y for x, y in batch],
            batch_first=True,
            padding_value=tokenizer.vocab["<PAD>"]
        )
    )
)


# -------------------------
# Model
# -------------------------

model = MyAI().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Training
# -------------------------

print("Device:", DEVICE)
print("Training samples:", len(dataset))

for epoch in range(EPOCHS):

    total_loss = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=tokenizer.vocab["<PAD>"]
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {total_loss:.4f}"
    )


# -------------------------
# Save
# -------------------------

torch.save(
    model.state_dict(),
    "../my_ai.pt"
)

print("Training complete!")
print("Model saved as my_ai.pt")
