import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("../model")

from transformer import MyAI


# -------------------------
# Settings
# -------------------------

BATCH_SIZE = 4
LEARNING_RATE = 3e-4
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Dataset
# -------------------------

class ConversationDataset(Dataset):

    def __init__(self, path):

        self.samples = []

        with open(path, "r", encoding="utf-8") as file:

            for line in file:

                item = json.loads(line)

                messages = item["messages"]

                text = ""

                for message in messages:
                    text += message["content"] + "\n"

                self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        return self.samples[index]


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
print("Starting training...")


dataset = ConversationDataset(
    "../data/conversations.jsonl"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


for epoch in range(EPOCHS):

    total_loss = 0

    for text in loader:

        # Temporary character-level conversion
        tokens = []

        for sentence in text:
            tokens.append([
                ord(char) % 16000
                for char in sentence
            ])

        max_length = min(
            max(len(x) for x in tokens),
            256
        )

        x = torch.tensor(
            [
                t[:max_length]
                for t in tokens
            ],
            dtype=torch.long
        ).to(DEVICE)

        if x.size(1) < 2:
            continue

        input_ids = x[:, :-1]
        target_ids = x[:, 1:]

        logits = model(input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
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
# Save model
# -------------------------

torch.save(
    model.state_dict(),
    "my_ai.pt"
)

print("Training finished.")
print("Model saved as my_ai.pt")
