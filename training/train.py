import json
import sys
import time
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
# Settings (Optimized for Fast Training on GitHub Actions)
# =========================

BATCH_SIZE = 16           # Batch size වැඩි කර Speed එක වැඩි කරන ලදී
LEARNING_RATE = 3e-4
EPOCHS = 5                # 300 තිබූ Epochs ගණන 5 දක්වා අඩු කරන ලදී
BLOCK_SIZE = 256
GRAD_CLIP = 1.0

SAMPLE_EVERY = 1          # සැම Epoch එකකදීම progress එක බලන්න
LOG_EVERY = 1             # print/log loss every N epochs

SAMPLE_PROMPTS = [
    "ඔයා කවුද?",
    "කොහොමද?",
    "python gana kiyaham"
]

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
BEST_MODEL_FILE = ROOT_DIR / "my_ai_best.pt"
LOG_FILE = ROOT_DIR / "training_log.txt"


# =========================
# Logging helper
# =========================

log_lines = []


def log(message):

    print(message)
    log_lines.append(message)

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(message + "\n")


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
# Sample generation (for progress checks during training)
# =========================

@torch.no_grad()
def generate_sample(prompt, max_new_tokens=40):

    model.eval()

    prompt_text = "<user>\n" + prompt + "\n<assistant>\n"

    ids = tokenizer.encode(prompt_text, add_bos=True)
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    eos_id = tokenizer.vocab.get("<EOS>")
    user_id = tokenizer.vocab.get("<user>")

    for _ in range(max_new_tokens):

        input_ids = idx[:, -BLOCK_SIZE:]
        logits = model(input_ids)
        next_logits = logits[:, -1, :]

        next_token = torch.argmax(
            next_logits, dim=-1, keepdim=True
        )

        idx = torch.cat([idx, next_token], dim=1)

        token_id = next_token.item()

        if token_id in (eos_id, user_id):
            break

    generated_ids = idx[0].tolist()[len(ids):]

    clean_ids = []
    for token_id in generated_ids:
        if token_id in (eos_id, user_id):
            break
        clean_ids.append(token_id)

    model.train()

    return tokenizer.decode(clean_ids).strip()


# =========================
# Information
# =========================

log("==============================")
log("MY-AI Fast Training")
log("==============================")
log(f"Device: {DEVICE}")
log(f"Training samples: {len(dataset)}")
log(f"Vocabulary: {len(tokenizer)}")
log(f"Epochs: {EPOCHS}")
log("==============================")
log("")


# =========================
# Training
# =========================

model.train()

best_loss = float("inf")
start_time = time.time()

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

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )

        optimizer.step()

        total_loss += loss.item()
        batches += 1

    average_loss = total_loss / batches

    if (epoch + 1) % LOG_EVERY == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        log(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {average_loss:.4f} "
            f"- Elapsed: {elapsed:.1f}s"
        )

    # Save best checkpoint
    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), BEST_MODEL_FILE)

    # Periodic Qualitative Check
    if (epoch + 1) % SAMPLE_EVERY == 0 or epoch == EPOCHS - 1:
        for prompt in SAMPLE_PROMPTS:
            sample = generate_sample(prompt)
            log(f"    [sample] {prompt!r} -> {sample!r}")
        log("")


# =========================
# Save final model
# =========================

log("")
log("Saving final model...")

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


log("")
log("==============================")
log("Training complete!")
log("==============================")
log(f"Final model file: {MODEL_FILE}")
log(f"Best model file (lowest loss): {BEST_MODEL_FILE} (loss {best_loss:.4f})")
log(f"Model size: {MODEL_FILE.stat().st_size} bytes")
log(f"Vocabulary file: {VOCAB_FILE}")
log(f"Training log: {LOG_FILE}")
