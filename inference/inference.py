import sys
from pathlib import Path

import torch


# =========================
# Project paths
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "model"
TOKENIZER_DIR = ROOT_DIR / "tokenizer"
DATA_DIR = ROOT_DIR / "data"

sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(TOKENIZER_DIR))


from transformer import MyAI
from tokenizer import Tokenizer


# =========================
# Settings
# =========================

BLOCK_SIZE = 256

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


MODEL_FILE = ROOT_DIR / "my_ai.pt"
VOCAB_FILE = DATA_DIR / "vocab.json"


# =========================
# Load tokenizer
# =========================

tokenizer = Tokenizer()

tokenizer.load(
    VOCAB_FILE
)


# =========================
# Load model
# =========================

model = MyAI(
    vocab_size=len(tokenizer)
).to(DEVICE)


state_dict = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict
)

model.eval()


# =========================
# Generate
# =========================

def generate(
    prompt,
    max_new_tokens=50,
    temperature=0.8
):

    prompt_text = (
        "<user>\n"
        + prompt
        + "\n"
        + "<assistant>\n"
    )

    ids = tokenizer.encode(
        prompt_text,
        add_bos=True
    )

    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    assistant_id = tokenizer.vocab.get(
        "<assistant>"
    )

    eos_id = tokenizer.vocab.get(
        "<EOS>"
    )

    with torch.no_grad():

        for _ in range(max_new_tokens):

            input_ids = idx[
                :, -BLOCK_SIZE:
            ]

            logits = model(
                input_ids
            )

            next_logits = logits[
                :, -1, :
            ]

            next_logits = (
                next_logits / temperature
            )

            probabilities = torch.softmax(
                next_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            idx = torch.cat(
                [
                    idx,
                    next_token
                ],
                dim=1
            )

            token_id = next_token.item()

            if token_id == eos_id:
                break

    generated_ids = idx[
        0
    ].tolist()

    # =========================
    # Get tokens after assistant
    # =========================

    if assistant_id in generated_ids:

        assistant_position = (
            len(generated_ids)
            - 1
            - generated_ids[::-1].index(
                assistant_id
            )
        )

        generated_ids = generated_ids[
            assistant_position + 1:
        ]

    # =========================
    # Stop at special tokens
    # =========================

    clean_ids = []

    for token_id in generated_ids:

        if token_id in {
            eos_id,
            tokenizer.vocab.get("<user>"),
            tokenizer.vocab.get("<assistant>"),
        }:

            break

        clean_ids.append(token_id)

    # =========================
    # Decode
    # =========================

    generated_text = tokenizer.decode(
        clean_ids
    )

    return generated_text.strip()


# =========================
# Chat
# =========================

def main():

    print(
        "MY-AI Prototype"
    )

    print(
        "Type 'exit' to stop."
    )

    print()

    while True:

        prompt = input(
            "You: "
        ).strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        response = generate(
            prompt
        )

        print(
            "AI:",
            response
        )

        print()


# =========================
# Start
# =========================

if __name__ == "__main__":

    main()
