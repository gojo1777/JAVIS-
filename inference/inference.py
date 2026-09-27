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

            token_id = (
                next_token.item()
            )

            if token_id == tokenizer.vocab.get(
                "<EOS>"
            ):
                break

    generated_ids = idx[
        0
    ].tolist()

    generated_text = tokenizer.decode(
        generated_ids
    )

    if "<assistant>" in generated_text:

        generated_text = (
            generated_text.split(
                "<assistant>",
                1
            )[1]
        )

    if "<user>" in generated_text:

        generated_text = (
            generated_text.split(
                "<user>",
                1
            )[0]
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


if __name__ == "__main__":
    main()
