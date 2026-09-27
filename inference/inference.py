import re
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
# Math Solver Helper (100% Accuracy Engine)
# =========================

def solve_math_if_present(prompt):
    """
    ප්‍රශ්නයේ ගණිතමය සමීකරණයක් (+, -, *, /) ඇත්නම්
    එය සොයාගෙන Python Evaluator එක හරහා 100% නිවැරදි උත්තරය සාදයි.
    """
    # Regex Pattern for basic math expressions (e.g., 5+5555, 10 * 20, 100/5)
    math_pattern = r'(\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*\d+(?:\.\d+)?)*)'
    match = re.search(math_pattern, prompt)

    if match:
        expression = match.group(1).strip()
        try:
            # 안전하게 expression එක evaluate කිරීම
            result = eval(expression)
            
            # පූර්ණ සංඛ්‍යාවක් නම් decimal අයින් කිරීම (.0)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
                
            return f"{expression} = {result}"
        except Exception:
            return None
            
    return None


# =========================
# Generate (LLM Model Output)
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
# Chat Loop
# =========================

def main():

    print("===============================")
    print(" JAVIS AI Assistant Loaded ")
    print("===============================")
    print("Type 'exit' to stop.\n")

    while True:

        prompt = input(
            "You: "
        ).strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        # 1. පළමුව Math Engine එකෙන් Check කිරීම
        math_result = solve_math_if_present(prompt)

        if math_result:
            response = math_result
        else:
            # 2. ගණනක් නොවේ නම් JAVIS AI Model එකෙන් Answer එක Generate කිරීම
            response = generate(prompt)

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
