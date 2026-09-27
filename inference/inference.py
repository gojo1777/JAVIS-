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

# Number of previous messages kept in memory
MAX_HISTORY_MESSAGES = 6

# Maximum tokens generated for one answer
MAX_NEW_TOKENS = 50

# Lower temperature = more predictable
TEMPERATURE = 0.7

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================
# Files
# =========================

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
# Math Solver
# =========================

def solve_math_if_present(prompt):
    """
    Detect simple arithmetic expressions and
    calculate them directly instead of asking
    the language model.
    """

    math_pattern = (
        r'(\d+(?:\.\d+)?'
        r'\s*[\+\-\*/]\s*'
        r'\d+(?:\.\d+)?'
        r'(?:\s*[\+\-\*/]\s*'
        r'\d+(?:\.\d+)?)*'
        r')'
    )

    match = re.search(
        math_pattern,
        prompt
    )

    if not match:
        return None

    expression = match.group(1).strip()

    try:

        # Only allow numbers and arithmetic operators
        if not re.fullmatch(
            r'[\d\s\+\-\*/\.]+',
            expression
        ):
            return None

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )

        if (
            isinstance(result, float)
            and result.is_integer()
        ):
            result = int(result)

        return f"{expression} = {result}"

    except Exception:

        return None


# =========================
# Build Conversation Prompt
# =========================

def build_prompt(
    current_prompt,
    history
):
    """
    Build the prompt using previous
    conversation messages.
    """

    prompt_text = ""

    # Previous conversation
    for role, content in history[
        -MAX_HISTORY_MESSAGES:
    ]:

        prompt_text += (
            f"<{role}>\n"
            f"{content}\n"
        )

    # Current user question
    prompt_text += (
        "<user>\n"
        + current_prompt
        + "\n"
        + "<assistant>\n"
    )

    return prompt_text


# =========================
# Generate
# =========================

def generate(
    prompt,
    history=None,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE
):

    if history is None:
        history = []

    # =========================
    # Build conversation
    # =========================

    prompt_text = build_prompt(
        prompt,
        history
    )

    # =========================
    # Tokenize
    # =========================

    ids = tokenizer.encode(
        prompt_text,
        add_bos=True
    )

    # Keep only latest context
    ids = ids[
        -BLOCK_SIZE:
    ]

    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    eos_id = tokenizer.vocab.get(
        "<EOS>"
    )

    user_id = tokenizer.vocab.get(
        "<user>"
    )

    assistant_id = tokenizer.vocab.get(
        "<assistant>"
    )

    # =========================
    # Generate
    # =========================

    with torch.no_grad():

        for _ in range(
            max_new_tokens
        ):

            input_ids = idx[
                :, -BLOCK_SIZE:
            ]

            logits = model(
                input_ids
            )

            next_logits = logits[
                :, -1, :
            ]

            # Temperature
            temperature = max(
                temperature,
                0.1
            )

            next_logits = (
                next_logits
                / temperature
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

            # Stop at EOS
            if token_id == eos_id:
                break

            # Stop if model starts another user turn
            if token_id == user_id:
                break

    # =========================
    # Get generated tokens
    # =========================

    all_ids = idx[
        0
    ].tolist()

    # Find the LAST assistant token
    if assistant_id in all_ids:

        assistant_position = (
            len(all_ids)
            - 1
            - all_ids[::-1].index(
                assistant_id
            )
        )

        generated_ids = all_ids[
            assistant_position + 1:
        ]

    else:

        # Fallback
        generated_ids = all_ids[
            len(ids):
        ]

    # =========================
    # Clean special tokens
    # =========================

    special_ids = {
        token_id
        for token_id in [
            eos_id,
            user_id,
            assistant_id
        ]
        if token_id is not None
    }

    clean_ids = []

    for token_id in generated_ids:

        if token_id in special_ids:
            break

        clean_ids.append(
            token_id
        )

    # =========================
    # Decode
    # =========================

    generated_text = tokenizer.decode(
        clean_ids
    )

    return generated_text.strip()


# =========================
# Clean Response
# =========================

def clean_response(response):

    response = response.strip()

    # Remove accidental role markers
    response = re.sub(
        r'<assistant>\s*',
        '',
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r'<user>.*',
        '',
        response,
        flags=re.IGNORECASE
    )

    return response.strip()


# =========================
# Chat Loop
# =========================

def main():

    print(
        "==============================="
    )

    print(
        " JAVIS AI Assistant Loaded "
    )

    print(
        "==============================="
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Memory: last "
        f"{MAX_HISTORY_MESSAGES} messages"
    )

    print(
        "Type 'exit' to stop."
    )

    print()

    # =========================
    # Conversation Memory
    # =========================

    history = []

    # =========================
    # Chat
    # =========================

    while True:

        try:

            prompt = input(
                "You: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print()
            break

        # =========================
        # Exit
        # =========================

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        # =========================
        # Clear memory
        # =========================

        if prompt.lower() in {
            "clear",
            "clear memory",
            "forget",
            "reset"
        }:

            history.clear()

            print(
                "AI: Conversation memory cleared."
            )

            print()

            continue

        # =========================
        # Math Engine
        # =========================

        math_result = (
            solve_math_if_present(
                prompt
            )
        )

        if math_result:

            response = math_result

        else:

            # =========================
            # AI Generation
            # =========================

            response = generate(
                prompt,
                history=history
            )

        # =========================
        # Clean response
        # =========================

        response = clean_response(
            response
        )

        # Empty response fallback
        if not response:

            response = (
                "මට ඒකට හොඳ උත්තරයක් "
                "දෙන්න බැරි වුණා."
            )

        # =========================
        # Display
        # =========================

        print(
            "AI:",
            response
        )

        print()

        # =========================
        # Save conversation
        # =========================

        history.append(
            (
                "user",
                prompt
            )
        )

        history.append(
            (
                "assistant",
                response
            )
        )

        # =========================
        # Limit memory
        # =========================

        if len(history) > MAX_HISTORY_MESSAGES:

            history = history[
                -MAX_HISTORY_MESSAGES:
            ]


# =========================
# Start
# =========================

if __name__ == "__main__":

    main()
