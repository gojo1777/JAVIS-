import json
from pathlib import Path


# =========================
# Dataset path
# =========================

ROOT_DIR = (
    Path(__file__).resolve().parent.parent
)

DATASET_FILE = (
    ROOT_DIR
    / "data"
    / "conversations.jsonl"
)


# =========================
# Validate dataset
# =========================

def validate():

    if not DATASET_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}"
        )

    total = 0
    valid = 0

    seen = set()
    duplicates = 0

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            1
        ):

            line = line.strip()

            if not line:
                continue

            total += 1

            # =========================
            # JSON check
            # =========================

            try:

                item = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {error}"
                )

            # =========================
            # Messages check
            # =========================

            messages = item.get(
                "messages"
            )

            if not isinstance(
                messages,
                list
            ) or not messages:

                raise ValueError(
                    f"Line {line_number}: "
                    "'messages' must be a "
                    "non-empty list."
                )

            # =========================
            # Message check
            # =========================

            for message in messages:

                if not isinstance(
                    message,
                    dict
                ):

                    raise ValueError(
                        f"Line {line_number}: "
                        "Invalid message."
                    )

                role = message.get(
                    "role"
                )

                content = message.get(
                    "content"
                )

                if role not in {
                    "user",
                    "assistant"
                }:

                    raise ValueError(
                        f"Line {line_number}: "
                        f"Invalid role: {role}"
                    )

                if not isinstance(
                    content,
                    str
                ) or not content.strip():

                    raise ValueError(
                        f"Line {line_number}: "
                        "Message content is empty."
                    )

            # =========================
            # Duplicate check
            # =========================

            signature = json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True
            )

            # Duplicates are allowed on purpose: hand-written Sinhala chats
            # are repeated (CUSTOM_REPEAT) so the model learns them.
            if signature in seen:
                duplicates += 1

            seen.add(signature)

            valid += 1

    # =========================
    # Final check
    # =========================

    if valid == 0:

        raise ValueError(
            "Dataset contains no valid "
            "conversations."
        )

    print(
        "Dataset validation passed."
    )

    print(
        f"Total entries: {total}"
    )

    print(
        f"Valid conversations: {valid}"
    )

    print(
        f"Repeated (intentional) rows: {duplicates}"
    )


# =========================
# Main
# =========================

if __name__ == "__main__":

    validate()
