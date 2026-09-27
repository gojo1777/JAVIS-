import json
from pathlib import Path


# =========================
# Settings
# =========================

DATASET_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "conversations.jsonl"
)


# =========================
# Validation
# =========================

def validate_dataset():

    if not DATASET_FILE.exists():
        print("ERROR: Dataset file not found.")
        print(f"Expected: {DATASET_FILE}")
        return

    total = 0
    valid = 0
    errors = 0
    duplicates = 0

    seen = set()

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            total += 1

            # -------------------------
            # JSON check
            # -------------------------

            try:
                item = json.loads(line)

            except json.JSONDecodeError as error:

                print(
                    f"[ERROR] Line {line_number}: "
                    f"Invalid JSON"
                )

                print(
                    f"        {error}"
                )

                errors += 1
                continue

            # -------------------------
            # messages check
            # -------------------------

            if "messages" not in item:

                print(
                    f"[ERROR] Line {line_number}: "
                    f"Missing 'messages'"
                )

                errors += 1
                continue

            messages = item["messages"]

            if not isinstance(messages, list):

                print(
                    f"[ERROR] Line {line_number}: "
                    f"'messages' must be a list"
                )

                errors += 1
                continue

            if len(messages) < 2:

                print(
                    f"[ERROR] Line {line_number}: "
                    f"Conversation needs at least "
                    f"2 messages"
                )

                errors += 1
                continue

            # -------------------------
            # Message validation
            # -------------------------

            conversation_valid = True

            for message in messages:

                if not isinstance(message, dict):

                    print(
                        f"[ERROR] Line {line_number}: "
                        f"Invalid message format"
                    )

                    conversation_valid = False
                    break

                if "role" not in message:

                    print(
                        f"[ERROR] Line {line_number}: "
                        f"Message missing 'role'"
                    )

                    conversation_valid = False
                    break

                if "content" not in message:

                    print(
                        f"[ERROR] Line {line_number}: "
                        f"Message missing 'content'"
                    )

                    conversation_valid = False
                    break

                role = message["role"]
                content = message["content"]

                if role not in {
                    "user",
                    "assistant",
                    "system"
                }:

                    print(
                        f"[ERROR] Line {line_number}: "
                        f"Unknown role: {role}"
                    )

                    conversation_valid = False
                    break

                if not isinstance(content, str):

                    print(
                        f"[ERROR] Line {line_number}: "
                        f"Content must be text"
                    )

                    conversation_valid = False
                    break

                if not content.strip():

                    print(
                        f"[ERROR] Line {line_number}: "
                        f"Empty message"
                    )

                    conversation_valid = False
                    break

            if not conversation_valid:

                errors += 1
                continue

            # -------------------------
            # Duplicate check
            # -------------------------

            conversation_key = json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True
            )

            if conversation_key in seen:

                print(
                    f"[WARNING] Line {line_number}: "
                    f"Duplicate conversation"
                )

                duplicates += 1

            else:

                seen.add(conversation_key)

            valid += 1

    # =========================
    # Results
    # =========================

    print()
    print("==============================")
    print("MY-AI Dataset Validation")
    print("==============================")

    print(
        f"Total entries : {total}"
    )

    print(
        f"Valid entries : {valid}"
    )

    print(
        f"Errors        : {errors}"
    )

    print(
        f"Duplicates    : {duplicates}"
    )

    print("==============================")

    if errors == 0:

        print(
            "Dataset structure looks good."
        )

    else:

        print(
            "Dataset contains errors."
        )


# =========================
# Start
# =========================

if __name__ == "__main__":
    validate_dataset()
