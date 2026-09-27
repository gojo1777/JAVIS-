import json
from pathlib import Path


OUTPUT_FILE = Path("conversations.jsonl")


def add_conversation(user, assistant):

    item = {
        "messages": [
            {
                "role": "user",
                "content": user
            },
            {
                "role": "assistant",
                "content": assistant
            }
        ]
    }

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )


def main():

    print("MY-AI Dataset Builder")
    print("Type 'exit' to stop.\n")

    while True:

        user = input("User: ")

        if user.lower() == "exit":
            break

        assistant = input("AI: ")

        if assistant.lower() == "exit":
            break

        if not user.strip() or not assistant.strip():
            print("Both fields are required.\n")
            continue

        add_conversation(
            user.strip(),
            assistant.strip()
        )

        print("Saved.\n")


if __name__ == "__main__":
    main()
