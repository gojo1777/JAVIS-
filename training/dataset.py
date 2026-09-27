import json
import torch
from torch.utils.data import Dataset


class AIDataset(Dataset):

    def __init__(
        self,
        path,
        tokenizer,
        block_size=256
    ):

        self.samples = []

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                item = json.loads(line)

                messages = item.get(
                    "messages",
                    []
                )

                if not messages:
                    continue

                # =========================
                # Build conversation
                # =========================

                conversation = ""

                for message in messages:

                    role = message.get(
                        "role",
                        ""
                    )

                    content = message.get(
                        "content",
                        ""
                    ).strip()

                    if not content:
                        continue

                    if role == "user":

                        conversation += (
                            "<user>\n"
                            + content
                            + "\n"
                        )

                    elif role == "assistant":

                        conversation += (
                            "<assistant>\n"
                            + content
                            + "\n"
                        )

                if not conversation.strip():
                    continue

                # =========================
                # Tokenize
                # =========================

                tokens = tokenizer.encode(
                    conversation,
                    add_bos=True,
                    add_eos=True
                )

                if len(tokens) < 2:
                    continue

                # =========================
                # Limit sequence length
                # =========================

                tokens = tokens[
                    :block_size + 1
                ]

                # =========================
                # Store
                # =========================

                self.samples.append(
                    tokens
                )

    # =========================
    # Dataset length
    # =========================

    def __len__(self):

        return len(self.samples)

    # =========================
    # Get sample
    # =========================

    def __getitem__(self, index):

        tokens = self.samples[index]

        x = torch.tensor(
            tokens[:-1],
            dtype=torch.long
        )

        y = torch.tensor(
            tokens[1:],
            dtype=torch.long
        )

        return x, y
