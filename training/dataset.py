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

                text_parts = []

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

                    # Make roles visible to the model
                    text_parts.append(
                        f"<{role}>"
                    )

                    text_parts.append(
                        content
                    )

                if not text_parts:
                    continue

                text = "\n".join(
                    text_parts
                )

                tokens = tokenizer.encode(
                    text,
                    add_bos=True,
                    add_eos=True
                )

                if len(tokens) < 2:
                    continue

                # Keep one training sequence
                tokens = tokens[
                    :block_size + 1
                ]

                if len(tokens) >= 2:

                    self.samples.append(
                        tokens
                    )

    def __len__(self):

        return len(self.samples)

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
