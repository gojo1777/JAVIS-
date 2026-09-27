import json
import torch
from torch.utils.data import Dataset


class AIDataset(Dataset):

    def __init__(self, path, tokenizer, block_size=256):

        self.samples = []

        with open(path, "r", encoding="utf-8") as file:

            for line in file:

                item = json.loads(line)

                text = ""

                for message in item["messages"]:
                    text += message["content"] + "\n"

                tokens = tokenizer.encode(text)

                if len(tokens) >= 2:
                    tokens = tokens[:block_size + 1]
                    self.samples.append(tokens)

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
