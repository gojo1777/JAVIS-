import json
import re


class Tokenizer:
    def __init__(self):
        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
        ]

        self.vocab = {}
        self.id_to_token = {}

    def build_vocab(self, texts):
        token_set = set()

        for text in texts:
            tokens = re.findall(
                r"\w+|[^\w\s]",
                text,
                re.UNICODE
            )
            token_set.update(tokens)

        all_tokens = self.special_tokens + sorted(token_set)

        self.vocab = {
            token: i
            for i, token in enumerate(all_tokens)
        }

        self.id_to_token = {
            i: token
            for token, i in self.vocab.items()
        }

    def encode(self, text):
        tokens = re.findall(
            r"\w+|[^\w\s]",
            text,
            re.UNICODE
        )

        return [
            self.vocab.get(
                token,
                self.vocab["<UNK>"]
            )
            for token in tokens
        ]

    def decode(self, ids):
        tokens = [
            self.id_to_token.get(
                i,
                "<UNK>"
            )
            for i in ids
        ]

        return " ".join(tokens)

    def save(self, path):
        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.vocab,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self, path):
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:
            self.vocab = json.load(file)

        self.id_to_token = {
            int(i): token
            for token, i in self.vocab.items()
        }
