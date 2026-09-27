import json
import re
from pathlib import Path


class Tokenizer:

    def __init__(self):

        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
            "<user>",
            "<assistant>",
            "<system>",
        ]

        self.vocab = {}
        self.id_to_token = {}

    # =========================
    # Tokenize
    # =========================

    def tokenize(self, text):

        return re.findall(
            r"<[^>]+>|\w+|[^\w\s]",
            text,
            re.UNICODE
        )

    # =========================
    # Build vocabulary
    # =========================

    def build_vocab(self, texts):

        token_set = set()

        for text in texts:

            tokens = self.tokenize(text)

            token_set.update(tokens)

        normal_tokens = sorted(
            token
            for token in token_set
            if token not in self.special_tokens
        )

        all_tokens = (
            self.special_tokens
            + normal_tokens
        )

        self.vocab = {
            token: index
            for index, token
            in enumerate(all_tokens)
        }

        self.id_to_token = {
            index: token
            for token, index
            in self.vocab.items()
        }

    # =========================
    # Encode
    # =========================

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False
    ):

        tokens = self.tokenize(text)

        ids = []

        if add_bos:

            ids.append(
                self.vocab["<BOS>"]
            )

        for token in tokens:

            ids.append(
                self.vocab.get(
                    token,
                    self.vocab["<UNK>"]
                )
            )

        if add_eos:

            ids.append(
                self.vocab["<EOS>"]
            )

        return ids

    # =========================
    # Decode
    # =========================

    def decode(self, ids):

        tokens = []

        for token_id in ids:

            token = self.id_to_token.get(
                int(token_id),
                "<UNK>"
            )

            if token in {
                "<PAD>",
                "<BOS>",
                "<EOS>",
            }:
                continue

            tokens.append(token)

        text = ""

        punctuation = {
            ".",
            ",",
            "!",
            "?",
            ":",
            ";",
            "%",
            ")",
            "]",
            "}",
        }

        opening = {
            "(",
            "[",
            "{",
        }

        for token in tokens:

            if not text:

                text = token

            elif token in punctuation:

                text += token

            elif text[-1:] in opening:

                text += token

            else:

                text += " " + token

        return text

    # =========================
    # Save
    # =========================

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

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

    # =========================
    # Load
    # =========================

    def load(self, path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            self.vocab = json.load(file)

        self.id_to_token = {
            int(index): token
            for token, index
            in self.vocab.items()
        }

    # =========================
    # Vocabulary size
    # =========================

    def __len__(self):

        return len(self.vocab)
