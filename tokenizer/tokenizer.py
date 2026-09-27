import json
import re


class Tokenizer:

    def __init__(self):

        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
            "<user>",
            "<assistant>",
        ]

        self.vocab = {}
        self.id_to_token = {}

    # =========================
    # Build vocabulary
    # =========================

    def build_vocab(self, texts):

        token_set = set()

        pattern = (
            r"<user>|"
            r"<assistant>|"
            r"<PAD>|"
            r"<UNK>|"
            r"<BOS>|"
            r"<EOS>|"
            r"\w+|"
            r"[^\w\s]"
        )

        for text in texts:

            tokens = re.findall(
                pattern,
                text,
                re.UNICODE
            )

            token_set.update(tokens)

        all_tokens = (
            self.special_tokens
            + sorted(
                token_set
                - set(self.special_tokens)
            )
        )

        self.vocab = {
            token: i
            for i, token in enumerate(
                all_tokens
            )
        }

        self.id_to_token = {
            i: token
            for token, i in self.vocab.items()
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

        pattern = (
            r"<user>|"
            r"<assistant>|"
            r"<PAD>|"
            r"<UNK>|"
            r"<BOS>|"
            r"<EOS>|"
            r"\w+|"
            r"[^\w\s]"
        )

        tokens = re.findall(
            pattern,
            text,
            re.UNICODE
        )

        ids = [
            self.vocab.get(
                token,
                self.vocab["<UNK>"]
            )
            for token in tokens
        ]

        if add_bos:

            ids.insert(
                0,
                self.vocab["<BOS>"]
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

        tokens = [
            self.id_to_token.get(
                int(i),
                "<UNK>"
            )
            for i in ids
        ]

        return " ".join(tokens)

    # =========================
    # Length
    # =========================

    def __len__(self):

        return len(self.vocab)

    # =========================
    # Save
    # =========================

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
            int(i): token
            for token, i in self.vocab.items()
        }
