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
        tokens = []

        for text in texts:
            # Sinhala + English + numbers + punctuation
            words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            tokens.extend(words)

        unique_tokens = sorted(set(tokens))

        all_tokens = self.special_tokens + unique_tokens

        self.vocab = {
            token: i for i, token in enumerate(all_tokens)
        }

        self.id_to_token = {
            i: token for token, i in self.vocab.items()
        }

    def encode(self, text):
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

        return [
            self.vocab.get(word, self.vocab["<UNK>"])
            for word in words
        ]

    def decode(self, ids):
        tokens = [
            self.id_to_token.get(i, "<UNK>")
            for i in ids
        ]

        return " ".join(tokens)
