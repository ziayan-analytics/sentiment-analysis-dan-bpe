from collections import Counter, defaultdict

class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = []

    def get_stats(self, vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def train(self, words):
        vocab = Counter()
        for w in words:
            vocab[" ".join(list(w)) + " </w>"] += 1

        while True:
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            vocab = self.merge_vocab(best, vocab)
            self.merges.append(best)

            if len(self.get_subwords(vocab)) >= self.vocab_size:
                break

        self.subwords = self.get_subwords(vocab)

    def get_subwords(self, vocab):
        subs = set()
        for word in vocab:
            for s in word.split():
                subs.add(s)
        return subs

    def encode_word(self, word):
        tokens = list(word)
        for a, b in self.merges:
            i = 0
            while i < len(tokens)-1:
                if tokens[i] == a and tokens[i+1] == b:
                    tokens[i:i+2] = [a+b]
                else:
                    i += 1
        return tokens
