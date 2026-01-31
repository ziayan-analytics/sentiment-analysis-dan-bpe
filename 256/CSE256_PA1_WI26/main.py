# models.py
from DANmodels import DAN
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW



from torch.utils.data import Dataset
import torch

class SubwordSentimentDataset(Dataset):
    def __init__(self, infile, examples, bpe, subword_to_idx, max_len=200, pad_token="<PAD>", unk_token="<UNK>"):
        self.max_len = max_len
        self.subword_to_idx = subword_to_idx
        self.pad_id = subword_to_idx[pad_token]
        self.unk_id = subword_to_idx[unk_token]

        xs = []
        ys = []
        for ex in examples:
            # ex.words is a list of words (already lowercased in read_sentiment_examples)
            sub_tokens = []
            for w in ex.words:
                sub_tokens.extend(bpe.encode_word(w))  # list of subwords/chars

            # map to ids
            ids = [subword_to_idx.get(t, self.unk_id) for t in sub_tokens]

            # truncate/pad
            ids = ids[:max_len]
            if len(ids) < max_len:
                ids = ids + [self.pad_id] * (max_len - len(ids))

            xs.append(ids)
            ys.append(ex.label)

        self.X = torch.tensor(xs, dtype=torch.long)
        self.y = torch.tensor(ys, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        if X.dtype != torch.long:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        if X.dtype != torch.long:
            X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')


    parser.add_argument("--bpe_vocab_size", type=int, default=1000)
    parser.add_argument("--subword_max_len", type=int, default=200)
    parser.add_argument("--subword_embed_dim", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    
    elif args.model == "DAN":

        print("\nRunning DAN with GloVe embeddings")

        from sentiment_data import read_word_embeddings

        print("\nRunning DAN with GloVe embeddings")

# load embeddings
        embs = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        embedding_layer = embs.get_initialized_embedding_layer(frozen=False)

        model = DAN(
            embedding_layer=embedding_layer,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout
        )


    # Convert SentimentDataset to index sequences
        from sentiment_data import read_sentiment_examples

        def convert_dataset(path, word_indexer):
            examples = read_sentiment_examples(path)
            X = []
            y = []
            for ex in examples:
                idxs = [word_indexer.index_of(w) if word_indexer.index_of(w) != -1 else 1 for w in ex.words]
                X.append(idxs)
                y.append(ex.label)
            return X, y

        train_X, train_y = convert_dataset("data/train.txt", embs.word_indexer)

        dev_X, dev_y = convert_dataset("data/dev.txt", embs.word_indexer)


        def pad_batch(X):
            max_len = max(len(x) for x in X)
            return torch.tensor([x + [0] * (max_len - len(x)) for x in X], dtype=torch.long)

        train_loader = DataLoader(list(zip(train_X, train_y)), batch_size=32, shuffle=True, collate_fn=lambda b: (pad_batch([x for x,_ in b]), torch.tensor([y for _,y in b])))
        dev_loader = DataLoader(list(zip(dev_X, dev_y)), batch_size=32, shuffle=False, collate_fn=lambda b: (pad_batch([x for x,_ in b]), torch.tensor([y for _,y in b])))

        train_acc, dev_acc = experiment(model, train_loader, dev_loader)


    elif args.model == "SUBWORDDAN":
        from sentiment_data import read_sentiment_examples
        from bpe import BPE
        from DANmodels import SubwordDAN
        

        print("\nRunning SUBWORDDAN (BPE subwords, random embeddings)")

    # 1) read train/dev
        train_examples = read_sentiment_examples("data/train.txt")
        dev_examples = read_sentiment_examples("data/dev.txt")

    # 2) train BPE on train words
        all_train_words = []
        for ex in train_examples:
            all_train_words.extend(ex.words)

        bpe = BPE(vocab_size=args.bpe_vocab_size)
        bpe.train(all_train_words)

    # 3) build subword vocab -> ids
        subword_to_idx = {"<PAD>": 0, "<UNK>": 1}
        for sw in bpe.subwords:
            if sw not in subword_to_idx:
                subword_to_idx[sw] = len(subword_to_idx)

        print(f"Subword vocab size = {len(subword_to_idx)} (target {args.bpe_vocab_size})")

    # 4) build datasets
        train_data = SubwordSentimentDataset(
            infile="data/train.txt",
            examples=train_examples,
            bpe=bpe,
            subword_to_idx=subword_to_idx,
            max_len=args.subword_max_len,
            pad_token="<PAD>",
            unk_token="<UNK>"
        )
        dev_data = SubwordSentimentDataset(
            infile="data/dev.txt",
            examples=dev_examples,
            bpe=bpe,
            subword_to_idx=subword_to_idx,
            max_len=args.subword_max_len,
            pad_token="<PAD>",
            unk_token="<UNK>"
        )

        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    # 5) init model
        model = SubwordDAN(
            vocab_size=len(subword_to_idx),
            embed_dim=args.subword_embed_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_idx=0
        )

    # 6) train/eval using existing experiment()
        train_acc_list, dev_acc_list = experiment(model, train_loader, dev_loader)

    # optional: save plots similar to BOW if you want




if __name__ == "__main__":
    main()
