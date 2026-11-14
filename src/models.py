import torch
from torch import nn


class MLP(nn.Module):
    # sieć w pełni połączona: obraz spłaszczony do wektora -> 2 warstwy ukryte -> 10 klas

    def __init__(self, n_inputs=3 * 64 * 64, n_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()                 # (N, 3, 64, 64) -> (N, 12288)
        self.hidden1 = nn.Linear(n_inputs, 512)     # warstwa ukryta 1: 512 neuronów
        self.hidden2 = nn.Linear(512, 256)          # warstwa ukryta 2: 256 neuronów
        self.output = nn.Linear(256, n_classes)     # warstwa wyjściowa: 10 neuronów, po jednym na klasę
        self.act = nn.ReLU()                        # funkcja aktywacji
        self.dropout = nn.Dropout(0.3)              # regularyzacja: losowo wyłącza 30% neuronów

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(self.act(self.hidden1(x)))
        x = self.dropout(self.act(self.hidden2(x)))
        return self.output(x)                       # logity, bez softmax (robi go funkcja straty)


class CNN(nn.Module):
    # sieć konwolucyjna: 3 bloki (filtry + ReLU + pooling), potem klasyfikator jak w MLP

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32 filtry 3x3 na 3 kanałach kolorów
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 filtry 3x3 na 32 mapach cech
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 filtrów 3x3 na 64 mapach cech
        self.pool = nn.MaxPool2d(2)  # zmniejsza każdą mapę o połowę
        self.act = nn.ReLU()
        self.flatten = nn.Flatten()  # (N, 128, 8, 8) -> (N, 8192)
        self.hidden = nn.Linear(128 * 8 * 8, 256)  # warstwa ukryta klasyfikatora
        self.output = nn.Linear(256, n_classes)  # 10 neuronów, po jednym na klasę
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):  # x: (N, 3, 64, 64)
        x = self.pool(self.act(self.conv1(x)))  # blok 1 -> (N, 32, 32, 32)
        x = self.pool(self.act(self.conv2(x)))  # blok 2 -> (N, 64, 16, 16)
        x = self.pool(self.act(self.conv3(x)))  # blok 3 -> (N, 128, 8, 8)
        x = self.flatten(x)  # -> (N, 8192)
        x = self.dropout(self.act(self.hidden(x)))  # -> (N, 256)
        return self.output(x)  # logity -> (N, 10)


def count_parameters(model):
    # liczba wszystkich wag i biasów, których sieć się uczy.
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from data import load_data

    torch.manual_seed(0)
    mlp, cnn = MLP(), CNN()
    print("parametry MLP:", count_parameters(mlp))
    print("parametry CNN:", count_parameters(cnn))
    print("  w tym warstwy konwolucyjne:", sum(count_parameters(c) for c in (cnn.conv1, cnn.conv2, cnn.conv3)))

    train_loader, _, _, class_names = load_data("data/processed/galaxy10_64.npz")
    xb, yb = next(iter(train_loader))

    # kształt danych po każdym bloku CNN: to samo co w forward tylko z wypisywaniem
    x = xb
    print("\nwejście:     ", x.shape)
    x = cnn.pool(cnn.act(cnn.conv1(x)))
    print("po bloku 1:  ", x.shape)
    x = cnn.pool(cnn.act(cnn.conv2(x)))
    print("po bloku 2:  ", x.shape)
    x = cnn.pool(cnn.act(cnn.conv3(x)))
    print("po bloku 3:  ", x.shape)
    x = cnn.flatten(x)
    print("po flatten:  ", x.shape)
    logits = cnn(xb)
    print("wyjście CNN: ", logits.shape)

    loss = nn.CrossEntropyLoss()(logits, yb)
    print("\nstrata nienauczonej CNN:", loss.item(), "(oczekiwane ok. ln(10) = 2.30)")
