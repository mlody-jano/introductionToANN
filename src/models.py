import torch
from torch import nn

# nazwa funkcji aktywacji (z wiersza poleceń) -> klasa z PyTorcha
ACTIVATIONS = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def classifier_layers(n_in, hidden, n_classes, activation, dropout):
    """Warstwy w pełni połączone: n_in -> kolejne warstwy ukryte z listy hidden -> n_classes."""
    layers = []
    for n_out in hidden:
        layers.append(nn.Linear(n_in, n_out))
        layers.append(ACTIVATIONS[activation]())    # nawiasy () tworzą nową warstwę aktywacji
        layers.append(nn.Dropout(dropout))
        n_in = n_out                                # wyjście tej warstwy = wejście następnej
    layers.append(nn.Linear(n_in, n_classes))       # warstwa wyjściowa: logity, bez softmax
    return layers


class MLP(nn.Module):
    """
    sieć w pełni połączona: obraz spłaszczony do wektora -> warstwy ukryte -> 10 klas.
    hidden = lista liczby neuronów w kolejnych warstwach ukrytych; domyślnie (512, 256).
    """

    def __init__(self, n_classes=10, img_size=64, hidden=(512, 256), activation="relu", dropout=0.3):
        super().__init__()
        n_inputs = 3 * img_size * img_size          # 3 kolory x wszystkie piksele
        self.net = nn.Sequential(
            nn.Flatten(),                           # (N, 3, H, W) -> (N, 3*H*W)
            *classifier_layers(n_inputs, hidden, n_classes, activation, dropout),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """
    sieć konwolucyjna: n_blocks bloków (filtry 3x3 + opcjonalnie BatchNorm + aktywacja + pooling),
    w każdym bloku 2 razy więcej filtrów niż w poprzednim na końcu klasyfikator jak w MLP.
    Domyślnie: 3 bloki (32, 64, 128 filtrów) i jedna warstwa ukryta z 256 neuronami.
    """

    def __init__(self, n_classes=10, img_size=64, n_blocks=3, filters=32, hidden=(256,),
                 activation="relu", dropout=0.3, batchnorm=False):
        super().__init__()
        size = img_size // 2 ** n_blocks            # każdy pooling zmniejsza obraz o połowę
        if size < 1:
            raise ValueError(f"za dużo bloków ({n_blocks}) dla obrazu {img_size}x{img_size}")

        layers = []
        channels = 3                                # na wejściu 3 kanały kolorów
        for b in range(n_blocks):
            out_channels = filters * 2 ** b         # np. 32, 64, 128, ...
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(ACTIVATIONS[activation]())
            layers.append(nn.MaxPool2d(2))
            channels = out_channels                 # wyjście tego bloku = wejście następnego
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),                           # (N, kanały, size, size) -> (N, kanały*size*size)
            *classifier_layers(channels * size * size, hidden, n_classes, activation, dropout),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(cfg):
    # tworzy sieć według ustawień przebiegu (słownik z config.json albo z wiersza poleceń).
    # brakujące ustawienia (np. ze starszych przebiegów) dostają wartości domyślne modelu bazowego.
    common = dict(img_size=cfg.get("size", 64), activation=cfg.get("activation", "relu"),
                  dropout=cfg.get("dropout", 0.3))
    if cfg.get("model", "cnn") == "mlp":
        return MLP(hidden=cfg.get("hidden") or (512, 256), **common)
    return CNN(n_blocks=cfg.get("blocks", 3), filters=cfg.get("filters", 32),
               hidden=cfg.get("hidden") or (256,), batchnorm=cfg.get("batchnorm", False), **common)


def count_parameters(model):
    # liczba wszystkich wag i biasów, których sieć się uczy.
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # kilka konfiguracji: liczba parametrów i kształt wyjścia dla sztucznej partii 64 obrazów
    configs = [
        {"model": "mlp"},                                   # MLP bazowy: 6 425 866
        {"model": "mlp", "hidden": [256]},                  # jedna warstwa ukryta
        {"model": "cnn"},                                   # CNN bazowy: 2 193 226
        {"model": "cnn", "blocks": 4},                      # 4 bloki: mniejsza mapa na końcu
        {"model": "cnn", "filters": 16},                    # o połowę mniej filtrów
        {"model": "cnn", "batchnorm": True, "activation": "elu"},
        {"model": "cnn", "size": 128},                      # obrazy 128x128
    ]
    for cfg in configs:
        model = build_model(cfg)
        size = cfg.get("size", 64)
        out = model(torch.zeros(64, 3, size, size))
        print(f"{str(cfg):<55} parametry: {count_parameters(model):>9,}   wyjście: {tuple(out.shape)}")