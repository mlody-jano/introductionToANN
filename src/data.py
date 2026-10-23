"""
Wczytanie przygotowanych danych Galaxy10 do PyTorcha.

Uruchom z głównego folderu projektu, żeby sprawdzić, czy wszystko działa:
    python src/data.py
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def to_tensor(X, mean, std):
    """Obrazy uint8 (N, H, W, C) -> float32 (N, C, H, W), znormalizowane."""
    tensor = torch.from_numpy(X)                        # (N, 64, 64, 3), uint8, 0-255
    tensor = tensor.permute(0, 3, 1, 2)                 # (N, 3, 64, 64)
    tensor = tensor.float() / 255.0                     # float32, 0-1

    mean_t = torch.from_numpy(mean).view(1, 3, 1, 1)    # średnie kanałów jako (1, 3, 1, 1)
    std_t = torch.from_numpy(std).view(1, 3, 1, 1)      # odchylenia kanałów jako (1, 3, 1, 1)
    tensor = (tensor - mean_t) / std_t                  # normalizacja każdego kanału
    return tensor


def load_data(path, batch_size=32):
    d = np.load(path)
    mean, std = d["mean"], d["std"]   # jedne wartości dla wszystkich zbiorów, policzone na train

    x_train = to_tensor(d["X_train"], mean, std)              # obrazy: (12330, 3, 64, 64)
    y_train = torch.from_numpy(d["y_train"]).long()           # etykiety: (12330,), typ int64
    train_ds = TensorDataset(x_train, y_train)                # pary (obraz, etykieta)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)   # partie po 64

    x_valid = to_tensor(d["X_val"], mean, std)
    y_valid = torch.from_numpy(d["y_val"]).long()
    valid_ds = TensorDataset(x_valid, y_valid)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    x_test = to_tensor(d["X_test"], mean, std)
    y_test = torch.from_numpy(d["y_test"]).long()
    test_ds = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, d["class_names"].tolist()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    path = "data/processed/galaxy10_64.npz"
    train_loader, valid_loader, test_loader, class_names = load_data(path)

    #kształty i typy całego zbioru treningowego
    x_train, y_train = train_loader.dataset.tensors
    print("Test 1")
    print("  obrazy:  ", x_train.shape, x_train.dtype)
    print("  etykiety:", y_train.shape, y_train.dtype)

    #czy normalizacja zadziałała
    x_valid = valid_loader.dataset.tensors[0]
    print("Test 2")
    print("  średnia kanałów (train):   ", x_train.mean(dim=(0, 2, 3)))
    print("  odchylenie kanałów (train):", x_train.std(dim=(0, 2, 3)))
    print("  średnia kanałów (val):     ", x_valid.mean(dim=(0, 2, 3)))

    #jedna partia i liczba partii
    xb, yb = next(iter(train_loader))
    print("Test 3")
    print("  jedna partia:", xb.shape, yb.shape)
    print("  liczba partii (train, val, test):", len(train_loader), len(valid_loader), len(test_loader))

    #podgląd jednego obrazu z partii
    d = np.load(path)
    mean = torch.from_numpy(d["mean"]).view(3, 1, 1)
    std = torch.from_numpy(d["std"]).view(3, 1, 1)
    img = xb[0] * std + mean           # odwrócenie normalizacji: z powrotem do zakresu 0-1
    img = img.permute(1, 2, 0)         # (3, 64, 64) -> (64, 64, 3), bo tak rysuje matplotlib
    plt.imshow(img.clamp(0, 1))        # clamp obcina wartości do 0-1 (błędy zaokrągleń)
    plt.title(class_names[yb[0].item()])
    plt.show()