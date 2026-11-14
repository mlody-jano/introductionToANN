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


def load_data(path, batch_size=64):
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


def augment_batch(xb):
    """
    Augmentacja partii obrazów (tylko przy treningu): każdy obraz osobno obracany o 0, 90, 180
    albo 270 stopni i w połowie przypadków odbijany lustrzanie.
    """
    n = xb.shape[0]
    turns = torch.randint(0, 4, (n,), device=xb.device)       # ile razy obrócić o 90 stopni (0-3)
    flip = torch.rand(n, device=xb.device) < 0.5              # True dla obrazów do odbicia
    out = torch.empty_like(xb)
    for k in range(4):
        sel = turns == k                                      # obrazy, które obracamy k razy
        out[sel] = torch.rot90(xb[sel], k, dims=(2, 3))       # obrót w płaszczyźnie (wiersze, kolumny)
    out[flip] = torch.flip(out[flip], dims=(3,))              # odbicie lewo-prawo
    return out


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
    torch.manual_seed(1)
    versions = augment_batch(xb[:1].repeat(8, 1, 1, 1))     # 8 kopii pierwszego obrazu, każda losowo zmieniona
    versions[0] = xb[0]                                     # pierwsza zostaje oryginałem
    fig, axes = plt.subplots(1, 8, figsize=(12, 1.9), dpi=120)
    for ax, v in zip(axes, versions):
        ax.imshow((v * std + mean).permute(1, 2, 0).clamp(0, 1))
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("oryginał", fontsize=9)
    fig.tight_layout()
    fig.savefig("results/figures/augmentation_examples.png")
    print("Test 4: zapisano results/figures/augmentation_examples.png")

    # --- Test 5: podgląd jednego obrazu z partii
    plt.figure()
    img = xb[0] * std + mean           # odwrócenie normalizacji: z powrotem do zakresu 0-1
    img = img.permute(1, 2, 0)         # (3, 64, 64) -> (64, 64, 3), bo tak rysuje matplotlib
    plt.imshow(img.clamp(0, 1))        # clamp obcina wartości do 0-1 (błędy zaokrągleń)
    plt.title(class_names[yb[0].item()])
    plt.show()