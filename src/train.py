"""
trening sieci z wczesnym zatrzymaniem. Każdy przebieg zapisuje swoje wyniki
do osobnego folderu results/runs/<nazwa>/:
    config.json   ustawienia przebiegu
    history.csv   wyniki każdej epoki
    curves.png    krzywe uczenia
    model.pt      wagi z najlepszej epoki (nie trafiają do gita)

np:
    python src/train.py --name cnn_base
    python src/train.py --name cnn_aug --augment
    python src/train.py --name mlp_base --model mlp
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # wykres do pliku, bez otwierania okna
import matplotlib.pyplot as plt
import torch
from torch import nn

from data import augment_batch, load_data
from models import CNN, MLP

MODELS = {"mlp": MLP, "cnn": CNN}   # nazwa z wiersza poleceń -> klasa modelu


def parse_args():
    p = argparse.ArgumentParser(description="Trening sieci na Galaxy10")
    p.add_argument("--name", required=True, help="nazwa przebiegu; wyniki trafią do results/runs/<nazwa>/")
    p.add_argument("--model", choices=list(MODELS), default="cnn", help="architektura sieci")
    p.add_argument("--augment", action="store_true", help="losowe obroty i odbicia obrazów treningowych")
    p.add_argument("--epochs", type=int, default=50, help="najwięcej epok")
    p.add_argument("--patience", type=int, default=5, help="koniec po tylu epokach bez poprawy straty val")
    p.add_argument("--lr", type=float, default=0.001, help="współczynnik uczenia")
    p.add_argument("--seed", type=int, default=0, help="ziarno losowania")
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, loss_fn, optimizer, device, augment=False):
    # jedno przejście przez cały zbiór treningowy; zwraca średnią stratę
    model.train()                                   # tryb treningu: dropout włączony
    total_loss, n = 0.0, 0
    for xb, yb in loader:                           # partia po partii
        xb, yb = xb.to(device), yb.to(device)       # dane na GPU
        if augment:
            xb = augment_batch(xb)                  # losowe obroty i odbicia (tylko przy treningu)
        logits = model(xb)                          # 1. przewidywania sieci
        loss = loss_fn(logits, yb)                  # 2. jak bardzo się pomyliła
        optimizer.zero_grad()                       # 3. wyzeruj gradienty z poprzedniej partii
        loss.backward()                             # 4. policz gradienty (propagacja wsteczna)
        optimizer.step()                            # 5. popraw wagi
        total_loss += loss.item() * len(yb)
        n += len(yb)
    return total_loss / n


@torch.no_grad()                                    # tylko ocena, więc bez gradientów
def evaluate(model, loader, loss_fn, device):
    # srednia strata i dokładność na podanym zbiorze; wagi się nie zmieniają
    model.eval()                                    # tryb oceny: dropout wyłączony
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()   # ile przewidywań trafiło
        n += len(yb)
    return total_loss / n, correct / n


def plot_curves(history, path, best_epoch, title):
    # krzywe uczenia: strata (train i val) oraz dokładność val; przerywana linia = najlepsza epoka
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    ax1.plot(epochs, history["train_loss"], color="#2a78d6", lw=2, label="zbiór treningowy")
    ax1.plot(epochs, history["val_loss"], color="#e8590c", lw=2, label="zbiór walidacyjny")
    ax1.set_title("Strata (entropia krzyżowa)", loc="left")
    ax1.legend(frameon=False)

    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], color="#e8590c", lw=2)
    ax2.set_title("Dokładność na zbiorze walidacyjnym (%)", loc="left")

    for ax in (ax1, ax2):
        ax.axvline(best_epoch, color="#888780", lw=1, ls="--")   # najlepsza epoka
        ax.set_xlabel("epoka")
        ax.grid(color="#e5e4df", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color="#52514e")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_history(history, path):
    # wyniki każdej epoki do pliku CSV, żeby później porównywać przebiegi ze sobą
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
        rows = zip(history["train_loss"], history["val_loss"], history["val_acc"])
        for epoch, row in enumerate(rows, start=1):
            w.writerow([epoch, *(f"{v:.5f}" for v in row)])


if __name__ == "__main__":
    args = parse_args()
    run_dir = Path("results/runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"przebieg: {args.name} | model: {args.model} | augmentacja: {args.augment} | urządzenie: {device}")

    train_loader, val_loader, _, class_names = load_data("data/processed/galaxy10_64.npz")
    model = MODELS[args.model]().to(device)        # MLP() albo CNN(), na GPU
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}   # wyniki każdej epoki
    best_val_loss = float("inf")    # najlepsza strata val do tej pory (na start: nieskończoność)
    best_epoch = 0
    best_state = None               # kopia wag z najlepszej epoki
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.augment)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)    # dopisuje wynik tej epoki na koniec listy
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"epoka {epoch}: strata train {train_loss:.3f} | strata val {val_loss:.3f} | "
              f"dokładność val {val_acc:.1%} | {time.time() - start:.1f} s")

        # --- wczesne zatrzymanie
        if val_loss < best_val_loss:                        # nowy rekord: pamietanie wersji sieci
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())  # niezależna kopia wszystkich wag
            epochs_without_improvement = 0
        else:                                               # brak poprawy licznik w górę
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"brak poprawy od {args.patience} epok: koniec treningu")
                break                                       # wyjście z pętli epok

    # przywrócenie najlepszej wersji sieci i zapis wyników
    model.load_state_dict(best_state)
    val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
    print(f"\nnajlepsza epoka: {best_epoch} | strata val {val_loss:.3f} | dokładność val {val_acc:.1%}")

    torch.save(best_state, run_dir / "model.pt")
    save_history(history, run_dir / "history.csv")
    plot_curves(history, run_dir / "curves.png", best_epoch,
                f"{args.name}: najlepsza epoka {best_epoch}, dokładność val {val_acc:.1%}")
    print(f"zapisano wyniki w {run_dir}/")