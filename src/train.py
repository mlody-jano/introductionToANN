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
    python src/train.py --name cnn_4bloki --augment --blocks 4
    python src/train.py --name cnn_elu_bn --augment --activation elu --batchnorm
    python src/train.py --name mlp_1warstwa --model mlp --hidden 256
    python src/train.py --name cnn_sgd --augment --optimizer sgd --lr 0.01
    python src/train.py --name cnn_wagi --augment --class-weights
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
from models import ACTIVATIONS, build_model, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="Trening sieci na Galaxy10")
    p.add_argument("--name", required=True, help="nazwa przebiegu; wyniki trafią do results/runs/<nazwa>/")
    # architektura
    p.add_argument("--model", choices=["mlp", "cnn"], default="cnn", help="rodzaj sieci")
    p.add_argument("--blocks", type=int, default=3, help="CNN: liczba bloków konwolucyjnych")
    p.add_argument("--filters", type=int, default=32,
                   help="CNN: filtry w pierwszym bloku (w każdym następnym 2 razy więcej)")
    p.add_argument("--hidden", type=int, nargs="+", default=None,
                   help="neurony w warstwach ukrytych, np. --hidden 512 256 "
                        "(domyślnie: MLP 512 256, CNN 256)")
    p.add_argument("--activation", choices=list(ACTIVATIONS), default="relu", help="funkcja aktywacji")
    p.add_argument("--dropout", type=float, default=0.3, help="ułamek neuronów wyłączanych przy treningu")
    p.add_argument("--batchnorm", action="store_true", help="CNN: normalizacja wsadowa w każdym bloku")
    # --- trening
    p.add_argument("--augment", action="store_true", help="losowe obroty i odbicia obrazów treningowych")
    p.add_argument("--epochs", type=int, default=50, help="najwięcej epok")
    p.add_argument("--patience", type=int, default=5, help="koniec po tylu epokach bez poprawy straty val")
    p.add_argument("--lr", type=float, default=0.001, help="współczynnik uczenia")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
                   help="adam albo sgd (czysty spadek gradientowy)")
    p.add_argument("--momentum", type=float, default=0.0, help="SGD: pęd (0 = czysty spadek gradientowy)")
    p.add_argument("--weight-decay", type=float, default=0.0, help="kara za duże wagi (regularyzacja)")
    p.add_argument("--class-weights", action="store_true",
                   help="ważona funkcja straty: błędy na rzadkich klasach kosztują więcej")
    # dane
    p.add_argument("--size", type=int, choices=[64, 128], default=64,
                   help="rozdzielczość obrazów (wymaga python src/prepare_data.py --size 128)")
    p.add_argument("--seed", type=int, default=0, help="ziarno losowania")
    return p.parse_args()


def make_optimizer(model, args):
    # optymalizator według opcji: Adam albo SGD, oba z opcjonalną karą za duże wagi
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def class_weights(labels, n_classes):
    # Waga każdej klasy = wszystkie obrazy / (liczba klas * obrazy tej klasy)
    # Rzadka klasa dostaje dużą wagę a częsta małą
    counts = torch.bincount(labels, minlength=n_classes).float()
    return len(labels) / (n_classes * counts)


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


@torch.no_grad()                                    # tylko oceniamy, więc bez liczenia gradientów
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
    print(f"przebieg: {args.name} | model: {args.model} | optymalizator: {args.optimizer} | "
          f"augmentacja: {args.augment} | obrazy: {args.size}px | urządzenie: {device}")

    data_path = Path(f"data/processed/galaxy10_{args.size}.npz")
    if not data_path.exists():
        raise SystemExit(f"Brak {data_path}. Najpierw: python src/prepare_data.py --size {args.size}")
    train_loader, val_loader, _, class_names = load_data(data_path)
    model = build_model(vars(args)).to(device)     # sieć według opcji z polecenia, na GPU
    print(f"parametry sieci: {count_parameters(model):,}".replace(",", " "))
    if args.class_weights:
        weights = class_weights(train_loader.dataset.tensors[1], len(class_names))
        print("wagi klas:", [round(w, 2) for w in weights.tolist()])
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, args)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}   # wyniki każdej epoki
    best_val_loss = float("inf")    # najlepsza strata val do tej pory (na start: nieskończoność)
    best_epoch = 0
    best_state = None               # kopia wag z najlepszej epoki
    epochs_without_improvement = 0

    run_start = time.time()                         # czas całego treningu
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, args.augment)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)    # dopisuje wynik tej epoki na koniec listy
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"epoka {epoch}: strata train {train_loss:.3f} | strata val {val_loss:.3f} | "
              f"dokładność val {val_acc:.1%} | {time.time() - start:.1f} s")

        # wczesne zatrzymanie
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
    summary = {                                     # najważniejsze liczby przebiegu, dla skryptu serii
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "val_loss": round(val_loss, 5),
        "val_acc": round(val_acc, 5),
        "seconds": round(time.time() - run_start, 1),
        "params": count_parameters(model),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_history(history, run_dir / "history.csv")
    plot_curves(history, run_dir / "curves.png", best_epoch,
                f"{args.name}: najlepsza epoka {best_epoch}, dokładność val {val_acc:.1%}")
    print(f"zapisano wyniki w {run_dir}/")