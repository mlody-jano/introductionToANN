"""
ocena zapisanej sieci na zbiorze walidacyjnym: dokładność, makro-F1, wyniki dla każdej klasy
i macierz pomyłek. Wyniki trafiają do folderu przebiegu: report.txt i confusion.png.

najpierw trening sieci, np. python src/train.py --name cnn_aug --augment
potem:
    python src/evaluate.py --name cnn_aug
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from data import load_data
from train import MODELS, pick_device


@torch.no_grad()
def predict(model, loader, device):
    # przewidywane klasy dla wszystkich obrazów z loadera, razem z prawdziwymi etykietami
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        y_pred.append(logits.argmax(dim=1).cpu())   # przewidywana klasa, z powrotem na procesor
        y_true.append(yb)
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()


def plot_confusion(cm, class_names, path, title):
    # macierz pomyłek: wiersz = prawdziwa klasa, kolumna = przewidywana; w komórkach % wiersza
    pct = cm / cm.sum(axis=1, keepdims=True) * 100          # jaki % danej klasy trafił do każdej kolumny
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] == 0:
                continue
            color = "white" if pct[i, j] > 55 else "#0b0b0b"
            label = f"{pct[i, j]:.0f}" if pct[i, j] >= 0.5 else "<1"   # pojedyncze obrazy: "<1", nie "0"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    labels = [f"{k} {name}" for k, name in enumerate(class_names)]
    ax.set_xticks(range(len(class_names)), [str(k) for k in range(len(class_names))], fontsize=9)
    ax.set_yticks(range(len(class_names)), labels, fontsize=9)
    ax.set_xlabel("przewidywana klasa (numer)")
    ax.set_ylabel("prawdziwa klasa")
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def most_confused(cm, class_names, top=5):
    # pary klas najczęściej mylone ze sobą (poza przekątną)
    pairs = [(cm[i, j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j]
    pairs.sort(reverse=True)
    return [(class_names[i], class_names[j], int(n), n / cm[i].sum()) for n, i, j in pairs[:top]]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ocena wytrenowanej sieci na zbiorze walidacyjnym")
    p.add_argument("--name", required=True, help="nazwa przebiegu z results/runs/")
    run_dir = Path("results/runs") / p.parse_args().name

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))   # ustawienia z treningu
    device = pick_device()
    _, val_loader, _, class_names = load_data("data/processed/galaxy10_64.npz")

    model = MODELS[config["model"]]().to(device)                                  # ta sama architektura co w treningu
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))  # wagi najlepszej epoki

    y_true, y_pred = predict(model, val_loader, device)
    print(f"przebieg: {run_dir.name}")
    print(f"dokładność val: {accuracy_score(y_true, y_pred):.1%}")
    print(f"makro-F1 val:   {f1_score(y_true, y_pred, average='macro'):.3f}\n")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("najczęściej mylone (prawdziwa -> przewidziana):")
    for true_name, pred_name, n, share in most_confused(cm, class_names):
        print(f"  {true_name} -> {pred_name}: {n} obrazów ({share:.0%} tej klasy)")

    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    plot_confusion(cm, class_names, run_dir / "confusion.png",
                   f"{run_dir.name}, zbiór walidacyjny: % obrazów danej klasy w każdej kolumnie")
    print(f"\nzapisano: {run_dir}/report.txt, {run_dir}/confusion.png")