"""
Seria eksperymentów: jeden parametr, kilka wartości, każda wartość z kilkoma seedami.
Dla każdego przebiegu uruchamia train.py i evaluate.py, a na końcu robi tabelę i wykres.

Przebiegi trafiają do results/runs/<seria>/<wartość>_s<seed>/,
podsumowanie do results/series/<seria>/ (runs.csv, summary.csv, macro_f1.png).
Przebiegi już policzone są pomijane, więc przerwaną serię można po prostu uruchomić ponownie.

Na ekranie: pasek postępu całej serii i jedna linijka wyników na przebieg. Pełne wyjście
train.py i evaluate.py trafia do train.log i eval.log w folderze przebiegu.

Wszystkie pozostałe opcje (np. --augment, --model mlp) trafiają bez zmian do każdego train.py,
czyli są wspólne dla całej serii.

Przykłady (z głównego folderu projektu):
    python src/run_series.py --param lr --values 0.01 0.003 0.001 0.0003 --augment
    python src/run_series.py --param blocks --values 2 3 4 5 --augment
    python src/run_series.py --param activation --values relu leakyrelu elu tanh sigmoid --augment
    python src/run_series.py --param batchnorm --values off on --augment
    python src/run_series.py --param hidden --values "256" "512 256" "1024 512 256" \
        --series mlp_hidden --model mlp
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

FLAGS = {"augment", "batchnorm", "class-weights"}   # opcje bez wartości: włączone (on) albo nie (off)


def parse_args():
    p = argparse.ArgumentParser(description="Seria eksperymentów dla jednego parametru")
    p.add_argument("--param", required=True, help="nazwa opcji train.py bez --, np. lr, blocks, activation")
    p.add_argument("--values", nargs="+", required=True,
                   help='wartości do sprawdzenia; kilka liczb w cudzysłowie, np. "512 256"; '
                        "dla opcji włącz/wyłącz: off on")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="seedy dla każdej wartości")
    p.add_argument("--series", default=None, help="nazwa serii (domyślnie nazwa parametru)")
    args, base = p.parse_known_args()               # nieznane tu opcje = opcje wspólne dla train.py
    return args, base


def option_args(param, value):
    """Zamienia parametr i wartość na opcje dla train.py, np. ("lr", "0.01") -> ["--lr", "0.01"]."""
    if param in FLAGS:
        if value not in ("on", "off"):
            raise SystemExit(f"--{param} to opcja włącz/wyłącz: podaj wartości off i on")
        return [f"--{param}"] if value == "on" else []
    return [f"--{param}", *value.split()]


def run(cmd, log_path):
    """
    Uruchamia polecenie, a wszystko, co wypisuje, zapisuje do pliku log_path zamiast na ekran.
    Przy błędzie pokazuje koniec logu i przerywa serię.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}   # linie trafiają do logu od razu, nie na końcu
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8").splitlines()[-15:]
        tqdm.write("\n".join(tail))
        raise SystemExit(f"\nbłąd w: {' '.join(cmd)}\npełny log: {log_path}")


def read_run(run_dir):
    """Wyniki jednego przebiegu: metrics.json (z evaluate.py) i summary.json (z train.py) razem."""
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {**metrics, **summary}


def summarize(rows, values, series, param):
    """Zapisuje wszystkie przebiegi, średnie ± odchylenia dla wartości i wykres makro-F1."""
    out_dir = Path("results/series") / series
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = ["value", "seed", "macro_f1", "accuracy", "best_epoch", "epochs_run", "seconds", "params"]
    with open(out_dir / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    table = []
    for v in values:
        r = [x for x in rows if x["value"] == v]
        f1 = np.array([x["macro_f1"] for x in r])
        acc = np.array([x["accuracy"] for x in r])
        sd = (lambda a: float(a.std(ddof=1)) if len(a) > 1 else 0.0)   # odchylenie z próby
        table.append({
            "value": v, "seeds": len(r),
            "macro_f1_mean": round(float(f1.mean()), 4), "macro_f1_std": round(sd(f1), 4),
            "accuracy_mean": round(float(acc.mean()), 4), "accuracy_std": round(sd(acc), 4),
            "best_epoch_mean": round(float(np.mean([x["best_epoch"] for x in r])), 1),
            "minutes_mean": round(float(np.mean([x["seconds"] for x in r])) / 60, 1),
            "params": r[0]["params"],
        })
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    print(f"\nSeria {series} ({param}), średnia ± odchylenie z {table[0]['seeds']} seedów:")
    print(f"{'wartość':<16}{'makro-F1':>18}{'dokładność':>20}{'najl. epoka':>13}{'min':>7}{'parametry':>12}")
    for t in table:
        print(f"{t['value']:<16}{t['macro_f1_mean']:>10.3f} ± {t['macro_f1_std']:.3f}"
              f"{t['accuracy_mean']:>12.1%} ± {t['accuracy_std']:.1%}"
              f"{t['best_epoch_mean']:>13}{t['minutes_mean']:>7}{t['params']:>12,}".replace(",", " "))

    # wykres: szare kropki = pojedyncze seedy, niebieski punkt z kreską = średnia ± odchylenie
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(values) + 2), 4), dpi=150)
    x = np.arange(len(values))
    for i, v in enumerate(values):
        pts = [r["macro_f1"] for r in rows if r["value"] == v]
        ax.scatter([i - 0.15] * len(pts), pts, color="#b4b2a9", s=18, zorder=2)   # obok średniej, żeby było je widać
    means = [t["macro_f1_mean"] for t in table]
    ax.errorbar(x, means, yerr=[t["macro_f1_std"] for t in table], fmt="o", color="#2a78d6",
                ms=7, capsize=5, lw=2, zorder=3)
    for i, m in enumerate(means):
        ax.annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(10, -3), fontsize=9)
    ax.set_xticks(x, values)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.set_xlabel(param)
    ax.set_ylabel("makro-F1, zbiór walidacyjny")
    ax.set_title(f"Seria {series}: średnia ± odchylenie z {table[0]['seeds']} seedów",
                 loc="left", fontsize=11)
    ax.grid(axis="y", color="#e5e4df", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "macro_f1.png")
    plt.close(fig)
    print(f"\nzapisano: {out_dir}/runs.csv, summary.csv, macro_f1.png")


if __name__ == "__main__":
    args, base = parse_args()                       # base: np. ["--augment"] albo ["--model", "mlp"]
    series = args.series or args.param
    runs_dir = Path("results/runs")

    # wszystkie przebiegi serii: (wartość, seed, nazwa); do policzenia tylko te bez metrics.json
    jobs = [(v, s, f"{series}/{v.replace(' ', '-')}_s{s}") for v in args.values for s in args.seeds]
    todo = [job for job in jobs if not (runs_dir / job[2] / "metrics.json").exists()]
    print(f"seria {series}: {args.param} = {args.values}, seedy {args.seeds}, "
          f"opcje wspólne: {' '.join(base) or 'brak'}")
    print(f"przebiegów: {len(jobs)}, już policzonych: {len(jobs) - len(todo)}, do policzenia: {len(todo)}")
    print("szczegóły trwającego przebiegu: tail -f results/runs/<seria>/<przebieg>/train.log\n")

    bar = tqdm(todo, desc=f"seria {series}", unit="przebieg", disable=not todo)   # bez paska, gdy nic do zrobienia
    for value, seed, name in bar:
        bar.set_postfix_str(name)                   # który przebieg trwa teraz
        run_dir = runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        run([sys.executable, "src/train.py", "--name", name, "--seed", str(seed),
             *base, *option_args(args.param, value)], run_dir / "train.log")
        run([sys.executable, "src/evaluate.py", "--name", name], run_dir / "eval.log")
        r = read_run(run_dir)
        tqdm.write(f"  {name:<24} makro-F1 {r['macro_f1']:.3f} | dokładność {r['accuracy']:.1%} | "
                   f"najl. epoka {r['best_epoch']} z {r['epochs_run']} | {r['seconds'] / 60:.1f} min")
    bar.close()

    rows = [{"value": v, "seed": s, **read_run(runs_dir / name)} for v, s, name in jobs]
    summarize(rows, args.values, series, args.param)