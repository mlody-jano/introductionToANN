# Przygotowanie danych Galaxy10 DECaLS do treningu.


import argparse
import csv
import hashlib
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")  # zapis do plików, bez otwierania okien
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree

# Suma kontrolna pliku Galaxy10_DECals.h5 podana przez autorów zbioru (astroNN)
KNOWN_SHA256 = "19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571"

# Kolejność zgodna z etykietami 0-9 w pliku Galaxy10_DECals.h5
CLASS_NAMES = [
    "Zaburzona",               # 0 Disturbed
    "Łącząca się",             # 1 Merging
    "Okrągła gładka",          # 2 Round Smooth
    "Pośrednia gładka",        # 3 In-between Round Smooth
    "Cygarowata gładka",       # 4 Cigar Shaped Smooth
    "Spirala z poprzeczką",    # 5 Barred Spiral
    "Spirala ciasna",          # 6 Unbarred Tight Spiral
    "Spirala luźna",           # 7 Unbarred Loose Spiral
    "Z boku, bez zgrubienia",  # 8 Edge-on without Bulge
    "Z boku, ze zgrubieniem",  # 9 Edge-on with Bulge
]
SPLIT_NAMES = ("train", "val", "test")
META_KEYS = ("ra", "dec", "redshift", "pxscale")  # dodatkowe kolumny w pliku, jeśli istnieją

# Kolory wykresów (stonowana siatka, jeden kolor dla jednej serii danych)
BAR_COLOR = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e5e4df"


def parse_args():
    p = argparse.ArgumentParser(description="Przygotowanie danych Galaxy10 DECaLS")
    p.add_argument("--h5", default="data/Galaxy10_DECals.h5", help="ścieżka do pobranego pliku .h5")
    p.add_argument("--size", type=int, default=64, help="docelowy rozmiar obrazu (px)")
    p.add_argument("--crop", type=int, default=256,
                   help="rozmiar środkowego wycinka przed zmniejszeniem (256 = bez przycinania)")
    p.add_argument("--out", default="data/processed", help="folder na przetworzone dane")
    p.add_argument("--figs", default="results/figures", help="folder na wykresy")
    p.add_argument("--tables", default="results/tables", help="folder na tabele CSV do raportu")
    p.add_argument("--splits-dir", default="splits", help="folder na listę podziału (trafia do repozytorium)")
    p.add_argument("--seed", type=int, default=42, help="ziarno losowania podziału")
    p.add_argument("--chunk", type=int, default=500, help="ile obrazów wczytywać naraz")
    p.add_argument("--dup-radius", type=float, default=1.0,
                   help="odległość na niebie (sekundy kątowe), poniżej której dwa wpisy to ta sama galaktyka")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="nie usuwaj duplikatów (kopie tej samej galaktyki trafią do jednego zbioru)")
    p.add_argument("--skip-checksum", action="store_true", help="pomiń sprawdzanie sumy kontrolnej")
    return p.parse_args()


#wczytanie danych

def verify_checksum(path: Path) -> None:
    """Liczy SHA-256 pliku porcjami po 8 MB i porównuje z sumą podaną przez autorów."""
    print("Sprawdzam sumę kontrolną pliku (kilka sekund)...")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() == KNOWN_SHA256:
        print("  suma kontrolna zgodna: plik pobrał się w całości")
    else:
        print("  UWAGA: suma kontrolna się nie zgadza. Plik może być uszkodzony albo niepełny;"
              " najlepiej pobrać go ponownie.")


def center_crop_resize(img: np.ndarray, size: int, crop: int) -> np.ndarray:
    """Wycina środek crop x crop i zmniejsza do size x size (filtr Lanczos = antyaliasing)."""
    h, w = img.shape[:2]
    top, left = (h - crop) // 2, (w - crop) // 2
    img = img[top:top + crop, left:left + crop]
    if size == crop:  # pełna rozdzielczość wycinka: bez przeskalowania, piksele 1:1
        return np.ascontiguousarray(img)
    return np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.LANCZOS))


def load_and_resize(h5_path: Path, size: int, crop: int, chunk: int):
    """Zwraca: zmniejszone obrazy, etykiety, skróty pikseli (do wykrywania kopii) i metadane."""
    with h5py.File(h5_path, "r") as f:
        if "images" not in f or "ans" not in f:
            raise SystemExit(f"Nieoczekiwana struktura pliku. Klucze w pliku: {list(f.keys())}")
        n, h, w, c = f["images"].shape
        if not (size <= crop <= min(h, w)):
            raise SystemExit(f"Musi zachodzić size <= crop <= {min(h, w)} (jest size={size}, crop={crop})")
        labels = f["ans"][:].astype(np.int64)
        meta = {k: f[k][:].astype(np.float64) for k in META_KEYS if k in f}
        images = np.empty((n, size, size, c), dtype=np.uint8)
        hashes = []
        print(f"Plik: {n} obrazów {h}x{w}x{c} -> wycinek {crop}x{crop} -> {size}x{size}")
        print(f"  dodatkowe kolumny w pliku: {', '.join(meta) or 'brak'}")
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            batch = f["images"][start:stop]                 # tylko ta porcja trafia do pamięci
            for i, img in enumerate(batch):
                hashes.append(hashlib.md5(img.tobytes()).hexdigest())  # "odcisk palca" obrazu
                images[start + i] = center_crop_resize(img, size, crop)
            print(f"\r  przetworzono {stop}/{n}", end="", flush=True)
        print()
    return images, labels, hashes, meta


#duplikaty

def find_duplicate_groups(labels, hashes, meta, radius_arcsec: float):
    """
    Łączy w grupy wpisy, które są tą samą galaktyką:
      - współrzędne na niebie (ra, dec) bliżej niż radius_arcsec, albo
      - identyczne piksele obrazu.
    Zwraca numer grupy dla każdego obrazu; grupy numerowane w kolejności pierwszego wystąpienia,
    więc gdy duplikatów brak, numer grupy = numer obrazu.
    """
    n = len(labels)
    rows, cols = [], []

    n_pos_pairs = 0
    if "ra" in meta and "dec" in meta:
        coords = np.radians(np.column_stack([meta["dec"], meta["ra"]]))  # haversine: [szerokość, długość]
        tree = BallTree(coords, metric="haversine")
        neighbours = tree.query_radius(coords, r=np.radians(radius_arcsec / 3600.0))
        for i, nb in enumerate(neighbours):
            for j in nb:
                if j > i:
                    rows.append(i)
                    cols.append(j)
                    n_pos_pairs += 1

    n_hash_pairs = 0
    first_with_hash = {}
    for i, hsh in enumerate(hashes):
        if hsh in first_with_hash:
            rows.append(first_with_hash[hsh])
            cols.append(i)
            n_hash_pairs += 1
        else:
            first_with_hash[hsh] = i

    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, comp = connected_components(graph, directed=False)

    # przenumerowanie grup w kolejności pierwszego wystąpienia
    _, first_idx, inverse = np.unique(comp, return_index=True, return_inverse=True)
    order = np.argsort(first_idx)
    new_id = np.empty_like(order)
    new_id[order] = np.arange(len(order))
    groups = new_id[inverse]

    sizes = np.bincount(groups)
    multi = np.flatnonzero(sizes > 1)
    conflicts = [g for g in multi if len(set(labels[groups == g])) > 1]
    stats = {
        "pary_wspolrzedne": n_pos_pairs,
        "pary_identyczne_piksele": n_hash_pairs,
        "grupy_z_duplikatami": len(multi),
        "obrazy_w_tych_grupach": int(sizes[multi].sum()),
        "grupy_z_roznymi_etykietami": len(conflicts),
    }
    return groups, stats


def select_images(labels, groups, keep_duplicates: bool):
    """
    Decyduje, które obrazy zostają w zbiorze.
    Domyślnie: galaktyka z kopiami o sprzecznych etykietach jest usuwana w całości
    (nie wiadomo, która etykieta jest poprawna), a z kopii o zgodnej etykiecie zostaje pierwsza.
    Zwraca posortowane numery zachowanych obrazów i numery grup ze sprzecznymi etykietami.
    """
    sizes = np.bincount(groups)
    conflict_groups = [g for g in np.flatnonzero(sizes > 1) if len(set(labels[groups == g])) > 1]
    if keep_duplicates:
        return np.arange(len(labels)), conflict_groups
    keep = np.ones(len(labels), dtype=bool)
    seen = set()
    conflict_set = set(conflict_groups)
    for i, g in enumerate(groups):
        if g in conflict_set or g in seen:
            keep[i] = False
        seen.add(g)
    return np.flatnonzero(keep), conflict_groups


def print_duplicate_report(stats, n_images, n_kept, keep_duplicates):
    print("\nDuplikaty (ta sama galaktyka zapisana więcej niż raz):")
    print(f"  pary o tych samych współrzędnych: {stats['pary_wspolrzedne']}")
    print(f"  pary o identycznych pikselach:     {stats['pary_identyczne_piksele']}")
    if stats["grupy_z_duplikatami"] == 0:
        print("  nie znaleziono duplikatów")
        return
    print(f"  {stats['grupy_z_duplikatami']} galaktyk występuje kilka razy "
          f"({stats['obrazy_w_tych_grupach']} z {n_images} obrazów), "
          f"w tym {stats['grupy_z_roznymi_etykietami']} ze SPRZECZNYMI etykietami")
    if keep_duplicates:
        print("  zostawiam wszystkie kopie (--keep-duplicates); każda galaktyka trafi w całości do jednego zbioru")
    else:
        print(f"  usuwam {n_images - n_kept} obrazów; zostaje {n_kept}")


#podział

def split_indices(labels: np.ndarray, groups: np.ndarray, seed: int):
    """
    70/15/15 z zachowaniem proporcji klas (stratyfikacja), losowane po grupach:
    wszystkie kopie tej samej galaktyki trafiają do tego samego zbioru.
    Etykietą grupy jest etykieta jej pierwszego obrazu.
    """
    gid, first_of_group = np.unique(groups, return_index=True)  # pierwszy obraz każdej grupy
    group_labels = labels[first_of_group]

    pos = np.arange(len(gid))  # losujemy pozycje grup, potem zamieniamy je na numery grup
    p_train, p_rest = train_test_split(pos, test_size=0.30, stratify=group_labels, random_state=seed)
    p_val, p_test = train_test_split(p_rest, test_size=0.50, stratify=group_labels[p_rest],
                                     random_state=seed)
    return tuple(np.flatnonzero(np.isin(groups, gid[p])) for p in (p_train, p_val, p_test))


def channel_stats(x: np.ndarray, chunk: int = 1000):
    """Średnia i odchylenie każdego kanału (skala 0-1), liczone porcjami, żeby oszczędzać RAM."""
    s, s2, cnt = np.zeros(3), np.zeros(3), 0
    for i in range(0, len(x), chunk):
        b = x[i:i + chunk].reshape(-1, 3).astype(np.float64) / 255.0
        s += b.sum(axis=0)
        s2 += (b ** 2).sum(axis=0)
        cnt += len(b)
    mean = s / cnt
    std = np.sqrt(s2 / cnt - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)


#tabele i pliki CSV

def class_table(labels, splits):
    """Liczebność klas w zachowanych obrazach (splits zawierają numery obrazów z pliku .h5)."""
    n_kept = sum(len(s) for s in splits)
    rows = []
    for k, name in enumerate(CLASS_NAMES):
        parts = [int((labels[s] == k).sum()) for s in splits]
        total = sum(parts)
        rows.append([k, name, total, *parts, total / n_kept])
    return rows


def print_class_table(rows, splits, n):
    print(f"\n{'klasa':<26}{'razem':>7}{'train':>7}{'val':>6}{'test':>6}{'udział':>8}")
    for k, name, total, tr, va, te, share in rows:
        print(f"{k} {name:<24}{total:>7}{tr:>7}{va:>6}{te:>6}{share:>8.1%}")
    print(f"{'RAZEM':<26}{n:>7}" + "".join(f"{len(s):>{w}}" for s, w in zip(splits, (7, 6, 6))))


def save_class_table(rows, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["klasa", "nazwa", "razem", "train", "val", "test", "udzial"])
        for k, name, total, tr, va, te, share in rows:
            w.writerow([k, name, total, tr, va, te, f"{share:.4f}"])


def save_split_csv(path: Path, labels, groups, splits, meta):
    """Jeden wiersz na obraz: numer w pliku .h5, etykieta, zbiór, grupa duplikatów, metadane."""
    split_of = np.full(len(labels), "usuniety", dtype=object)   # obrazy pominięte jako duplikaty
    for name, idx in zip(SPLIT_NAMES, splits):
        split_of[idx] = name
    keys = list(meta)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["h5_index", "label", "class_name", "split", "group", *keys])
        for i in range(len(labels)):
            w.writerow([i, int(labels[i]), CLASS_NAMES[labels[i]], split_of[i], int(groups[i]),
                        *(f"{meta[k][i]:.6f}" for k in keys)])


def save_duplicates_csv(path: Path, labels, groups, kept_idx, meta):
    """Lista wszystkich galaktyk występujących kilka razy: do obejrzenia i do raportu."""
    sizes = np.bincount(groups)
    kept = set(kept_idx.tolist())
    keys = [k for k in ("ra", "dec") if k in meta]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "h5_index", "label", "class_name", "status", *keys])
        for i in np.flatnonzero(sizes[groups] > 1):
            w.writerow([int(groups[i]), int(i), int(labels[i]), CLASS_NAMES[labels[i]],
                        "zachowany" if i in kept else "usuniety", *(f"{meta[k][i]:.6f}" for k in keys)])


#wykresy

def style_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_class_counts(labels, path: Path):
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    y = np.arange(len(CLASS_NAMES))
    ax.barh(y, counts, height=0.5, color=BAR_COLOR)
    for yi, c in zip(y, counts):
        ax.text(c + counts.max() * 0.01, yi, f"{c}", va="center", fontsize=9, color=TEXT_2)
    ax.set_yticks(y, [f"{i}  {n}" for i, n in enumerate(CLASS_NAMES)], fontsize=9, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("liczba obrazów", color=TEXT_2, fontsize=9)
    ax.set_title(f"Galaxy10 DECaLS: liczebność klas (razem {len(labels)} obrazów)",
                 loc="left", fontsize=11, color=TEXT)
    ax.set_xlim(0, counts.max() * 1.12)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_examples(h5_path: Path, labels, idx_pool, path: Path, seed: int, per_class: int = 6):
    """Siatka: wiersz = klasa, kolumny = losowe przykłady ze zbioru treningowego (pełna rozdzielczość)."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class,
                             figsize=(per_class * 1.25 + 1.9, len(CLASS_NAMES) * 1.25), dpi=150)
    with h5py.File(h5_path, "r") as f:
        for k, name in enumerate(CLASS_NAMES):
            pool = idx_pool[labels[idx_pool] == k]
            chosen = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
            for j in range(per_class):
                ax = axes[k, j]
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if j < len(chosen):
                    ax.imshow(f["images"][int(chosen[j])])
            axes[k, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8, color=TEXT)
    fig.suptitle("Przykładowe galaktyki z każdej klasy (256x256)", fontsize=11, color=TEXT, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_conflicts(h5_path: Path, labels, groups, conflict_groups, path: Path, n: int = 8):
    """Galaktyki zapisane kilka razy ze sprzecznymi etykietami: obraz + etykiety wszystkich kopii."""
    shown = conflict_groups[:n]
    cols = 4
    rows = (len(shown) + cols - 1) // cols
    height = rows * 2.9 + 0.5  # 0,5 cala zapasu na tytuł nad siatką
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, height), dpi=150, squeeze=False)
    with h5py.File(h5_path, "r") as f:
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        for ax, g in zip(axes.flat, shown):
            members = np.flatnonzero(groups == g)
            ax.imshow(f["images"][int(members[0])])
            names = "\n".join(f"#{m}: {CLASS_NAMES[labels[m]]}" for m in members)
            ax.set_title(names, fontsize=7, color=TEXT)
    fig.suptitle("Ta sama galaktyka, różne etykiety w zbiorze danych", fontsize=11, color=TEXT,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / height))
    fig.savefig(path)
    plt.close(fig)


def plot_resolutions(h5_path: Path, labels, idx_pool, path: Path, crop: int, seed: int):
    """Ta sama galaktyka w różnych rozdzielczościach: co widzi sieć przy 128, 64 i 32 px."""
    rng = np.random.default_rng(seed + 1)
    picks = [5, 7, 9]  # spirala z poprzeczką, spirala luźna, widok z boku: drobne szczegóły
    sizes = [256, 128, 64, 32]
    fig, axes = plt.subplots(len(picks), len(sizes), figsize=(len(sizes) * 1.7, len(picks) * 1.8), dpi=150)
    with h5py.File(h5_path, "r") as f:
        for r, k in enumerate(picks):
            pool = idx_pool[labels[idx_pool] == k]
            img = f["images"][int(rng.choice(pool))]
            for c, s in enumerate(sizes):
                ax = axes[r, c]
                view = center_crop_resize(img, min(s, crop), crop) if s < img.shape[0] else img
                ax.imshow(view, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if r == 0:
                    ax.set_title(f"{s}x{s}", fontsize=9, color=TEXT_2)
            axes[r, 0].set_ylabel(CLASS_NAMES[k], fontsize=8, color=TEXT)
    fig.suptitle("Wpływ rozdzielczości wejścia na widoczne szczegóły", fontsize=11, color=TEXT, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


#główny program

def main():
    args = parse_args()
    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise SystemExit(f"Nie ma pliku {h5_path}. Pobierz Galaxy10_DECals.h5 z Zenodo do folderu data/.")
    out_dir, fig_dir, tab_dir, split_dir = map(Path, (args.out, args.figs, args.tables, args.splits_dir))
    for d in (out_dir, fig_dir, tab_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not args.skip_checksum:
        verify_checksum(h5_path)

    images, labels, hashes, meta = load_and_resize(h5_path, args.size, args.crop, args.chunk)

    groups, dup_stats = find_duplicate_groups(labels, hashes, meta, args.dup_radius)
    kept_idx, conflict_groups = select_images(labels, groups, args.keep_duplicates)
    print_duplicate_report(dup_stats, len(labels), len(kept_idx), args.keep_duplicates)

    # podział liczony na zachowanych obrazach, a potem przeliczony na numery obrazów z pliku .h5
    sub_splits = split_indices(labels[kept_idx], groups[kept_idx], args.seed)
    splits = tuple(kept_idx[s] for s in sub_splits)
    idx_train, idx_val, idx_test = splits
    mean, std = channel_stats(images[idx_train])

    rows = class_table(labels, splits)
    print_class_table(rows, splits, len(kept_idx))
    print(f"\nŚrednia kanałów (train): {np.round(mean, 4)}   odchylenie: {np.round(std, 4)}")

    suffix = f"{args.size}" if args.crop == 256 else f"{args.size}_crop{args.crop}"
    out_path = out_dir / f"galaxy10_{suffix}.npz"
    np.savez(
        out_path,
        X_train=images[idx_train], y_train=labels[idx_train],
        X_val=images[idx_val], y_val=labels[idx_val],
        X_test=images[idx_test], y_test=labels[idx_test],
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,   # numery obrazów z pliku .h5
        mean=mean, std=std, class_names=np.array(CLASS_NAMES),
    )
    print(f"\nZapisano dane:        {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")

    split_path = split_dir / f"galaxy10_seed{args.seed}.csv"
    save_split_csv(split_path, labels, groups, splits, meta)
    save_class_table(rows, tab_dir / "class_counts.csv")
    print(f"Zapisano podział:     {split_path}")
    print(f"Zapisano tabelę klas: {tab_dir / 'class_counts.csv'}")
    figs = ["class_counts.png", "class_examples.png", "resolution_comparison.png"]
    if dup_stats["grupy_z_duplikatami"]:
        save_duplicates_csv(tab_dir / "duplicates.csv", labels, groups, kept_idx, meta)
        print(f"Zapisano duplikaty:   {tab_dir / 'duplicates.csv'}")

    plot_class_counts(labels[kept_idx], fig_dir / "class_counts.png")
    plot_examples(h5_path, labels, idx_train, fig_dir / "class_examples.png", args.seed)
    plot_resolutions(h5_path, labels, idx_train, fig_dir / "resolution_comparison.png", args.crop, args.seed)
    if conflict_groups:
        plot_conflicts(h5_path, labels, groups, conflict_groups, fig_dir / "duplicates_conflicts.png")
        figs.append("duplicates_conflicts.png")
    print(f"Zapisano wykresy:     {fig_dir}/ {', '.join(figs)}")


if __name__ == "__main__":
    main()