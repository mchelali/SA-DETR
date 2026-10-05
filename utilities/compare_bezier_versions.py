"""
Compare deux conversions polygone -> Bézier (v3 et v4 de utilities/curves.py)
sur un JSON COCO de tampons.

Pour chaque annotation, le contour des 2 Béziers est échantillonné comme dans
adet/data/datasets/text.py, puis comparé à la segmentation (IoU, contour qui se
croise). Écrit un résumé par catégorie de polygone et une planche des cas
demandés ou des pires cas : pour chaque version, courbe du haut et courbe du
bas fléchées dans leur sens de parcours, et ligne médiane (polyline).

Usage (depuis la racine du projet) :
  python utilities/compare_bezier_versions.py \
      datasets/forbin_dataset/test_fold_0_single.json \
      --out-dir output/bezier2coco_check --ids 733 488 53 24 713 719
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.curves import (  # noqa: E402
    robust_polygon_to_bezier_v3,
    robust_polygon_to_bezier_v4,
)

VERSIONS = {"v3": robust_polygon_to_bezier_v3, "v4": robust_polygon_to_bezier_v4}


def sample_boundary(bezier_pts, n=50):
    """Contour fermé (2n, 2), même calcul que load_text_json."""
    b = np.asarray(bezier_pts, dtype=float).reshape(-1, 2)
    ctrl = b.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    u = np.linspace(0, 1, n)
    p = (
        np.outer((1 - u) ** 3, ctrl[:, 0])
        + np.outer(3 * u * (1 - u) ** 2, ctrl[:, 1])
        + np.outer(3 * u**2 * (1 - u), ctrl[:, 2])
        + np.outer(u**3, ctrl[:, 3])
    )
    return np.vstack([p[:, :2], p[:, 2:]])


def iou_and_valid(bezier_pts, seg_poly):
    boundary = Polygon(sample_boundary(bezier_pts))
    valid = boundary.is_valid
    if not valid:
        boundary = boundary.buffer(0)
    union = boundary.union(seg_poly).area
    return (boundary.intersection(seg_poly).area / union if union else 0.0), valid


def category(seg):
    """Catégorie du polygone (cf. README_add_bezier2coco.md)."""
    if len(seg) == 4:
        return "quadrilatère"
    poly = Polygon(seg)
    if poly.convex_hull.area > poly.buffer(0).area * 1.02:
        return "non convexe"
    return "> 4 points, convexe"


def sample_side(ctrl4, n=50):
    """Points d'une Bézier cubique (4 points de contrôle), de P0 à P3."""
    u = np.linspace(0, 1, n)[:, None]
    basis = np.hstack([(1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u**2 * (1 - u), u**3])
    return basis @ np.asarray(ctrl4, dtype=float)


def draw_directed(ax, pts, color, label, lw=2.0, ls="-", start_label=None):
    """Trace une courbe avec des flèches dans le sens de parcours et marque son départ."""
    ax.plot(*pts.T, ls, color=color, lw=lw, label=label)
    n = len(pts)
    for i in (n // 4, n // 2, 3 * n // 4):
        ax.annotate(
            "", xy=pts[min(i + 2, n - 1)], xytext=pts[i - 2],
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=16),
        )
    ax.plot(*pts[0], "o", color=color, ms=8, mec="black", mew=0.8)
    if start_label:
        ax.annotate(start_label, pts[0], xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold")


def plot_cases(cases, path, versions=("v3", "v4"), images=None):
    """
    Planche : un panneau par (cas, version).

    Bleu : segmentation. Orange : courbe du haut (P0 -> P3, départ « H »).
    Violet : courbe du bas (P0 -> P3, départ « B »). Pointillé gris : ligne
    médiane (polyline cible du modèle), dans le sens appris.

    `images` : {id d'annotation: chemin de l'image}. Si fourni, le tampon est
    affiché en fond pour comparer la direction des courbes à celle du texte.
    """
    from PIL import Image
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 2 * len(versions) if len(versions) <= 2 else len(versions)
    per_row = cols // len(versions)
    rows = int(np.ceil(len(cases) / per_row))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (ann_id, seg, res) in enumerate(cases):
        r, c0 = divmod(k, per_row)
        for j, name in enumerate(versions):
            ax = axes[r, c0 * len(versions) + j]
            ax.axis("on")
            bez, iou, valid = res[name]
            bez = np.asarray(bez, dtype=float).reshape(-1, 2)
            top, bottom = sample_side(bez[:4]), sample_side(bez[4:])

            img_path = (images or {}).get(ann_id)
            if img_path and os.path.exists(img_path):
                pad = 0.15 * max(np.ptp(seg[:, 0]), np.ptp(seg[:, 1]))
                x0, y0 = np.floor(seg.min(0) - pad).astype(int)
                x1, y1 = np.ceil(seg.max(0) + pad).astype(int)
                with Image.open(img_path) as im:
                    x0, y0 = max(x0, 0), max(y0, 0)
                    x1, y1 = min(x1, im.width), min(y1, im.height)
                    crop = np.asarray(im.convert("RGB").crop((x0, y0, x1, y1)))
                # extent : coordonnées image ; l'axe y est déjà orienté vers le bas
                ax.imshow(crop, extent=(x0, x1, y1, y0))
            closed = np.vstack([seg, seg[:1]])
            ax.fill(*closed.T, color="tab:blue", alpha=0.12)
            ax.plot(*closed.T, ".-", color="tab:blue", lw=0.8, ms=4, label="segmentation")
            draw_directed(ax, top, "tab:orange", "haut (P0→P3)", start_label="H")
            draw_directed(ax, bottom, "tab:purple", "bas (P0→P3)", start_label="B")
            # Ligne médiane, comme dans load_text_json : (haut + bas inversé) / 2
            draw_directed(ax, (top + bottom[::-1]) / 2, "dimgray", "polyline", lw=1.2, ls="--")
            ax.plot(*bez[:4].T, "x", color="tab:orange", ms=5)
            ax.plot(*bez[4:].T, "x", color="tab:purple", ms=5)

            ax.set_title(
                f"id {ann_id} ({len(seg)} pts) — {name} IoU={iou:.2f}"
                + ("" if valid else " (se croise)"),
                fontsize=9,
            )
            if not (img_path and os.path.exists(img_path)):
                ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            if k == 0:
                ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("json", help="JSON COCO avec segmentations")
    parser.add_argument("--out-dir", default="output/bezier2coco_check")
    parser.add_argument("--ids", type=int, nargs="*", default=[], help="annotations à dessiner")
    parser.add_argument("--n-worst", type=int, default=12, help="pires cas v3 ajoutés à la planche")
    parser.add_argument("--image-root", default=None,
                        help="dossier des images (défaut : dossier du JSON) ; 'none' pour ne pas les afficher")
    args = parser.parse_args()

    coco = json.load(open(args.json, encoding="utf-8"))
    image_root = args.image_root or os.path.dirname(os.path.abspath(args.json))
    img_files = {i["id"]: i["file_name"] for i in coco["images"]}
    ann_images = {}
    by_cat = defaultdict(lambda: defaultdict(list))
    all_res = {}
    for ann in coco["annotations"]:
        seg = np.asarray(ann["segmentation"][0], dtype=float).reshape(-1, 2)
        seg_poly = Polygon(seg).buffer(0)
        res = {}
        for name, fn in VERSIONS.items():
            bez = np.asarray(fn(seg.tolist()), dtype=float).reshape(-1, 2)
            iou, valid = iou_and_valid(bez, seg_poly)
            res[name] = (bez, iou, valid)
            for c in (category(seg), "TOUS"):
                by_cat[c][name].append((iou, valid))
        all_res[ann["id"]] = (seg, res)
        if image_root.lower() != "none":
            ann_images[ann["id"]] = os.path.join(image_root, img_files[ann["image_id"]])

    lines = ["| Catégorie | n | " + " | ".join(
        f"{v} IoU moy | {v} IoU min | {v} IoU<0,9 | {v} se croise" for v in VERSIONS) + " |"]
    lines.append("|---" * (2 + 4 * len(VERSIONS)) + "|")
    for c in sorted(by_cat, key=lambda k: (k == "TOUS", k)):
        cells = []
        for v in VERSIONS:
            iou = np.array([x[0] for x in by_cat[c][v]])
            inv = sum(not x[1] for x in by_cat[c][v])
            cells += [f"{iou.mean():.3f}", f"{iou.min():.3f}", str(int((iou < 0.9).sum())), str(inv)]
        lines.append(f"| {c} | {len(by_cat[c]['v3'])} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)

    worst = sorted(all_res, key=lambda i: all_res[i][1]["v3"][1])[: args.n_worst]
    ids = list(dict.fromkeys(args.ids + worst))
    cases = [(i, *all_res[i]) for i in ids if i in all_res]
    for i, seg, res in cases:
        print(f"id {i:5d}: v3 IoU={res['v3'][1]:.3f}  ->  v4 IoU={res['v4'][1]:.3f}"
              f"{'' if res['v4'][2] else ' (v4 se croise)'}")

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.json))[0]
    png = os.path.join(args.out_dir, f"compare_v3_v4_{stem}.png")
    plot_cases(cases, png, images=ann_images)
    with open(os.path.join(args.out_dir, f"compare_v3_v4_{stem}.md"), "w", encoding="utf-8") as f:
        f.write(f"# v3 vs v4 — {args.json}\n\n{table}\n\n![cas]({os.path.basename(png)})\n")
    print(f"\nPlanche : {png}")


if __name__ == "__main__":
    main()
