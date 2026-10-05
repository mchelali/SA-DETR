"""
Enrichit un JSON COCO de tampons pour l'entraînement SA-DETR / DeepSolo.

Pour chaque annotation, ajoute :
  - `bezier_pts` : 8 points (x, y) = 2 Béziers cubiques. Les 4 premiers décrivent
    le côté « haut » (P0 -> P3), les 4 suivants le côté « bas », qui part de
    l'extrémité où finit le haut (convention en boucle attendue par
    adet/data/datasets/text.py pour calculer polyline et boundary).
  - `rec`        : texte encodé pour la CTC (MAX_LEN indices, complété par PAD_IDX).
  - `text`       : texte brut (ou "###" si absent).
  - `language`   : toujours "Latin".

Les annotations sont modifiées en place : `segmentation` est réécrite sans ses
points dupliqués. Les annotations en double (même image, même polygone) sont
supprimées, sauf avec --keep-duplicates.

Usage (depuis la racine du projet) :
  python utilities/add_bezier2coco.py IN.json OUT.json [--bezier-version v4]

Voir utilities/README_add_bezier2coco.md pour l'analyse des cas d'échec.
"""

import argparse
import json
import logging
import os
from typing import List, Dict

import numpy as np
from tqdm import tqdm

# Fonctionne lancé depuis la racine (`python utilities/add_bezier2coco.py`,
# où seul utilities/ est dans sys.path) comme importé en tant que module.
try:
    from utilities.curves import (
        robust_polygon_to_bezier_v3,
        robust_polygon_to_bezier_v4,
    )
except ImportError:
    from curves import robust_polygon_to_bezier_v3, robust_polygon_to_bezier_v4

# v4 : coupe haut/bas de coin en coin, ajustement sur le contour densifié,
# choix des coins par IoU (cf. README_add_bezier2coco.md). v3 gardée pour comparaison.
BEZIER_FUNCS = {
    "v3": robust_polygon_to_bezier_v3,
    "v4": robust_polygon_to_bezier_v4,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Longueur fixe de `rec` (= nombre de point queries de DeepSolo à l'origine).
# Indépendante de MODEL.TRANSFORMER.NUM_POINTS pour la détection seule.
MAX_LEN = 25
# Index de padding. Un `rec` entièrement à PAD_IDX fait écarter l'instance par
# load_text_json (text.py) à l'entraînement : un texte vide est donc remplacé
# par NO_TEXT, dont les caractères existent dans le char2idx.
PAD_IDX = 0
NO_TEXT = "###"


class NumpyEncoder(json.JSONEncoder):
    """Convertit les types NumPy (tableaux, float32/64, int32/64) en types Python pour json.dump."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


def load_char2idx(path: str) -> Dict[str, int]:
    """Charge le dictionnaire caractère -> index (ex. char_map/char2idx/Latin.json)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Le fichier de mapping de caractères est introuvable : {path}"
        )
    with open(path, encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f).items()}


def encode_text(text: str, char2idx: Dict[str, int]) -> List[int]:
    """
    Encode le texte pour la CTC : tronqué à MAX_LEN, complété par PAD_IDX.

    Les caractères absents de char2idx sont encodés PAD_IDX (donc perdus).
    Un texte vide donne MAX_LEN fois PAD_IDX : process_annotation passe donc
    NO_TEXT à la place (cf. PAD_IDX).
    """
    if not text:
        return [PAD_IDX] * MAX_LEN
    seq = [char2idx.get(ch, PAD_IDX) for ch in text[:MAX_LEN]]
    return seq + [PAD_IDX] * (MAX_LEN - len(seq))


def process_annotation(
    ann: Dict, char2idx: Dict[str, int], to_bezier=robust_polygon_to_bezier_v4
) -> Dict:
    """
    Enrichit une annotation (en place) avec `bezier_pts`, `rec`, `text` et `language`.

    `bezier_pts` est absent en sortie si la conversion a échoué (main() compte
    ces cas et les signale).
    """
    segs = ann.get("segmentation", [])
    bbox = ann.get("bbox", [])

    # 0. Sans segmentation, on la reconstruit à partir de la bbox (rectangle).
    if len(segs) == 0 and (not bbox or len(bbox) < 4):
        logging.warning(
            f"Annotation {ann.get('id')} ignorée (pas de segmentation/bbox)."
        )
        return ann
    if len(segs) == 0:
        x, y, w, h = bbox
        segs = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        ann["segmentation"] = segs

    # 1. Points de Bézier
    if segs and isinstance(segs, list) and len(segs) > 0:
        # Seul le premier polygone est utilisé (les multi-polygones sont tronqués).
        try:
            # Liste plate [x1, y1, x2, y2, ...] -> (N, 2), puis suppression des
            # doublons exacts en conservant l'ordre d'origine des sommets.
            polygon = np.array(segs[0], dtype=np.float32).reshape(-1, 2)
            _, idx = np.unique(polygon, axis=0, return_index=True)
            polygon = polygon[np.sort(idx)]

            if len(polygon) >= 3:
                # La segmentation est réécrite sans les doublons.
                ann["segmentation"] = [polygon.reshape(-1).tolist()]
                ann["bezier_pts"] = np.asarray(
                    to_bezier(polygon.tolist()), dtype=float
                ).reshape(-1, 2)
            else:
                logging.warning(
                    f"Annotation {ann.get('id')} : Polygone trop petit ({len(polygon)} pts)."
                )
        except Exception as e:
            # L'annotation est conservée sans `bezier_pts` ; main() le signale.
            logging.error(f"Erreur Bézier sur annotation {ann.get('id')}: {e}")

    # 2. Texte : `texts` est le champ source. Absent, vide ou blanc -> NO_TEXT,
    # sinon l'instance serait écartée à l'entraînement (rec tout à PAD_IDX).
    text_val = ann.get("texts", None)
    if text_val is None or not str(text_val).strip():
        text_val = NO_TEXT
    ann["rec"] = encode_text(str(text_val), char2idx)
    ann["text"] = text_val
    ann["language"] = "Latin"
    return ann


def main():
    parser = argparse.ArgumentParser(
        description="Enrichissement COCO : Points de Bézier + Encodage CTC."
    )
    parser.add_argument("input", type=str, help="JSON COCO d'entrée")
    parser.add_argument(
        "output",
        type=str,
        help="JSON de sortie (choisir un autre chemin que l'entrée pour ne pas l'écraser)",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="char_map/char2idx/Latin.json",
        help="Chemin vers le char2idx",
    )
    parser.add_argument(
        "--bezier-version",
        choices=sorted(BEZIER_FUNCS),
        default="v4",
        help="Conversion polygone -> Bézier (v4 par défaut)",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Conserve les annotations en double (même image, même polygone)",
    )
    parser.add_argument(
        "--no-indent",
        action="store_true",
        help="Désactive l'indentation pour réduire la taille du fichier",
    )

    args = parser.parse_args()

    try:
        char2idx = load_char2idx(args.map)
    except Exception as e:
        logging.error(e)
        return

    logging.info(f"Chargement de {args.input}...")
    with open(args.input, "r") as f:
        coco = json.load(f)

    annotations = coco.get("annotations", [])
    logging.info(f"Traitement de {len(annotations)} annotations...")

    # Doublons exacts : même image, même catégorie, même polygone. Ils comptent
    # deux fois dans la vérité terrain de l'évaluation (l'un est toujours manqué).
    if not args.keep_duplicates:
        seen, unique = set(), []
        for ann in annotations:
            segs = ann.get("segmentation") or [[]]
            key = (ann.get("image_id"), ann.get("category_id"),
                   tuple(np.round(segs[0], 1)), tuple(ann.get("bbox", [])))
            if key in seen:
                logging.warning(f"Annotation {ann.get('id')} supprimée (doublon).")
                continue
            seen.add(key)
            unique.append(ann)
        annotations = unique

    to_bezier = BEZIER_FUNCS[args.bezier_version]
    enriched_annotations = []
    for ann in tqdm(annotations):
        enriched_annotations.append(process_annotation(ann, char2idx, to_bezier))

    coco["annotations"] = enriched_annotations

    missing = [a.get("id") for a in enriched_annotations if "bezier_pts" not in a]
    if missing:
        logging.warning(
            f"{len(missing)} annotation(s) sans bezier_pts (non utilisables à "
            f"l'entraînement) : {missing[:20]}"
        )

    logging.info(f"Sauvegarde vers {args.output}...")
    indent = None if args.no_indent else 2
    with open(args.output, "w") as f:
        json.dump(coco, f, indent=indent, cls=NumpyEncoder)

    logging.info("✔️ Terminé avec succès.")


if __name__ == "__main__":
    main()
