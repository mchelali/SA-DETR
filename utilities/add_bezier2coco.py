import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from tqdm import tqdm

from utilities.curves import (
    robust_polygon_to_bezier,
    robust_polygon_to_bezier_v2,
    robust_polygon_to_bezier_v3,
)

# Import de tes fonctions spécialisées
try:
    from curves import quad_to_bezier, sort_polygon_clockwise
except ImportError:
    logging.error(
        "Le module 'curves' est introuvable. Assurez-vous qu'il est dans le PYTHONPATH."
    )
    raise

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MAX_LEN = 25
PAD_IDX = 0


class NumpyEncoder(json.JSONEncoder):
    """Encodeur spécial pour convertir les types NumPy en types Python natifs."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


def load_char2idx(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Le fichier de mapping de caractères est introuvable : {path}"
        )
    with open(path, encoding="utf-8") as f:
        return {k: int(v) for k, v in json.load(f).items()}


def encode_text(text: str, char2idx: Dict[str, int]) -> List[int]:
    """Encode le texte pour le CTC avec padding."""
    if not text:
        return [PAD_IDX] * MAX_LEN
    seq = [char2idx.get(ch, PAD_IDX) for ch in text[:MAX_LEN]]
    return seq + [PAD_IDX] * (MAX_LEN - len(seq))


def process_annotation(ann: Dict, char2idx: Dict[str, int]) -> Dict:
    """Enrichit une seule annotation avec Bézier et encodage texte."""
    segs = ann.get("segmentation", [])
    bbox = ann.get("bbox", [])
    if len(segs) == 0 and (bbox and len(bbox) < 4):
        logging.warning(
            f"Annotation {ann.get('id')} ignorée (pas de segmentation/bbox)."
        )
        return ann
    if bbox and len(segs) == 0:
        x, y, w, h = bbox
        segs = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        ann["segmentation"] = segs

    # 1. Traitement Bézier
    if segs and isinstance(segs, list) and len(segs) > 0:
        # On traite le premier polygone (cas standard)
        try:
            # Conversion en numpy et remise en forme
            polygon = np.array(segs[0], dtype=np.float32).reshape(-1, 2)
            _, idx = np.unique(polygon, axis=0, return_index=True)
            polygon = polygon[np.sort(idx)]

            # Un polygone nécessite au moins 3 points (ou 4 pour quad_to_bezier selon l'implémentation)
            if len(polygon) >= 3:
                # polygon = sort_polygon_clockwise(polygon)
                ann["segmentation"] = [polygon.reshape(-1).tolist()]
                # ann["bezier_pts"] = (
                #     quad_to_bezier(polygon.tolist()).reshape(-1).tolist()
                # )
                ann["bezier_pts"] = robust_polygon_to_bezier_v3(polygon.tolist())
            else:
                logging.warning(
                    f"Annotation {ann.get('id')} : Polygone trop petit ({len(polygon)} pts)."
                )
        except Exception as e:
            logging.error(f"Erreur Bézier sur annotation {ann.get('id')}: {e}")
            # exit(-1)

    # 2. Traitement Texte
    text_val = ann.get("texts", None)
    if text_val is not None:
        ann["rec"] = encode_text(str(text_val), char2idx)
    else:
        ann["rec"] = encode_text("###", char2idx)

    ann["text"] = text_val if text_val is not None else "###"
    ann["language"] = "Latin"
    return ann


def main():
    parser = argparse.ArgumentParser(
        description="Enrichissement COCO : Points de Bézier + Encodage CTC."
    )
    parser.add_argument("input", type=str, help="JSON COCO d'entrée")
    parser.add_argument("output", type=str, help="JSON de sortie")
    parser.add_argument(
        "--map",
        type=str,
        default="char_map/char2idx/Latin.json",
        help="Chemin vers le char2idx",
    )
    parser.add_argument(
        "--no-indent",
        action="store_true",
        help="Désactive l'indentation pour réduire la taille du fichier",
    )

    args = parser.parse_args()

    # Chargement des ressources
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

    # Boucle de traitement avec barre de progression
    enriched_annotations = []
    for ann in tqdm(annotations):
        enriched_annotations.append(process_annotation(ann, char2idx))

    coco["annotations"] = enriched_annotations

    # Sauvegarde
    logging.info(f"Sauvegarde vers {args.output}...")
    indent = None if args.no_indent else 2
    with open(args.output, "w") as f:
        json.dump(coco, f, indent=indent, cls=NumpyEncoder)

    logging.info("✔️ Terminé avec succès.")


if __name__ == "__main__":
    main()
