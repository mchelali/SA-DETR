"""
Training script that runs k-fold cross-validation using the same Trainer/setup
logic as `train_net.py`.

Usage example:
  python train_kfold.py --config-file configs/your.yaml --folds-dir /path/to/folds \
      --image-root /path/to/images --n-folds 5

This script expects COCO annotation files named:
  annotations_train_fold{fold}.json
  annotations_val_fold{fold}.json
inside `--folds-dir` for fold in [0..n_folds-1].
"""

import sys
import os
import json
import argparse
from statistics import mean
from typing import Dict, Any, List

import numpy as np
import torch

# Add the parent directory to sys.path to import from tools/train_net.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adet.data.datasets.text import register_stamp_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_argument_parser, launch

# Import shared setup and Trainer from the local train_net.py
from train_net import setup, Trainer
from adet.config import get_cfg


def get_classes_from_coco(json_path):
    """
    Extrait la liste des noms de classes depuis un fichier COCO JSON.
    Trie les classes par ID pour garantir le bon mapping.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extraire les catégories
    categories = data.get("categories", [])

    # Trier par ID pour que l'index dans la liste corresponde à l'ID COCO
    # (Detectron2 mappe souvent les IDs pour qu'ils soient contigus à partir de 0)
    categories.sort(key=lambda x: x["id"])

    class_names = [cat["name"] for cat in categories]
    return class_names


def register_fold_datasets(
    folds_dir: str,
    image_root: str,
    base_name: str,
    n_folds: int,
    num_pts_cfg=25,
):

    for fold in range(n_folds):
        train_ann = os.path.join(folds_dir, f"train_fold_{fold}_single.json")
        val_ann = os.path.join(folds_dir, f"val_fold_{fold}_single.json")
        test_ann = os.path.join(folds_dir, f"test_fold_{fold}_single.json")

        metadata = {
            "thing_classes": get_classes_from_coco(train_ann),
        }

        train_name = f"{base_name}_fold{fold}_train"
        val_name = f"{base_name}_fold{fold}_val"
        test_name = f"{base_name}_fold{fold}_test"

        if not os.path.isfile(train_ann) or not os.path.isfile(val_ann):
            print(f"⚠️ Fold {fold} ignoré : fichiers manquants.")
            continue

        # --- UTILISATION DU REGISTRE ADET AU LIEU DE DETECTRON2 ---

        # Enregistrement du Train name, metadata, json_file, image_root, num_pts_cfg
        register_stamp_instances(
            name=train_name,
            metadata=metadata,
            json_file=train_ann,
            image_root=image_root,
            num_pts_cfg=num_pts_cfg,
        )

        # Enregistrement du Val
        register_stamp_instances(
            name=val_name,
            metadata=metadata,
            json_file=val_ann,
            image_root=image_root,
            num_pts_cfg=num_pts_cfg,
        )
        # Enregistrement du Test
        register_stamp_instances(
            name=test_name,
            metadata=metadata,
            json_file=test_ann,
            image_root=image_root,
            num_pts_cfg=num_pts_cfg,
        )

        print(
            f"✅ Registered {train_name}, {val_name}, and {test_name} with {num_pts_cfg} Bezier points."
        )


def aggregate_metrics(
    results_list: List[Dict[str, Any]], num_folds: int = 5, prefix: str = ""
) -> Dict[str, float]:
    """
    Agrège des métriques numériques sur N folds (par défaut 5).
    Retourne moyenne + écart-type pour chaque métrique.

    Args:
        results_list: liste de dicts (1 par fold)
        num_folds: nombre attendu de folds
        prefix: préfixe optionnel pour les clés de sortie

    Returns:
        Dict avec clés:
          - metric
          - metric_std
    """

    values = {}

    for fold_id, res in enumerate(results_list):
        if not isinstance(res, dict):
            continue

        # --- flatten récursif (1 niveau suffit en pratique)
        flat = {}
        for k, v in res.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}/{kk}"] = vv
            else:
                flat[k] = v

        # --- collecte des valeurs
        for k, v in flat.items():
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue

            values.setdefault(k, []).append(val)

    # --- agrégation
    aggregated = {}
    for k, vals in values.items():
        vals = np.array(vals, dtype=float)

        if len(vals) == 0:
            continue

        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        key = f"{prefix}{k}" if prefix else k
        aggregated[key] = mean
        aggregated[f"{key}_std"] = std
        aggregated[f"{key}_n"] = len(vals)

    return aggregated


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="K-fold cross-validation training script"
    )
    # Add custom arguments first
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument(
        "--folds-dir", required=True, help="Directory with per-fold COCO annotations"
    )
    parser.add_argument(
        "--image-root",
        required=True,
        help="Root folder containing images referenced in annotations",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds to run")
    parser.add_argument(
        "--base-name",
        type=str,
        default="forbin_stamp",
        help="Base name for registered datasets",
    )
    # parser.add_argument(
    #     "--thing-classes",
    #     type=str,
    #     nargs="*",
    #     default=None,
    #     help="(optional) class names for MetadataCatalog",
    # )

    # Add detectron2 standard arguments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory.",
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )

    # Port for distributed training
    import sys

    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend.",
    )

    # opts for additional config overrides
    parser.add_argument(
        "opts",
        help="Modify config options at the end of the command. For Yacs configs, use space-separated 'PATH.KEY VALUE' pairs.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def main(args):
    # Prepare base cfg and logging by calling shared setup
    cfg_base = setup(args)

    # Register datasets for all folds
    register_fold_datasets(
        args.folds_dir,
        args.image_root,
        args.base_name,
        args.n_folds,
    )
    test_dataset = "StaVer_test"

    fold_results = []
    for fold in range(args.n_folds):
        print(f"\n===== Fold {fold} / {args.n_folds} =====")
        cfg = cfg_base.clone()
        cfg.defrost()
        cfg.DATASETS.TRAIN = (f"{args.base_name}_fold{fold}_train",)
        cfg.DATASETS.TEST = (f"{args.base_name}_fold{fold}_val",)
        if args.eval_only:
            # cfg.DATASETS.TEST = (f"{args.base_name}_fold{fold}_test",)
            cfg.DATASETS.TEST = (f"{test_dataset}",)
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"forbin_stamps_fold{fold}")
        # cfg.MODEL.DEVICE = "cpu"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        # Optionally set a different seed per fold for reproducibility
        if hasattr(cfg, "SEED"):
            cfg.SEED = getattr(cfg, "SEED", 42) + fold
        cfg.freeze()

        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        if cfg.TEST.AUG.ENABLED:
            trainer.register_hooks(
                [
                    trainer.hooks.EvalHook(
                        0, lambda: trainer.test_with_TTA(cfg, trainer.model)
                    )
                ]
            )

        if args.eval_only:
            best_model = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")
            print("Evaluating using model:", best_model)
            trainer.model.load_state_dict(
                torch.load(best_model, map_location=torch.device("cpu"))["model"]
            )
            res = trainer.test(cfg, trainer.model)
            if cfg.TEST.AUG.ENABLED:
                res.update(trainer.test_with_TTA(cfg, trainer.model))
            fold_results.append(res)
            continue

        res = trainer.train()
        # If evaluation ran, trainer.train returns results
        if (
            res is None
            and hasattr(trainer, "_last_eval_results")
            and trainer._last_eval_results is not None
        ):
            res = trainer._last_eval_results
        fold_results.append(res or {})

    # Aggregate
    avg = aggregate_metrics(fold_results)
    print("\n===== Cross-validation aggregate metrics =====")
    print(json.dumps(avg, indent=2))

    # Save aggregated metrics
    if args.eval_only:
        out_file = os.path.join(
            cfg_base.OUTPUT_DIR, f"{test_dataset}_cv_aggregate_metrics.json"
        )
    else:
        out_file = os.path.join(cfg_base.OUTPUT_DIR, "cv_aggregate_metrics_val.json")
    with open(out_file, "w", encoding="utf8") as f:
        json.dump({"folds": args.n_folds, "average": avg}, f, indent=2)
    print(f"Saved aggregate metrics to {out_file}")


if __name__ == "__main__":
    args = parse_args()
    # Use detectron2's launch utility to support distributed training similarly to train_net.py
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
