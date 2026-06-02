import json
import os
import argparse
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from .datasets.text import register_text_instances, register_stamp_instances
from adet.config import get_cfg
from detectron2.engine import default_argument_parser

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {"thing_classes": ["person"]}

_PREDEFINED_SPLITS_TEXT = {
    "mlt19_train": ("mlt19/train_images", "mlt19/mlt19_train.json"),
    "Arabic": ("Arabic/train_images", "Arabic/train.json"),
    "Bangla": ("Bangla/train_images", "Bangla/train.json"),
    "Chinese": ("Chinese/train_images", "Chinese/train.json"),
    "Hindi": ("Hindi/train_images", "Hindi/train.json"),
    "Japanese": ("Japanese/train_images", "Japanese/train.json"),
    "Korean": ("Korean/train_images", "Korean/train.json"),
    "Latin": ("Latin/train_images", "Latin/train.json"),
    "RCTW": ("RCTW/train_images", "RCTW/train.json"),
    "ArT": ("ArT/rename_artimg_train", "ArT/train.json"),
    "LSVT": ("LSVT/rename_lsvtimg_train", "LSVT/train.json"),
    "ic15_train": ("ic15/train_images", "ic15/train_37voc.json"),
    "ic15_test": ("ic15/test_images", "ic15/test.json"),
    "icdar2015_train": (
        "icdar2015/ch4_training_images",
        "icdar2015/icdar2015_train.json",
    ),
    "icdar2015_test": (
        "icdar2015/ch4_training_images",
        "icdar2015/icdar2015_test.json",
    ),
    "icdar2015_val": (
        "icdar2015/ch4_training_images",
        "icdar2015/icdar2015_val.json",
    ),
    # synthetic dataset
    "Synthetic_train": (
        "synthetic_dataset/images",
        "synthetic_dataset/train.json",
    ),
    "Synthetic_val": (
        "synthetic_dataset/images/",
        "synthetic_dataset/val.json",
    ),
    "Synthetic_test": (
        "synthetic_dataset/images/",
        "synthetic_dataset/test.json",
    ),
    # Forbin dataset
    # "forbin_train": (
    #     "Forbin/Fichiers_de_diffusion",
    #     "Forbin/arkindex_train.json",
    # ),
    # "forbin_val": (
    #     "Forbin/Fichiers_de_diffusion/",
    #     "Forbin/arkindex_val.json",
    # ),
    # "forbin_test": (
    #     "Forbin/Fichiers_de_diffusion/",
    #     "Forbin/arkindex_test.json",
    # ),
    # Forbin Stamps
    # "forbin_stamps_train": (
    #     "forbin_dataset/images/",
    #     "forbin_dataset/train_fold_0_single.json",
    # ),
    # "forbin_stamps_val": (
    #     "forbin_dataset/images/",
    #     "forbin_dataset/val_fold_0_single.json",
    # ),
    # "forbin_stamps_test": (
    #     "forbin_dataset/images/",
    #     "forbin_dataset/test_fold_0_single.json",
    # ),
    # StaVer Dataset
    "StaVer_train": (
        "StaVer_dataset/train/",
        "StaVer_dataset/train/_annotations.coco.json",
    ),
    "StaVer_val": (
        "StaVer_dataset/valid/",
        "StaVer_dataset/valid/_annotations.coco.json",
    ),
    "StaVer_test": (
        "StaVer_dataset/test/",
        "StaVer_dataset/test/_annotations.coco.json",
    ),
    # Historical_Postcards_Dataset_v1
    "hist_postcard_train": (
        "Historical_Postcards/Historical_Postcards_Dataset_v1-Train2025/images/Train/",
        "Historical_Postcards/processed_stamp/instances_Train_stamp_train.json",
    ),
    "hist_postcard_val": (
        "Historical_Postcards/Historical_Postcards_Dataset_v1-Train2025/images/Train/",
        "Historical_Postcards/processed_stamp/instances_Train_stamp_val.json",
    ),
    "hist_postcard_test": (
        "Historical_Postcards/Historical_Postcards_Dataset_v1-Test2025/images/Test/",
        "Historical_Postcards/processed_stamp/instances_Test_stamp.json",
    ),
    # evaluation, just for reading images, annotations may be empty
    "mlt19_test": ("mlt19/test_images", "mlt19/mlt19_test.json"),
    "mlt17_test": ("mlt17/test_images", "mlt17/mlt17_test.json"),
}

metadata_text = {"thing_classes": ["text"]}


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


metadata_text_stamp = {
    "thing_classes": get_classes_from_coco(
        f'datasets/{_PREDEFINED_SPLITS_TEXT["StaVer_train"][1]}'
    )
}


def register_all_coco(root="datasets", num_pts_cfg=25):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        if "StaVer" in key:
            register_stamp_instances(
                key,
                metadata_text_stamp,
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                num_pts_cfg,
            )
        else:
            register_text_instances(
                key,
                metadata_text,
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                num_pts_cfg,
            )


# get the vocabulary size and number of point queries in each instance
# to eliminate blank text and sample gt according to Bezier control points


# Only parse arguments and register datasets if running as main script
def _register_builtin_datasets():
    try:
        parser = default_argument_parser()
        # add the following argument to avoid some errors while running demo/demo.py
        parser.add_argument(
            "--input", nargs="+", help="A list of space separated input images"
        )
        parser.add_argument(
            "--output",
            help="A file or directory to save output visualizations. "
            "If not given, will show output in an OpenCV window.",
        )
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )
        args = parser.parse_args()
        cfg = get_cfg()
        if args.config_file:
            cfg.merge_from_file(args.config_file)
        # Only register if config was successfully loaded
        if hasattr(cfg.MODEL, "TRANSFORMER") and hasattr(
            cfg.MODEL.TRANSFORMER, "NUM_POINTS"
        ):
            register_all_coco(num_pts_cfg=cfg.MODEL.TRANSFORMER.NUM_POINTS)
        else:
            # Default registration without custom NUM_POINTS
            register_all_coco(num_pts_cfg=25)
    except SystemExit:
        # If argument parsing fails (e.g., when imported as a library), register with defaults
        register_all_coco(num_pts_cfg=25)
    except Exception as e:
        # Fallback: register with default values
        try:
            register_all_coco(num_pts_cfg=25)
        except Exception:
            pass


# Register datasets with default values immediately (safe for imports)
register_all_coco(num_pts_cfg=25)
