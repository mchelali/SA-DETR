# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os
import time

import numpy as np
import torch
import tqdm

from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.structures.boxes import Boxes
from detectron2.utils.logger import setup_logger

from adet.config import get_cfg
from predictor import VisualizationDemo


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    set_score_threshold(cfg, args.confidence_threshold)
    cfg.freeze()
    return cfg


def set_score_threshold(cfg, threshold):
    cfg.defrost()
    for key in (
        ("MODEL", "RETINANET", "SCORE_THRESH_TEST"),
        ("MODEL", "ROI_HEADS", "SCORE_THRESH_TEST"),
        ("MODEL", "FCOS", "INFERENCE_TH_TEST"),
        ("MODEL", "MEInst", "INFERENCE_TH_TEST"),
        ("MODEL", "PANOPTIC_FPN", "COMBINE", "INSTANCES_CONFIDENCE_THRESH"),
    ):
        node = cfg
        try:
            for part in key[:-1]:
                node = getattr(node, part)
            if key[-1] in node:
                node[key[-1]] = threshold
        except AttributeError:
            pass


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run inference on a COCO json and save detections in COCO result format."
    )
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--json-input",
        required=True,
        help="COCO json file containing at least an 'images' list.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help=(
            "directory used to resolve image file_name values. "
            "Defaults to the directory containing --json-input."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path of the output COCO detections json.",
    )
    parser.add_argument(
        "--output-format",
        choices=("results", "dataset"),
        default="results",
        help=(
            "'results' writes the standard COCO detection-result list. "
            "'dataset' writes a COCO-like dataset json with detections as annotations."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="minimum score for instance predictions.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="override MODEL.DEVICE, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--category-id-mode",
        choices=("auto", "metadata", "json", "contiguous"),
        default="auto",
        help=(
            "how to convert model contiguous class ids to COCO category ids. "
            "'auto' uses metadata when available, otherwise categories from the input json."
        ),
    )
    parser.add_argument(
        "--opts",
        help="modify config options using command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def resolve_image_path(image_root, file_name):
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(image_root, file_name)


def ensure_pred_boxes(instances):
    if len(instances) == 0:
        return instances

    if instances.has("bd"):
        box_points = instances.bd
        if not isinstance(box_points, torch.Tensor):
            box_points = torch.as_tensor(np.asarray(box_points))

        if box_points.dim() == 3 and box_points.size(2) == 4:
            x = box_points[..., [0, 2]].reshape(box_points.shape[0], -1)
            y = box_points[..., [1, 3]].reshape(box_points.shape[0], -1)
            boxes = torch.stack(
                [
                    x.min(dim=1).values,
                    y.min(dim=1).values,
                    x.max(dim=1).values,
                    y.max(dim=1).values,
                ],
                dim=1,
            )
            instances.pred_boxes = Boxes(boxes)
            return instances

        if box_points.dim() == 2 and box_points.size(1) == 4:
            x = box_points[:, [0, 2]]
            y = box_points[:, [1, 3]]
            boxes = torch.stack(
                [
                    x.min(dim=1).values,
                    y.min(dim=1).values,
                    x.max(dim=1).values,
                    y.max(dim=1).values,
                ],
                dim=1,
            )
            instances.pred_boxes = Boxes(boxes)
            return instances

        raise ValueError(
            "Cannot infer pred_boxes from bd: expected shape (N, M, 4) or (N, 4)."
        )

    if instances.has("pred_boxes"):
        return instances
    if instances.has("proposal_boxes"):
        instances.pred_boxes = instances.proposal_boxes
        return instances
    if instances.has("ctrl_points"):
        box_points = instances.ctrl_points
        if not isinstance(box_points, torch.Tensor):
            box_points = torch.as_tensor(np.asarray(box_points))
        if box_points.dim() == 2 and box_points.size(1) % 2 == 0:
            x = box_points[:, 0::2]
            y = box_points[:, 1::2]
            boxes = torch.stack(
                [
                    x.min(dim=1).values,
                    y.min(dim=1).values,
                    x.max(dim=1).values,
                    y.max(dim=1).values,
                ],
                dim=1,
            )
            instances.pred_boxes = Boxes(boxes)
            return instances
        raise ValueError(
            "Cannot infer pred_boxes from ctrl_points: expected shape (N, 2K)."
        )

    return instances


def bd_to_segmentation(bd_points):
    bd = np.asarray(bd_points)
    if bd.ndim != 2 or bd.shape[1] != 4:
        raise ValueError("Cannot build segmentation from bd: expected shape (M, 4).")
    bd_split = np.hsplit(bd, 2)
    polygon = np.vstack([bd_split[0], bd_split[1][::-1]])
    return polygon.flatten().tolist()


def instances_to_coco_json_with_bd(instances, img_id):
    results = instances_to_coco_json(instances, img_id)
    if len(results) == 0:
        return results

    bd = instances.bd if instances.has("bd") else None
    ctrl_points = instances.ctrl_points if instances.has("ctrl_points") else None

    if bd is not None:
        bd = np.asarray(bd)
    if ctrl_points is not None:
        ctrl_points = np.asarray(ctrl_points)

    for idx, result in enumerate(results):
        if bd is not None:
            result["bd"] = bd[idx].tolist()
            result["segmentation"] = bd_to_segmentation(bd[idx])
        if ctrl_points is not None:
            result["ctrl_points"] = ctrl_points[idx].tolist()
    return results


def get_category_id_mapping(cfg, coco_data, mode):
    if mode in ("auto", "metadata") and len(cfg.DATASETS.TEST):
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
            return {
                contiguous_id: dataset_id
                for dataset_id, contiguous_id in metadata.thing_dataset_id_to_contiguous_id.items()
            }
        if mode == "metadata":
            raise ValueError(
                "No thing_dataset_id_to_contiguous_id metadata was found for "
                f"{cfg.DATASETS.TEST[0]}"
            )

    if mode in ("auto", "json"):
        categories = sorted(coco_data.get("categories", []), key=lambda x: x["id"])
        if categories:
            return {idx: category["id"] for idx, category in enumerate(categories)}
        if mode == "json":
            raise ValueError(
                "The input json does not contain a non-empty 'categories' list."
            )

    return None


def remap_category_ids(results, category_id_mapping):
    if category_id_mapping is None:
        return results

    remapped = []
    for result in results:
        result = result.copy()
        category_id = result["category_id"]
        if category_id not in category_id_mapping:
            raise ValueError(
                f"Predicted class id {category_id} is not present in the category mapping."
            )
        result["category_id"] = category_id_mapping[category_id]
        remapped.append(result)
    return remapped


def polygon_area(segmentation):
    if not segmentation:
        return 0.0
    if isinstance(segmentation[0], list):
        coords = np.asarray(segmentation[0], dtype=np.float64)
    else:
        coords = np.asarray(segmentation, dtype=np.float64)
    if coords.ndim != 1 or coords.size < 6:
        return 0.0
    coords = coords.reshape(-1, 2)
    x = coords[:, 0]
    y = coords[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def result_area(result):
    if "segmentation" in result:
        if isinstance(result["segmentation"], dict):
            from pycocotools import mask as mask_util

            return float(mask_util.area(result["segmentation"]))
        if isinstance(result["segmentation"], list):
            return polygon_area(result["segmentation"])
    return float(result["bbox"][2] * result["bbox"][3])


def to_dataset_format(coco_data, results):
    annotations = []
    for idx, result in enumerate(results, start=1):
        annotation = result.copy()
        annotation["id"] = idx
        annotation["area"] = result_area(result)
        annotation["iscrowd"] = 0
        annotations.append(annotation)

    return {
        "images": coco_data.get("images", []),
        "categories": coco_data.get("categories", []),
        "annotations": annotations,
    }


def main():
    parser = get_parser()
    args, extra_opts = parser.parse_known_args()
    args.opts.extend(extra_opts)
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    with open(args.json_input, "r") as f:
        coco_data = json.load(f)

    if "images" not in coco_data:
        raise ValueError(f"{args.json_input} does not contain an 'images' list.")

    cfg = setup_cfg(args)
    if args.device:
        cfg.defrost()
        cfg.MODEL.DEVICE = args.device
        cfg.freeze()

    image_root = args.image_root or os.path.dirname(os.path.abspath(args.json_input))
    category_id_mapping = get_category_id_mapping(cfg, coco_data, args.category_id_mode)
    demo = VisualizationDemo(cfg)

    all_results = []
    for image_info in tqdm.tqdm(coco_data["images"]):
        image_path = resolve_image_path(image_root, image_info["file_name"])
        image = read_image(image_path, format="BGR")

        start_time = time.time()
        predictions, _ = demo.run_on_image(image)
        instances = predictions.get("instances")
        if instances is None or len(instances) == 0:
            results = []
        else:
            instances = instances.to(torch.device("cpu"))
            instances = ensure_pred_boxes(instances)
            results = instances_to_coco_json_with_bd(instances, image_info["id"])
            results = remap_category_ids(results, category_id_mapping)
            all_results.extend(results)

        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                image_path, len(results), time.time() - start_time
            )
        )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_data = (
        all_results
        if args.output_format == "results"
        else to_dataset_format(coco_data, all_results)
    )
    with open(args.output, "w") as f:
        json.dump(output_data, f)

    logger.info("Saved {} detections to {}".format(len(all_results), args.output))


if __name__ == "__main__":
    main()
