import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import re
import torch
from collections import OrderedDict, defaultdict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
import glob
import shutil
from shapely.geometry import Polygon, LinearRing
import zipfile
import pickle
import editdistance
import cv2
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def safe_polygon(coords):
    poly = Polygon(coords)

    if not poly.is_valid:
        poly = poly.buffer(0)  # fixe la plupart des self-intersections

    return poly


class TextEvaluator:
    """
    Evaluate text proposals and recognition.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._tasks = ("polygon", "recognition")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            raise AttributeError(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
            )
        self.voc_sizes = cfg.MODEL.TRANSFORMER.LANGUAGE.VOC_SIZES
        self.char_map = {}
        self.language_list = cfg.MODEL.TRANSFORMER.LANGUAGE.CLASSES
        for language_type, voc_size in self.voc_sizes:
            with open("char_map/idx2char/" + language_type + ".json") as f:
                idx2char = json.load(f)
            f.close()
            # index 0 is the background class
            assert len(idx2char) == voc_size
            self.char_map[language_type] = idx2char

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self.dataset_name = dataset_name
        self.submit = False
        # use dataset_name to decide eval_gt_path
        if "mlt19" in dataset_name:
            self.submit = True
            self._text_eval_gt_path = ""
            self.dataset_name = "mlt19"
        elif "mlt17" in dataset_name:
            self.submit = True
            self._text_eval_gt_path = ""
            self.dataset_name = "mlt17"
        elif "Synthetic" in dataset_name:
            self.submit = True
            self._text_eval_gt_path = ""
            self.dataset_name = "Synthetic"
        elif "forbin" in dataset_name:
            self.submit = True
            self._text_eval_gt_path = ""
            self.dataset_name = "forbin"
        elif "icdar2015" in dataset_name:
            self.submit = True
            self._text_eval_gt_path = ""
            self.dataset_name = "icdar2015"
        elif "StaVer" in dataset_name:
            self.submit = True
            self._text_eval_gt_path = ""
            self.dataset_name = "StaVer"
        elif "hist_postcard" in dataset_name:
            self.submit = True
            self._text_eval_gt_path = ""
            self.dataset_name = "hist_postcard"
        else:
            raise NotImplementedError

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            txt_name = (
                input["file_name"].split("/")[-1].split(".")[0].replace("ts", "res")
                + ".txt"
            )
            prediction = {"image_id": input["image_id"], "txt_name": txt_name}
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = self.instances_to_coco_json(instances, input)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if len(predictions) == 0:
            self._logger.warning("No valid predictions.")
            return {}
        PathManager.mkdirs(self._output_dir)

        if self.submit:
            self._logger.info("Saving results to {}".format(self._output_dir))
            if self.dataset_name == "mlt19":
                # for mlt19 task4: e2e text detection and recognition
                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, confidence = inst["polys"], inst["score"]
                                if confidence < 0.4:
                                    continue  # 0.4 for e2e task
                                write_lan, write_text = inst["language"], inst["rec"]
                                write_poly = ",".join(list(map(str, write_poly)))
                                f.write(
                                    write_poly
                                    + ","
                                    + str(confidence)
                                    + ","
                                    + write_text
                                    + "\n"
                                )
                        f.flush()
                zip_name = os.path.join(self._output_dir, "mlt19_task4.zip")
                os.system(
                    "zip -rqj "
                    + zip_name
                    + " "
                    + os.path.join(self._output_dir, "*.txt")
                )

                # for mlt19 task3: joint text detection and script identification
                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, confidence = inst["polys"], str(
                                    inst["score"]
                                )
                                write_lan, write_text = inst["language"], inst["rec"]
                                write_poly = ",".join(list(map(str, write_poly)))
                                f.write(
                                    write_poly
                                    + ","
                                    + confidence
                                    + ","
                                    + write_lan
                                    + "\n"
                                )
                        f.flush()
                zip_name = os.path.join(self._output_dir, "mlt19_task3.zip")
                os.system(
                    "zip -rqj "
                    + zip_name
                    + " "
                    + os.path.join(self._output_dir, "*.txt")
                )

                # for mlt19 task1: multi-script text detection
                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, confidence = inst["polys"], str(
                                    inst["score"]
                                )
                                write_lan, write_text = inst["language"], inst["rec"]
                                write_poly = ",".join(list(map(str, write_poly)))
                                f.write(write_poly + "," + confidence + "\n")
                        f.flush()
                zip_name = os.path.join(self._output_dir, "mlt19_task1.zip")
                os.system(
                    "zip -rqj "
                    + zip_name
                    + " "
                    + os.path.join(self._output_dir, "*.txt")
                )
                os.system("rm -rf " + os.path.join(self._output_dir, "*.txt"))
            elif self.dataset_name == "mlt17":
                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, confidence = inst["polys"], str(
                                    inst["score"]
                                )
                                write_lan, write_text = inst["language"], inst["rec"]
                                if write_lan == "Hindi":
                                    continue
                                write_poly = ",".join(list(map(str, write_poly)))
                                f.write(
                                    write_poly
                                    + ","
                                    + confidence
                                    + ","
                                    + write_lan
                                    + "\n"
                                )

                        f.flush()
                zip_name = os.path.join(self._output_dir, "mlt17_task3.zip")
                os.system(
                    "zip -rqj "
                    + zip_name
                    + " "
                    + os.path.join(self._output_dir, "*.txt")
                )

                for prediction in tqdm(predictions):
                    file_path = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(file_path, "w") as f:
                        if len(prediction["instances"]) > 0:
                            for inst in prediction["instances"]:
                                write_poly, confidence = inst["polys"], str(
                                    inst["score"]
                                )
                                write_lan, write_text = inst["language"], inst["rec"]
                                write_poly = ",".join(list(map(str, write_poly)))
                                f.write(write_poly + "," + confidence + "\n")
                        f.flush()
                zip_name = os.path.join(self._output_dir, "mlt17_task1.zip")
                os.system(
                    "zip -rqj "
                    + zip_name
                    + " "
                    + os.path.join(self._output_dir, "*.txt")
                )
                os.system("rm -rf " + os.path.join(self._output_dir, "*.txt"))
            elif (
                self.dataset_name == "Synthetic"
                or self.dataset_name == "forbin"
                or self.dataset_name == "StaVer"
                or self.dataset_name == "icdar2015"
                or self.dataset_name == "hist_postcard"
            ):
                # ----------------------------------------------------------
                #   Synthetic : on génère UN zip avec toutes les prédictions
                #   (texte + langue) – même format que MLT19 task4
                # ----------------------------------------------------------
                print("Inférence Evaluation Synthetic Dataset")
                for prediction in tqdm(predictions):
                    fp = os.path.join(self._output_dir, prediction["txt_name"])
                    with PathManager.open(fp, "w") as f:
                        for inst in prediction["instances"]:
                            poly, score = inst["polys"], inst["score"]
                            # if score < 0.4:  # garde le même seuil
                            #     continue
                            lan, txt = inst["language"], inst["rec"]
                            poly = ",".join(map(str, poly))
                            f.write(f"{poly},{score},{txt}\n")
                        f.flush()
                zip_name = os.path.join(self._output_dir, "synthetic_task4.zip")
                os.system(f"zip -rqj {zip_name} {self._output_dir}/*.txt")
                os.system(f"rm -rf {self._output_dir}/*.txt")
            else:
                raise NotImplementedError

            self._logger.info(
                "Ready to submit results from {}".format(self._output_dir)
            )

        # --------------------------------------------------------------------------
        # 1. Charge les ground-truth depuis le JSON COCO
        # --------------------------------------------------------------------------
        coco_json = PathManager.get_local_path(self._metadata.json_file)
        with open(coco_json) as f:
            coco = json.load(f)

        gt_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            # print(ann)
            # exit(-1)
            if ann["iscrowd"] or ann["category_id"] != 1:
                continue
            img_id = ann["image_id"]
            try:
                poly = np.array(ann["bezier_pts"]).reshape(-1, 2)
            except:
                print(img_id)
                print(ann)

            gt_by_img[img_id].append(
                {
                    "poly": safe_polygon(poly),
                    "txt": self.ctc_decode(ann["rec"], ann["language"]),
                }
            )

        coco_results = []
        for pred in predictions:
            img_id = pred["image_id"]
            for inst in pred["instances"]:
                # On prépare l'objet pour pycocotools
                poly = [float(x) for x in inst["polys"]]
                xs = poly[0::2]
                ys = poly[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                segmentation = [poly]
                coco_results.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(inst.get("category_id", 1)),
                        "segmentation": segmentation,  # Format polygone
                        "bbox": [
                            x_min,
                            y_min,
                            x_max - x_min,
                            y_max - y_min,
                        ],  # Format COCO bbox
                        "score": float(inst["score"]),
                        # "area": self._calculate_area(inst["polys"]),
                        "iscrowd": 0,
                    }
                )
        pred_json_path = os.path.join(self._output_dir, "coco_predictions.json")
        with open(pred_json_path, "w") as f:
            json.dump(coco_results, f, indent=2)

        coco_stats = [0.0] * 12

        if len(coco_results) > 0:
            try:
                coco_gt = COCO(coco_json)
                coco_dt = coco_gt.loadRes(pred_json_path)
                coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                coco_stats = coco_eval.stats
            except Exception as e:
                print(
                    f"⚠️ Erreur lors de l'évaluation COCO (possiblement peu de données) : {e}"
                )
        else:
            print(
                "ℹ️ Aucune prédiction détectée à cette époque. Les scores COCO sont mis à zéro."
            )

        # --------------------------------------------------------------------------
        # 2. Boucle d’évaluation
        # --------------------------------------------------------------------------
        tp_det = fp_det = fn_det = 0
        tp_e2e = fp_e2e = fn_e2e = 0

        # Si predictions est vide, tp/fp restent à 0, et fn sera égal à la somme des GT
        if len(predictions) == 0:
            # On calcule au moins les False Negatives (tous les GT manqués)
            for img_id, gt_list in gt_by_img.items():
                fn_det += len(gt_list)
                fn_e2e += len(gt_list)
        else:
            for pred in predictions:
                img_id = pred["image_id"]
                gt_list = gt_by_img.get(img_id, [])
                matched = set()

                for inst in pred["instances"]:
                    poly_pred = Polygon(np.array(inst["polys"]).reshape(-1, 2))
                    txt_pred = inst["rec"].strip().lower()

                    best_iou, best_idx = 0, -1
                    for j, g in enumerate(gt_list):
                        if j in matched:
                            continue

                        iou = (
                            poly_pred.intersection(g["poly"]).area
                            / g["poly"].union(poly_pred).area
                        )
                        if iou > best_iou:
                            best_iou, best_idx = iou, j

                    if best_iou >= 0.5:
                        tp_det += 1
                        # end-to-end
                        if txt_pred == gt_list[best_idx]["txt"].strip().lower():
                            tp_e2e += 1
                        else:
                            fp_e2e += 1
                        matched.add(best_idx)
                    else:
                        fp_det += 1
                        fp_e2e += 1

                miss = len(gt_list) - len(matched)
                fn_det += miss
                fn_e2e += miss

        # --------------------------------------------------------------------------
        # 3. Calcul des métriques
        # --------------------------------------------------------------------------
        prec_det = tp_det / max(1, tp_det + fp_det)
        rec_det = tp_det / max(1, tp_det + fn_det)
        h_det = 2 * prec_det * rec_det / max(1e-7, prec_det + rec_det)

        prec_e2e = tp_e2e / max(1, tp_e2e + fp_e2e)
        rec_e2e = tp_e2e / max(1, tp_e2e + fn_e2e)
        h_e2e = 2 * prec_e2e * rec_e2e / max(1e-7, prec_e2e + rec_e2e)

        self._results = OrderedDict(
            {
                "det/hmean": h_det,
                "det/precision": prec_det,
                "det/recall": rec_det,
                "e2e/hmean": h_e2e,
                "e2e/precision": prec_e2e,
                "e2e/recall": rec_e2e,
            }
        )

        self._results["bbox/AP"] = coco_stats[0]
        self._results["bbox/AP50"] = coco_stats[1]
        self._results["bbox/AP75"] = coco_stats[2]
        self._results["bbox/APs"] = coco_stats[3]
        self._results["bbox/APm"] = coco_stats[4]
        self._results["bbox/APl"] = coco_stats[5]
        self._results["bbox/AR@1"] = coco_stats[6]
        self._results["bbox/AR@10"] = coco_stats[7]
        self._results["bbox/AR@100"] = coco_stats[8]
        self._results["bbox/ARs"] = coco_stats[9]
        self._results["bbox/ARm"] = coco_stats[10]
        self._results["bbox/ARl"] = coco_stats[11]
        return copy.deepcopy(self._results)

    def instances_to_coco_json(self, instances, inputs):
        img_id = inputs["image_id"]
        width = inputs["width"]
        height = inputs["height"]
        num_instances = len(instances)
        if num_instances == 0:
            return []

        scores = instances.scores.tolist()
        languages = instances.languages.tolist()
        pnts = instances.bd.numpy()
        recs = instances.recs
        results = []
        if recs != []:
            for pnt, rec, score, language in zip(pnts, recs, scores, languages):
                lan = self.language_list[language]
                poly = self.pnt_to_polygon(pnt)
                poly = polygon2rbox(
                    poly, height, width
                )  # only 4 points are required for MLT
                pgt = Polygon(poly)
                if not pgt.is_valid:
                    continue
                if not LinearRing(poly).is_ccw:
                    poly = poly[::-1]
                poly = poly.reshape(-1).tolist()
                s = self.ctc_decode(rec, lan)
                if lan == "Arabic":
                    s = s[::-1]
                if s == "":
                    continue
                result = {
                    "image_id": img_id,
                    "category_id": 1,
                    "polys": poly,
                    "rec": s,
                    "score": score,
                    "language": lan,
                }
                results.append(result)
        return results

    def pnt_to_polygon(self, ctrl_pnt):
        ctrl_pnt = np.hsplit(ctrl_pnt, 2)
        ctrl_pnt = np.vstack([ctrl_pnt[0], ctrl_pnt[1][::-1]])
        return ctrl_pnt.tolist()

    def ctc_decode(self, rec, lan):
        last_char = "###"
        s = ""
        for c in rec:
            c = int(c)
            if c != 0:
                if last_char != c:
                    s += self.char_map[lan][str(c)]
                    last_char = c
            else:
                last_char = "###"
        return s

    def _calculate_area(self, poly):
        """Calcule l'aire du polygone (requis par COCO)"""
        x = np.array(poly[0::2])
        y = np.array(poly[1::2])
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon2rbox(polygon, image_height, image_width):
    poly = np.array(polygon).reshape((-1, 2)).astype(np.float32)
    rect = cv2.minAreaRect(poly)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = np.array(pts).reshape(-1, 2)
    return pts


def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]
