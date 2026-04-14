# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Set
import torch
import itertools
from torch.nn.parallel import DistributedDataParallel


import detectron2.utils.comm as comm
from detectron2.data import (
    MetadataCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.engine.hooks import BestCheckpointer
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.utils.events import (
    EventStorage,
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
    get_event_storage,
)
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

# from adet.modeling import swin, vitae_v2  # Commented due to timm incompatibility


class EpochLoggingHook(hooks.HookBase):
    def __init__(self, iterations_per_epoch):
        self.iterations_per_epoch = iterations_per_epoch

    def before_step(self):
        # Calculer l'epoch (ex: iteration 1000 / 500 par epoch = epoch 2)
        epoch = self.trainer.iter // self.iterations_per_epoch
        # Stocker la valeur pour qu'elle soit récupérée par les Writers (JSON, Tensorboard)
        self.trainer.storage.put_scalar("epoch", epoch, smoothing_hint=False)


class EarlyStoppingHook(hooks.HookBase):
    def __init__(self, metric_name, patience=15, mode="max"):
        self.metric_name = metric_name
        self.patience = patience
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.bad_epochs = 0

    def after_step(self):
        # On ne vérifie que lors des phases d'évaluation (quand la métrique est mise à jour)
        # Typiquement après chaque EVAL_PERIOD
        if (self.trainer.iter + 1) % self.trainer.cfg.TEST.EVAL_PERIOD == 0:
            latest_metrics = self.trainer.storage.latest()
            if self.metric_name not in latest_metrics:
                return

            current_val = latest_metrics[self.metric_name][0]

            improved = (
                (current_val > self.best_metric)
                if self.mode == "max"
                else (current_val < self.best_metric)
            )

            if improved:
                self.best_metric = current_val
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1

            if self.bad_epochs >= self.patience:
                print(
                    f"Early stopping déclenché ! Pas d'amélioration depuis {self.patience} évaluations."
                )
                self.trainer.storage.put_scalar("early_stop", 1)
                # Force la fin de la boucle d'entraînement
                raise StopIteration


class ValidationMetricWriter(hooks.HookBase):
    """After each evaluation (triggered by EvalHook) sends the metrics dict into
    EventStorage, prefixed by ``val/`` so they are flushed by PeriodicWriter.
    """

    def __init__(self, metric_names: List[str] | None = None):
        super().__init__()
        self.metric_names = metric_names  # if None → log everything

    def after_eval(self):
        if not hasattr(self.trainer, "_last_eval_results"):
            return
        results: Dict[str, Any] = self.trainer._last_eval_results

        # If the evaluator returned a mapping per‑dataset, flatten it.
        if all(isinstance(v, dict) for v in results.values()):
            flat: Dict[str, float] = {}
            for ds_name, sub in results.items():
                for k, v in sub.items():
                    flat[f"{ds_name}/{k}"] = v
            results = flat

        storage = get_event_storage()
        for k, v in results.items():
            if self.metric_names is None or k in self.metric_names:
                storage.put_scalar(f"val/{k}", float(v), smoothing_hint=False)


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """

    def build_hooks(self):
        """
        Ajoute un PeriodicWriter pour logger les losses + remplace
        le checkpointer Detectron2 par AdetCheckpointer.
        """
        hooks_list = super().build_hooks()

        # --- Calcul des itérations par epoch ---
        # On récupère la taille du dataset via le loader
        dataset_names = self.cfg.DATASETS.TRAIN
        dataset_dicts = DatasetCatalog.get(dataset_names[0])
        dataset_len = len(dataset_dicts)
        batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        iters_per_epoch = max(1, dataset_len // batch_size)

        # On force l'EVAL_PERIOD de la config à être égal à une epoch
        # Cela impactera automatiquement les Hooks qui se basent sur self.cfg
        self.cfg.defrost()
        self.cfg.TEST.EVAL_PERIOD = iters_per_epoch
        self.cfg.freeze()

        # 1. Hook pour l'affichage de l'epoch
        hooks_list.append(EpochLoggingHook(iters_per_epoch))

        # 2. Hook pour les métriques de validation
        hooks_list.append(ValidationMetricWriter())
        # ------------------------------------------------------------
        # 3) Writer pour pertes, lr, time, etc.
        #    - Console  : toutes les 20 iters
        #    - metrics.json : append mode
        #    - TensorBoard : events.*
        # ------------------------------------------------------------
        hooks_list.append(
            hooks.PeriodicWriter(
                writers=[
                    CommonMetricPrinter(self.max_iter),
                    JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                    TensorboardXWriter(self.cfg.OUTPUT_DIR),
                ],
                period=self.cfg.TEST.EVAL_PERIOD,
            )
        )

        # 4. Early Stopping (à placer APRES l'évaluation)
        hooks_list.append(
            EarlyStoppingHook(
                metric_name="bbox/AP",  # det/hmean
                patience=30,  # Nombre de fois où on tolère pas d'amélioration (basé sur EVAL_PERIOD)
                mode="max",
            )
        )

        # ------------------------------------------------------------
        # 2) Remplacement du PeriodicCheckpointer par AdetCheckpointer
        # ------------------------------------------------------------
        for i, h in enumerate(hooks_list):
            if isinstance(h, hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                hooks_list[i] = hooks.PeriodicCheckpointer(
                    self.checkpointer,
                    self.cfg.SOLVER.CHECKPOINT_PERIOD,
                )

        # 3. Ajout d’un BestCheckpointer AP/Hmean
        #    -> attends que l'évaluateur stocke une métrique nommée metric_name
        # Choisis la clé retournée par ton TextEvaluator (ex. "text/hmean")
        metric_name = "bbox/AP"  #  det/precision", "det/recall", "e2e/hmean", "e2e/precision", "e2e/recall"
        hooks_list.append(
            BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD,
                self.checkpointer,
                metric_name,
                mode="max",
                file_prefix="model_best",
            )
        )

        return hooks_list

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        # param = sum(p.numel() for p in self.model.parameters())
        # logger.info(f"Model Params: {param}")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except StopIteration:  # Capturer l'arrêt ici
                logger.info("Training stopped early.")
            finally:
                self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """

        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = DatasetMapperWithBasis(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_optimizer(cls, cfg, model):
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            if match_name_keywords(key, cfg.SOLVER.LR_BACKBONE_NAMES):
                lr = cfg.SOLVER.LR_BACKBONE
            elif match_name_keywords(key, cfg.SOLVER.LR_LINEAR_PROJ_NAMES):
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.LR_LINEAR_PROJ_MULT

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups]
                    )
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    # cfg.defrost()
    # cfg.MODEL.DEVICE = "cpu"
    # cfg.freeze()

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    cfg = setup(args)
    # cfg.defrost()
    # cfg.MODEL.DEVICE = "cpu"
    # cfg.freeze()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)  # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
