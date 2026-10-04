# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  
        self.niou = self.iouv.numel()
        self.lb = []  
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

    def preprocess(self, batch):
        
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        
        val = self.data.get(self.args.split, "")  
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.paper_ground_truth = {"images": [], "annotations": [],
                                   "categories": [{"id": int(index) + 1, "name": name} for index, name in self.names.items()]}
        self.paper_predictions = []

    def get_desc(self):
        
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=not self.args.paper_evaluation,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  
        return predn

    def update_metrics(self, preds, batch):
        
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            if self.args.paper_evaluation:
                height, width = pbatch["ori_shape"]
                self.paper_ground_truth["images"].append({"id": self.seen, "file_name": str(batch["im_file"][si]),
                                                          "width": int(width), "height": int(height)})
                for category, box in zip(cls.cpu().tolist(), bbox.cpu().tolist()):
                    x1, y1, x2, y2 = box
                    self.paper_ground_truth["annotations"].append({"id": len(self.paper_ground_truth["annotations"]) + 1,
                        "image_id": self.seen, "category_id": int(category) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            if self.args.paper_evaluation:
                for x1, y1, x2, y2, score, category in predn.cpu().tolist():
                    self.paper_predictions.append({"image_id": self.seen, "category_id": int(category) + 1,
                                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def finalize_metrics(self, *args, **kwargs):
        
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        
        stats = {k: torch.cat(v, 0).cpu().numpy() if v else np.zeros((0, self.niou), dtype=bool) if k == "tp" else np.zeros(0)
                 for k, v in self.stats.items()}
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        result = self.metrics.results_dict
        if self.args.paper_evaluation:
            from ultralytics.utils.paper_metrics import coco_ap
            ground_truth, predictions = self._paper_annotations()
            requested = str(self.args.checkpoint_class).lower()
            aliases = {"drone", "uav"} if requested in {"drone", "uav"} else {requested}
            selected = [category["id"] for category in ground_truth["categories"] if str(category["name"]).lower() in aliases]
            ap = coco_ap(ground_truth, predictions, selected or None)["ap"]
            if ap is None:
                raise ValueError("Validation annotations do not define checkpoint-selection AP")
            result["metrics/paper_AP50-95(B)"] = ap / 100.0
            result["fitness"] = ap / 100.0
        return result

    def _paper_annotations(self):
        annotation_config = self.data.get("annotations", {})
        source = annotation_config.get(self.args.split) if isinstance(annotation_config, dict) else None
        if not source:
            return self.paper_ground_truth, self.paper_predictions
        path = Path(source)
        if not path.is_absolute():
            path = Path(self.data["path"]) / path
        original = json.loads(path.read_text(encoding="utf-8"))
        def path_parts(value):
            return tuple(part for part in PurePosixPath(str(value).replace("\\", "/")).parts if part not in {"/", "."})

        lookup, suffix_lookup = {}, {}
        for image in original["images"]:
            parts = path_parts(image["file_name"])
            lookup.setdefault(parts, set()).add(image["id"])
            for start in range(len(parts)):
                suffix_lookup.setdefault(parts[start:], set()).add(image["id"])
        image_map = {}
        for image in self.paper_ground_truth["images"]:
            parts = path_parts(image["file_name"])
            keys = [parts]
            try:
                keys.append(path_parts(Path(image["file_name"]).resolve().relative_to(Path(self.data["path"]).resolve())))
            except ValueError:
                pass
            matches = set().union(*(lookup.get(key, set()) for key in keys))
            if not matches:
                matches = next((suffix_lookup[parts[start:]] for start in range(len(parts)) if parts[start:] in suffix_lookup), set())
            if len(matches) != 1:
                raise ValueError("Validation images cannot be matched uniquely to original COCO annotations")
            image_map[image["id"]] = next(iter(matches))
        if len(set(image_map.values())) != len(image_map) or set(image_map.values()) != {image["id"] for image in original["images"]}:
            raise ValueError("Original COCO validation annotations must cover the exact validation split")
        def category_name(value):
            value = str(value).strip().lower()
            return "drone" if value in {"drone", "uav"} else value

        category_names = {}
        for category in original["categories"]:
            category_names.setdefault(category_name(category["name"]), set()).add(category["id"])
        category_map = {}
        for index, name in self.names.items():
            matches = category_names.get(category_name(name), set())
            if len(matches) != 1:
                raise ValueError("Validation class names differ from or ambiguously match original COCO categories")
            category_map[int(index) + 1] = next(iter(matches))
        predictions = [{**prediction, "image_id": image_map[prediction["image_id"]],
                        "category_id": category_map[prediction["category_id"]]} for prediction in self.paper_predictions]
        return original, predictions

    def print_results(self):
        
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  

    def plot_val_samples(self, batch, ni):
        
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  

    def save_one_txt(self, predn, save_conf, shape, file):
        
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  
        box[:, :2] -= box[:, 2:] / 2  
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  
                    from pycocotools.cocoeval import COCOeval  

                    anno = COCO(str(anno_json))  
                    pred = anno.loadRes(str(pred_json))  
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  
                    pred = anno._load_json(str(pred_json))  
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  
                
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats
