import os
import pickle
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog


def main():

    output_dir = os.path.join("models", "output_kidney_patches")
    os.makedirs(output_dir, exist_ok=True)

    with open("train.pkl", "rb") as f:
        dataset_dicts = pickle.load(f)
    with open("test.pkl", "rb") as f:
        test_dataset_dicts = pickle.load(f)

    def get_kidney_dicts():
        return dataset_dicts

    def get_test_kidney_dicts():
        return test_dataset_dicts

    DatasetCatalog.register("kidney_patches", get_kidney_dicts)
    DatasetCatalog.register("test_kidney_patches", get_test_kidney_dicts)

    kidney_metadata = MetadataCatalog.get("kidney_patches").set(
        thing_classes=["lymphocytes", "monocytes"]
    )
    test_kidney_metadata = MetadataCatalog.get("test_kidney_patches").set(
        thing_classes=["lymphocytes", "monocytes"]
    )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    cfg.DATASETS.TRAIN = ("kidney_patches",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.03
    cfg.SOLVER.STEPS = (600,750,900)
    cfg.SOLVER.GAMMA = 0.3
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    cfg.OUTPUT_DIR = output_dir

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())


if __name__ == "__main__":
    main()
