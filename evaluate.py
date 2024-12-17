import os
import pickle
import random
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog


def main():

    output_dir = os.path.join("models", "output_kidney_patches")
    model_weights = os.path.join(output_dir, "model_final.pth")

    with open("test.pkl", "rb") as f:
        test_dicts = pickle.load(f)

    def get_kidney_dicts():
        return test_dicts

    DatasetCatalog.register("kidney_patches_eval", get_kidney_dicts)
    MetadataCatalog.get("kidney_patches_eval").set(
        thing_classes=["lymphocytes", "monocytes"]
    )
    kidney_metadata = MetadataCatalog.get("kidney_patches_eval")

    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(output_dir, "config.yaml"))

    cfg.DATASETS.TEST = ("kidney_patches_eval",)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)

    for d in random.sample(test_dicts, 10):
        print(d["file_name"])
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=kidney_metadata, scale=0.8)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
