from fastapi import FastAPI, UploadFile, HTTPException
import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

app = FastAPI()


def detect_watermark(image):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    outputs = predictor(image)
    classes = outputs["instances"].pred_classes.cpu().numpy()
    watermark_detected = any(class_id == watermark_class_id for class_id in classes)

    return watermark_detected


@app.post("/detect_watermark")
async def detect_watermark_endpoint(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        np_image = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)

        watermark_detected = detect_watermark(np_image)

        if watermark_detected:
            raise HTTPException(status_code=404, detail="Watermark detected")
        else:
            return {"status": "Success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)