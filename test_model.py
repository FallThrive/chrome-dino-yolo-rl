from ultralytics import YOLO
from take_screenshots import capture_screenshot
import supervision as sv
import cv2

model = YOLO("weights/yolo26n_dino_260418.pt")

bounding_box_annotator = sv.BoxAnnotator()

while True:
    image = capture_screenshot()
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(
        scene=image,
        detections=detections
    )

    cv2.imshow("Annotated Image", annotated_image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
