from time import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# find only person
TARGET_CLASSES = [0]


class ObjectDetection:
    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("model/yolov8n.pt")
        model.fuse()

        if torch.cuda.is_available():
            model = model.half()

        return model

    def get_results(self, results):
        detections_list = []

        # Extract detections
        for result in results[0]:

            bbox = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()

            merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0]]

            detections_list.append(merged_detection)

        return np.array(detections_list)

    def draw_bounding_boxes_with_id(self, img, bboxes, ids):

        for bbox, id_ in zip(bboxes, ids):

            cv2.rectangle(
                img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2
            )
            cv2.putText(
                img,
                "ID: " + str(id_),
                (int(bbox[0]), int(bbox[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        return img

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps_list = []

        while True:
            start_time = time()

            ret, frame = cap.read()
            assert ret

            results = self.model(frame, classes=TARGET_CLASSES, verbose=True)
            detections_list = self.get_results(results)

            boxes_track = detections_list[:, :-1]
            boxes_ids = detections_list[:, -1].astype(int)

            frame = self.draw_bounding_boxes_with_id(frame, boxes_track, boxes_ids)

            end_time = time()
            frame_time = end_time - start_time
            fps = 1 / frame_time

            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)

            avg_fps = np.mean(fps_list)

            cv2.putText(
                frame,
                f"FPS: {int(avg_fps)}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLOv8 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_device = 1
    detector = ObjectDetection(capture_device)
    detector()
