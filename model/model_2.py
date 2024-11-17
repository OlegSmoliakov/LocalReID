import logging
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from model.comparator import Comparator
from model.SFSORT_mod import SFSORT, Track

# find only person
CLASSES = [0]
IMGSZ = (384, 640)  # input size of yolo model

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


@dataclass(slots=True)
class Person:
    track: Track
    img: np.ndarray
    last_frame: int = None


class ObjectTracking:
    def __init__(
        self,
        input_source: int | str,
        path_to_model="model/yolov8n.pt",
        output_video=None,
        show_output=True,
    ):
        self.desired_fps = 30

        self.input_source = input_source
        self.output_video = output_video
        self.show_output = show_output

        self.tracker = None
        self.model = None
        self.active_persons: dict[int, Person] = {}
        self.new_persons: dict[int, Person] = {}
        self.model_name = path_to_model.split("/")[-1]

        self.load_model(path_to_model)
        self.comparator = Comparator()
        self.init_input()
        self.init_sfsort()

        if self.output_video:
            self.init_output()

    def init_input(self):
        if isinstance(self.input_source, int):
            self.cap = cv2.VideoCapture(self.input_source)
            assert self.cap.isOpened(), "Cannot open camera"
        else:
            self.cap = cv2.VideoCapture(self.input_source)
            assert self.cap.isOpened(), "Cannot open video file"

        frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(
            f"Input source:\nResolution: {self.frame_width}x{self.frame_height}, FPS: {frame_rate}"
        )

        time.sleep(0.2)  # give some time to open the camera on mac

        self.fps = 0
        self.frame_counter = 0
        self.start_time = time.time()

    def init_output(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(
            self.output_video + ".mp4",
            fourcc,
            self.desired_fps,
            (self.frame_width, self.frame_height),
        )

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return self.release_resources()

        results = self.model.predict(
            frame,
            conf=0.1,
            imgsz=IMGSZ,
            half=False,
            max_det=99,
            iou=0.45,
            classes=CLASSES,
            verbose=False,
        )

        tracks = self.sort_sfsort(results)
        # if len(tracks) > 0 and (max(tracks[:, 1]) + 1) > len(self.active_persons):
        self.reid(frame, tracks)

        marked_frame = self.draw_tracks(frame.copy(), tracks)
        marked_frame = self.draw_fps(marked_frame)

        if self.output_video:
            self.out.write(marked_frame)

        if self.show_output:
            cv2.imshow(f"{self.model_name} Detection", marked_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return self.release_resources()

        persons_to_send = {}
        for track_id, person in self.new_persons.items():
            # wait for N frames before compare new persons
            N = 15
            if self.tracker.frame_no - person.last_frame > N:
                persons_to_send[track_id] = self.new_persons[track_id]

        return persons_to_send

    def release_resources(self):
        if self.output_video:
            self.out.release()
        self.cap.release()
        cv2.destroyAllWindows()

    def load_model(self, path_to_model: str):
        model = YOLO(path_to_model, "detect")
        model.fuse()

        if torch.cuda.is_available():
            model = model.half()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using Device: {device}")

        self.model = model

    def init_sfsort(self):
        self.colors = {}

        frame_rate = self.desired_fps
        frame_height, frame_width = IMGSZ

        log.info(f"SFSORT settings:\nResolution: {frame_width}x{frame_height}, FPS: {frame_rate}")

        tracker_arguments = {
            "dynamic_tuning": True,
            "cth": 0.5,
            "high_th": 0.6,
            "high_th_m": 0.1,
            "match_th_first": 0.67,
            "match_th_first_m": 0.05,
            "match_th_second": 0.2,
            "low_th": 0.1,
            "new_track_th": 0.5,
            "new_track_th_m": 0.08,
            "marginal_timeout": 2 * frame_rate,
            "central_timeout": 3 * frame_rate,
            "horizontal_margin": frame_width // 10,
            "vertical_margin": frame_height // 10,
            "frame_width": frame_width,
            "frame_height": frame_height,
        }

        self.tracker = SFSORT(tracker_arguments)

    def sort_sfsort(self, results: list[Results]):
        prediction_results = results[0].boxes.cpu().numpy()
        tracks = self.tracker.update(prediction_results.xyxy, prediction_results.conf)

        return tracks

    def reid(self, frame: np.ndarray, tracks: np.ndarray):
        if len(tracks) == 0:
            return

        bbox_list = tracks[:, 0]
        track_id_list = tracks[:, 1]

        for track_id, bbox in zip(track_id_list, bbox_list):
            x1, y1, x2, y2 = bbox
            cropped_img = frame[int(y1) : int(y2), int(x1) : int(x2)]

            current_track = self.get_current_track(track_id)

            if track_id in self.active_persons:
                self.active_persons[track_id].track = current_track
                self.active_persons[track_id].img = cropped_img
            elif track_id in self.new_persons:
                self.new_persons[track_id].track = current_track
                self.new_persons[track_id].img = cropped_img
            else:
                self.new_persons[track_id] = Person(
                    current_track, cropped_img, current_track.last_frame
                )

        self.save_persons()  # for debug only

    def remove_duplicates(self):
        persons = self.active_persons.copy()
        person_ids = list(self.active_persons.keys())
        log.debug(f"Person IDs: {person_ids}")

        for i in range(len(person_ids)):
            for j in range(i + 1, len(person_ids)):
                person_id_1 = person_ids[i]
                person_id_2 = person_ids[j]
                person_1 = self.active_persons[person_id_1]
                person_2 = self.active_persons[person_id_2]

                if self.comparator.compare_by_similarity(person_1.img, person_2.img, 0.7):
                    persons.pop(person_id_2)
        self.active_persons = persons

    def get_current_track(self, track_id):
        for track in self.tracker.active_tracks:
            if track.track_id == track_id:
                return track
        return None

    def check_among_detected(self, second_cam_persons: dict[int, Person]):
        changes = []

        for new_person_id, new_person in second_cam_persons.items():
            cv2.imwrite(f"cache/second_cam_person_{new_person_id}.png", new_person.img)

            for person_id, person in self.active_persons.items():
                if self.comparator.compare_by_similarity(person.img, new_person.img, 0.7):
                    changes.append({"old_id": new_person_id, "new_id": person_id})
                    break
            else:
                # real new person on second camera
                self.tracker.id_counter += 1
                changes.append({"old_id": new_person_id, "new_id": new_person_id})

        return {"id_counter": self.tracker.id_counter, "changes": changes}

    def add_new_persons(self, response: dict[str]):
        id_counter = response["id_counter"]
        changes: list[dict[str, int]] = response["changes"]

        for change in changes:
            person = self.new_persons.pop(change["old_id"])
            person.track.track_id = change["new_id"]
            self.tracker.id_counter = id_counter
            self.active_persons[change["new_id"]] = person

        self.save_persons()  # for debug only

    def save_persons(self):
        for track_id, person in self.active_persons.items():
            cv2.imwrite(f"cache/active_person_{track_id}.png", person.img)
        for track_id, person in self.new_persons.items():
            cv2.imwrite(f"cache/new_person_{track_id}.png", person.img)

    def draw_tracks(self, frame: np.ndarray, tracks: np.ndarray):
        if len(tracks) == 0:
            return frame

        # Extract tracking data from the tracker
        bbox_list = tracks[:, 0]
        track_id_list = tracks[:, 1]

        # Visualize tracks
        for idx, (track_id, bbox) in enumerate(zip(track_id_list, bbox_list)):
            # Define a new color for newly detected tracks
            if track_id not in self.colors:
                self.colors[track_id] = (
                    random.randrange(255),
                    random.randrange(255),
                    random.randrange(255),
                )

            color = self.colors[track_id]

            # Extract the bounding box coordinates
            x0, y0, x1, y1 = map(int, bbox)

            # Calculate the center of the bounding box
            center_x = (x0 + x1) // 2
            center_y = (y0 + y1) // 2

            # Draw the bounding boxes on the frame
            annotated_frame = cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            # Put the track label on the frame alongside the bounding box
            cv2.putText(
                annotated_frame,
                str(track_id),
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        return frame

    def draw_fps(self, frame):
        self.frame_counter += 1
        elapsed_time = time.time() - self.start_time

        if elapsed_time >= 1.0:
            self.fps = self.frame_counter / elapsed_time
            self.frame_counter = 0
            self.start_time = time.time()

        cv2.putText(
            frame, f"FPS: {self.fps:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2
        )

        return frame


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(levelname)s: %(message)s")

    capture_device = 1
    input_video = "draft/campus4-c0.avi"
    # out_path = "output"
    detector = ObjectTracking(input_video, "model/yolov8n.pt")
    detector()
