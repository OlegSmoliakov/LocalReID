import logging
import os
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from src.base import MODELS_DIR
from src.comparator import Comparator
from src.SFSORT import SFSORT, Track

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
    color: tuple[int, int, int] = None
    prev_img: np.ndarray = None
    lost: bool = False


class ObjectTracking:
    def __init__(
        self,
        input_source: int | str,
        path_to_model=os.path.join(MODELS_DIR, "yolov8n.pt"),
        output_video=None,
        show_output=True,
    ):
        self.desired_fps = 25

        self.input_source = input_source
        self.output_video = output_video
        self.show_output = show_output

        self.tracker = None
        self.model = None
        self.persons: dict[int, Person] = {}
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
        log.debug(f"Total number of frames: {self.cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

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
        persons_to_send = self.reid(frame)

        marked_frame = self.draw_tracks(frame.copy(), tracks)
        marked_frame = self.draw_fps(marked_frame)

        if self.output_video:
            self.out.write(marked_frame)

        if self.show_output:
            cv2.imshow(f"{self.model_name} Detection", marked_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return self.release_resources()

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
            "central_timeout": 8 * frame_rate,
            "horizontal_margin": frame_width // 10,
            "vertical_margin": frame_height // 10,
            "frame_width": frame_width,
            "frame_height": frame_height,
        }

        self.tracker = SFSORT(tracker_arguments)

    def sort_sfsort(self, results: list[Results]):
        prediction_results = results[0].boxes.cpu().numpy()
        self.boxes = prediction_results.xyxy.copy()
        tracks = self.tracker.update(prediction_results.xyxy, prediction_results.conf)

        return tracks

    def reid(self, frame: np.ndarray):
        if self.tracker.active_tracks:
            self.process_active_tracks(frame)

        if self.tracker.lost_tracks:
            self.process_lost_tracks()

        self.save_persons()  # for debug only

        persons_to_send = {}
        for track_id, person in self.new_persons.items():
            # wait for N frames before sending new persons
            N = 5
            if self.tracker.frame_no - person.last_frame == N:
                persons_to_send[track_id] = self.new_persons[track_id]

        return persons_to_send

    def process_active_tracks(self, frame):
        tracks = np.asarray(
            [[x.bbox, x.track_id] for x in self.tracker.active_tracks], dtype=object
        )
        bbox_list = tracks[:, 0]
        track_id_list = tracks[:, 1]
        h_gap = int(self.frame_width * 0.03)
        v_gap = int(self.frame_height * 0.02)

        iou_matrix = SFSORT.calculate_cost(self.tracker.active_tracks, self.boxes, True)

        for track_id, bbox, iou in zip(track_id_list, bbox_list, iou_matrix):
            # delete the diagonal elements from the IoU matrix
            iou = np.delete(iou, np.where(iou == 1.0)[0])
            x1, y1, x2, y2 = map(int, bbox)

            # Check if the bounding box does not touch the frame boundaries
            if (
                x1 > h_gap
                and y1 > v_gap
                and x2 < self.frame_width - h_gap
                and y2 < self.frame_height - v_gap
                and iou.max() < 0.9
            ):
                cropped_img = frame[y1:y2, x1:x2]

                # if (
                #     self.persons[track_id].lost
                #     and self.persons[track_id].track.track_id != track_id
                # ):
                #     self.check_among_detected({track_id: self.persons[track_id]}, 0.6)

                if track_id in self.persons:
                    self.persons[track_id].img = cropped_img
                elif track_id in self.new_persons:
                    self.new_persons[track_id].img = cropped_img
                else:
                    current_track = self.get_current_track(track_id)
                    self.generate_track_color(track_id)
                    self.new_persons[track_id] = Person(
                        current_track, cropped_img, current_track.last_frame, self.colors[track_id]
                    )

    def process_lost_tracks(self):
        # for track in self.tracker.lost_tracks:
        #     try:
        #         self.persons[track.track_id].lost = True
        #     except KeyError:
        #         self.new_persons[track.track_id].lost = True
        pass

    def get_current_track(self, track_id):
        for track in self.tracker.active_tracks:
            if track.track_id == track_id:
                return track
        return None

    def check_among_local(self, threshold=0.8):
        persons = self.new_persons.copy()
        response = self.check_among_detected(persons, threshold)
        try:
            changes: dict[int, dict[int, float]] = response["changes"]
            for probe_id, sim_map in changes.items():
                if not sim_map:
                    return False
                gallery_id = max(sim_map, key=sim_map.get)
                person = self.new_persons.pop(probe_id)
                person.track.track_id = gallery_id
                person.color = self.colors[gallery_id]
                self.persons[gallery_id] = person
                return True
        except KeyError:
            return False

    def check_among_detected(self, persons_from_second_cam: dict[int, Person], threshold=0.63):
        changes = {}
        for probe_id, probe in persons_from_second_cam.items():
            cv2.imwrite(f"cache/second_cam_person_{probe_id}.png", probe.img)  # for debug only

            self.colors[probe_id] = probe.color
            gallery = {person_id: person.img for person_id, person in self.persons.items()}
            sim_map = self.comparator.get_similarity_map(probe.img, gallery, threshold)
            changes[probe_id] = sim_map

        return {"id_counter": self.tracker.id_counter, "changes": changes}

    def add_new_persons(self, response: dict[str]):
        id_counter = response["id_counter"]
        try:
            changes: dict[int, dict[int, float]] = response["changes"]
        except KeyError:
            if (step := self.tracker.id_counter - id_counter + 1) > 0:
                for track in self.tracker.active_tracks:
                    if track.track_id >= id_counter - 1:
                        original_track = track.track_id
                        track.track_id += step
                        if original_track in self.new_persons:
                            self.new_persons[track.track_id] = self.new_persons.pop(original_track)
                            self.generate_track_color(track.track_id)
                            self.new_persons[track.track_id].color = self.colors[track.track_id]
                self.tracker.id_counter += step
                log.debug(f"id_counter set to: {id_counter}")
            else:
                self.tracker.id_counter = id_counter
                log.debug(f"id_counter set to: {id_counter}")
            return

        response = {}
        track_ids = [track.track_id for track in self.tracker.active_tracks]
        for probe_id, sim_map in changes.items():
            for gallery_id, sim in sim_map.items():
                if gallery_id in track_ids:
                    continue
                if gallery_id in self.persons:
                    # check if new person absolutely not similar to received one
                    local_sim_map = self.comparator.get_similarity_map(
                        self.new_persons[probe_id].img, {0: self.persons[gallery_id].img}, 0.6
                    )
                    if not local_sim_map:
                        continue
                person = self.new_persons.pop(probe_id)
                person.track.track_id = gallery_id
                person.color = self.colors[gallery_id]
                self.persons[gallery_id] = person
                self.tracker.id_counter = id_counter
                log.info(f"Person_id changed: {probe_id} -> {gallery_id}")
                break
            else:
                # add new person
                self.persons[probe_id] = self.new_persons.pop(probe_id)
                response["id_counter"] = self.tracker.id_counter
                log.info(f"New person added with id: {probe_id}")

        self.save_persons()  # for debug only

        return response

    def save_persons(self):
        for track_id, person in self.persons.items():
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
            self.generate_track_color(track_id)
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

    def generate_track_color(self, track_id):
        if track_id not in self.colors:
            self.colors[track_id] = (
                random.randrange(255),
                random.randrange(255),
                random.randrange(255),
            )

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
    detector = ObjectTracking(input_video)
    detector()
