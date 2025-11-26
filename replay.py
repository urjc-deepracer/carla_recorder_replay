#!/usr/bin/env python3
#
#
#  Copyright (C) URJC DeepRacer
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see http://www.gnu.org/licenses/. 
#
#  Author : Roberto Calvo Palomino <roberto.calvo at urjc dot es
#           Sergio Robledo <s.robledo.2021 at alumnos dot urjc dot es>


import carla
import pygame
import numpy as np
import sys
import time
import os
import argparse
import csv
import cv2
from pathlib import Path

import queue
from queue import Queue

from dataset_manager import DatasetSaver

RATE_CONTROL_LOOP = 30

CUSTOM_MAPPING = {
    0: [11],
    1: [14],
    2: [34],
    3: [0, 30],
    4: [33],
    5: [31],
    6: [25, 29]
}
MAX_CUSTOM_LABELS = 7

# SPEED_CSV = "./speedTown4.csv"

# # Dataset paths
# current_time   = str(int(time.time() * 1000))
# DATASET_ID = current_time + "_dataset"  
# SAVE_DIR   = DATASET_ID
# RGB_DIR    = os.path.join(SAVE_DIR, "rgb")
# MASK_DIR   = os.path.join(SAVE_DIR, "masks")
# CSV_PATH   = os.path.join(SAVE_DIR, "dataset.csv")

# os.makedirs(RGB_DIR, exist_ok=True)
# os.makedirs(MASK_DIR, exist_ok=True)
# if not os.path.exists(CSV_PATH):
#     os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
#     with open(CSV_PATH, "w", newline="") as f:
#         csv.writer(f).writerow(["rgb_path","mask_path","timestamp","throttle","steer","brake","speed"])



# def save_data(timestamp, bgr, mask_rgb, throttle, steer, brake, speed):
#     rgb_name  = f"{timestamp}_rgb_{DATASET_ID}.png"
#     mask_name = f"{timestamp}_mask_{DATASET_ID}.png"
#     cv2.imwrite(os.path.join(RGB_DIR,  rgb_name),  bgr)
#     cv2.imwrite(os.path.join(MASK_DIR, mask_name), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
#     with open(CSV_PATH, "a", newline="") as f:
#         csv.writer(f).writerow([f"/rgb/{rgb_name}", f"/masks/{mask_name}", timestamp,
#                                 throttle, steer, brake, speed])

def convert_to_equidistant_gray(image_array, num_classes):
    # handle edge case for single class or empty
    if num_classes < 2: 
        return np.zeros_like(image_array, dtype=np.uint8)
    
    # scale ids to cover full 0-255 range
    factor = 255.0 / (num_classes - 1)
    return (image_array * factor).astype(np.uint8)

def apply_custom_mapping(raw_semseg):

    output = np.zeros_like(raw_semseg, dtype=np.uint8)
    
    # map carla ids to custom labels using boolean masks
    for new_label, old_labels_list in CUSTOM_MAPPING.items():
        mask = np.isin(raw_semseg, old_labels_list)
        output[mask] = new_label
        
    return output

def get_log_duration(client, log_file):
    import re           
    info = client.show_recorder_file_info(log_file, False)  
    # Look at for Duration: 12.34 s"
    match = re.search(r"Duration:\s+([0-9.]+)", info)
    if not match:
        raise RuntimeError("Duration time cannot be read!")
    return float(match.group(1))

def replay_loop(args, view="car"):

    pygame.init()
    pygame.display.set_caption(f"CARLA Replay - Replay view {view} ")
    display_width, display_height = 800, 600
    screen = pygame.display.set_mode((display_width, display_height))

    client = carla.Client('localhost', args.port)  
    client.set_timeout(10.0)

    world = client.get_world()
    
    path = Path(args.log_path)
    logs = list(path.glob("*.log"))    
    if (len(logs) == 0):
        print(f"Error, no log file found in {args.log_path}")
        exit(-1)    

    p = logs[0]
    if not p.is_absolute():
        p = Path.cwd() / p
    log_filename = str(p)
    print(f'Using log file {log_filename}')

    duration = get_log_duration(client, log_filename)
    duration = duration + world.get_snapshot().timestamp.elapsed_seconds
    print(f"Replaying: {log_filename}, duration: {duration:.2f} s")

    dataset = None
    if args.generate_dataset_path is not None:        
        dataset = DatasetSaver(args.generate_dataset_path)
    
    client.replay_file(log_filename, 0, 0, 0)

    blueprint_library = world.get_blueprint_library()
    
    actors = None

    if (view == "car"):
        actors = world.get_actors().filter("vehicle.tesla.model3")
    elif (view == "bike"):
        actors = world.get_actors().filter("vehicle.diamondback.century")

    if not actors:
        raise RuntimeError("No vehicles found in the replay")

    vehicle = actors[0]  # first vehicle is ego
    print(f"Using ego con id={vehicle.id}, type={vehicle.type_id}")
    
    # spawn rgb camera
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(display_width))
    camera_bp.set_attribute("image_size_y", str(display_height))
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # spawn semantic segmentation camera
    sem_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    sem_bp.set_attribute("image_size_x", str(display_width))
    sem_bp.set_attribute("image_size_y", str(display_height))
    sem_bp.set_attribute("fov", "90")

    sem_camera = world.spawn_actor(sem_bp, camera_transform, attach_to=vehicle)

    frame_q = Queue(maxsize=1)   # save (rgb, bgr, (w, h))
    sem_queue = Queue(maxsize=1)

    def _safe_put(q: Queue, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)

    def process_image(image):
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = np.reshape(bgra, (image.height, image.width, 4))
        bgr  = bgra[:, :, :3].copy()
        rgb  = bgr[:, :, ::-1]
        _safe_put(frame_q, (rgb, bgr, (image.width, image.height)))

    camera.listen(lambda img: process_image(img))
    sem_camera.listen(lambda img: _safe_put(sem_queue, img))

    clock = pygame.time.Clock()

    # Start at a relative time 0.0 to syncronize with speed csv
    t0_sim = 0.0

    try:
        while True:            
            
            clock.tick(RATE_CONTROL_LOOP)

            snapshot = world.get_snapshot()  
            sim_time = snapshot.timestamp.elapsed_seconds                      
            
            if sim_time >= duration:
                print("Replay finished")
                break

            try:
                rgb, bgr, (w, h) = frame_q.get_nowait()
            except queue.Empty:
             
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        raise KeyboardInterrupt
                continue

            raw_semseg_img = None
            try:
                sem_data = sem_queue.get(timeout=0.2)
                sem_arr = np.frombuffer(sem_data.raw_data, dtype=np.uint8)
                sem_arr = np.reshape(sem_arr, (sem_data.height, sem_data.width, 4))
                raw_semseg_img = sem_arr[:, :, 2] 
            except queue.Empty:
                pass

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Get relative time for speedcsv/replay sync
            if t0_sim == 0.0:
                t0_sim = sim_time

            rel_time = sim_time - t0_sim

            
            #if image_surface is not None:
            surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            screen.blit(surface, (0, 0))

            pygame.display.flip()

            if dataset is not None:
                # parse dataset type flags
                types = args.dataset_types
                save_all = "all" in types
                do_mask = save_all or "mask" in types
                do_rgb = save_all or "rgb" in types
                do_semseg = save_all or "semseg" in types
                do_custom = save_all or "custom_semseg" in types

                mask_rgb = None
                out_semseg = None
                out_custom = None

                # compute semseg or custom_semseg if enabled
                if raw_semseg_img is not None:
                    if do_semseg:
                        out_semseg = convert_to_equidistant_gray(raw_semseg_img, num_classes=29)

                    if do_custom:
                        mapped = apply_custom_mapping(raw_semseg_img)
                        out_custom = convert_to_equidistant_gray(mapped, num_classes=MAX_CUSTOM_LABELS)

                # compute masks if enabled
                if do_mask:
                    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                    mask_y = cv2.inRange(hsv, np.array([18, 50, 150]), np.array([40, 255, 255]))
                    mask_w = cv2.inRange(hsv, np.array([0, 0, 200]),  np.array([180, 30, 255]))
                    mask_c = np.zeros(mask_w.shape, np.uint8)
                    mask_c[mask_w > 0] = 1
                    mask_c[mask_y > 0] = 2
                    mask_rgb = np.zeros_like(rgb)
                    mask_rgb[mask_c == 1] = [255, 255, 255]
                    mask_rgb[mask_c == 2] = [255, 255, 0]

                # extract vehicle control data
                ctrl = vehicle.get_control()
                throttle = float(ctrl.throttle)
                steer = max(-1.0, min(1.0, float(ctrl.steer)))
                brake = float(ctrl.brake)
                speed = 0.0

                # select outputs based on flags
                image_out = bgr if do_rgb else None
                mask_out = mask_rgb if do_mask else None

                # save sample to dataset
                dataset.save_sample(rel_time, image_out, mask_out, out_semseg, out_custom, throttle, steer, brake, speed)


    except KeyboardInterrupt:
        print("Exit...")
    except Exception as e:
        print(e)
    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()

        if sem_camera is not None:
            sem_camera.stop()
            sem_camera.destroy()

        vehicle.destroy()
        
        if dataset is not None:

            # Takes both dataset and speed CSV files and do the matching
            path = Path(args.log_path)
            logs = list(path.glob("*.csv"))    
            if (len(logs) == 0):
                print(f"Error, no data csv file found in {args.log_path}")
                exit(-1) 

            csv_data_filename = str(logs[0])

            dataset.adjust_speed(csv_data_filename)


        pygame.quit()
        sys.exit()


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="recorder")

    parser.add_argument("--log_path", type=str, required=True,
                        help="Directory where log files will be loaded")
    
    parser.add_argument("--port", "--carla-port", type=int, default=3010,
                        help="Port used to connect to the CARLA simulator")
    
    parser.add_argument("--tport","--carla-traffic-port", type=int, default=3020,
                        help="Port used by the CARLA traffic manager")
    
    parser.add_argument("--generate_dataset_path",  type=str, default=None,
                        help="Enable dataset generation and set the path to save it")
    
    parser.add_argument(
                        "--dataset_types", "--carla-dataset-types",
                        nargs="+",
                        choices=["rgb", "mask", "semseg", "custom_semseg", "all"],
                        default=["all"],
                        metavar="TYPE",
                        help=(
                            "Types of frames to export. Options: rgb, mask, segmented, all. "
                            "Example: --dataset_types rgb mask"
                        )
                    )
    args = parser.parse_args()

    # Use "bike" or "car" to choose from where point of view you want to replay de simulation
    replay_loop(args, "car")
