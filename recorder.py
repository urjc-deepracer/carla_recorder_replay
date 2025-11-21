#!/usr/bin/env python3
#
#
#  Copyright (C) Roberto Calvo Palomino
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

 
 

import os
import sys
import csv
import time
import argparse
import numpy as np

import carla
import pygame
import random


RATE_CONTROL_LOOP = 30
VEHICLE_MODEL = "model3"
CSV_FILENAME = "data.csv"

last_time = None
count = 0

def game_loop(args):

    pygame.init()
    pygame.display.set_caption("CARLA recorder")
    display_width, display_height = 800, 600
    screen = pygame.display.set_mode((display_width, display_height))

    client = carla.Client('localhost', args.port)
    client.set_timeout(10.0)

    world = client.load_world(args.town)

    log_path = args.log_path + "/" + str(int(time.time())) + "_" + args.town + "/"
    os.makedirs(log_path, exist_ok=True)

    # Speed CSV settings    
    csv_file_path = log_path + CSV_FILENAME
    csv_fh = open(csv_file_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["sim_time", "speed_m_s"])

    blueprint_library = world.get_blueprint_library()

    # Select vehicle
    vehicle_bp = blueprint_library.filter(VEHICLE_MODEL)[0]
    print(vehicle_bp)
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)    
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Traffic manager
    tm = client.get_trafficmanager(args.tport)
    tm_port = tm.get_port()        

    if (args.extra_actor):
        # Spawn the bicycle in front of the car
        moto_bp = blueprint_library.filter('vehicle.diamondback.century')[0] 
        moto_transform = carla.Transform(
            spawn_point.location + spawn_point.get_forward_vector() * 10.0,  # 10 meters in front of
            spawn_point.rotation
        )
        motorcycle = world.try_spawn_actor(moto_bp, moto_transform)
        motorcycle.set_autopilot(True, tm_port)

    # Enable autopilot
    vehicle.set_autopilot(True, tm_port)

    # Start the log and recording
    # Important! Make sure you start the recording after spawn all your actors
    
    log_filename = log_path + args.town + ".log"
    print(log_filename)
    client.start_recorder(log_filename, True)

    # Camera RGB
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", str(display_width))
    camera_bp.set_attribute("image_size_y", str(display_height))
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(carla.Location(x=0, z=1.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    image_surface = None

    def process_image(image):
        nonlocal image_surface
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    camera.listen(lambda img: process_image(img))

    clock = pygame.time.Clock()


    # Measure the execution rate (Hz) in the server
    def on_tick(snapshot):
        fps_server = 1.0 / snapshot.timestamp.delta_seconds
        print(f"Frame {snapshot.frame} | Server ~{fps_server:.1f} Hz  ", end="\r")

    world.on_tick(on_tick)

    # Start time at 0 for the speed csv
    snapshot = world.get_snapshot()               
    t0 = snapshot.timestamp.elapsed_seconds

    try:
        while True:
            
            rel_time = world.get_snapshot().timestamp.elapsed_seconds - t0
            # Get vehicle speed
            raw_vel = vehicle.get_velocity()

            # rate of this control loop
            clock.tick(RATE_CONTROL_LOOP)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            if image_surface is not None:
                screen.blit(image_surface, (0, 0))

            pygame.display.flip()

            
            speed = float(np.linalg.norm([raw_vel.x, raw_vel.y, raw_vel.z]))
            csv_writer.writerow([f"{rel_time:.6f}", f"{speed:.6f}"])

    except KeyboardInterrupt:
        print("Exit...")

    finally:
        
        client.stop_recorder()
        
        if camera is not None:
            camera.stop()
            camera.destroy()
            
        if vehicle is not None:
            vehicle.destroy()

        if args.extra_actor and motorcycle is not None:
            motorcycle.destroy()

        csv_fh.flush()
        csv_fh.close()

        pygame.quit()
        sys.exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="recorder")

    parser.add_argument("--log_path", type=str, default=os.getcwd() + "/logs/",
                        help="Folder to save logs")
    
    parser.add_argument("--town", "--carla-town", type=str, default="Town04",
                        help="Map where run recorder")
    
    parser.add_argument("--port", "--carla-port", type=int, default=3010,
                        help="Default port for CARLA")
    
    parser.add_argument("--tport","--carla-traffic-port", type=int, default=3020,
                        help="Default port for CARLA")
    
    parser.add_argument("--extra_actor", "--carla-extra-actor", action="store_true",
                        help="Enable an extra actor in fron of the ego vehicle")

    args = parser.parse_args()

    try:
        game_loop(args)
    except SystemExit:
        pass
