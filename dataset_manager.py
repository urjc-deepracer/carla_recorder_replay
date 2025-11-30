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

import os
import time
import csv
import cv2

import numpy as np
import pandas as pd

class DatasetSaver:

    def __init__ (self, path):

        self.path = path
        current_time   = str(int(time.time() * 1000))
        
        self.dataset_id = current_time + "_dataset"  
        self.dataset_path = self.path + self.dataset_id

        self.rgb_foldername = "rgb"
        self.mask_foldername = "mask"

        self.rgb_path = os.path.join(self.dataset_path, self.rgb_foldername)
        self.mask_path = os.path.join(self.dataset_path, self.mask_foldername)
        self.csv_filename = os.path.join(self.dataset_path, "dataset.csv")

        self.counter = 0

        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)

        print (f"DatasetSaver loaded for {self.dataset_path}")

        if not os.path.exists(self.csv_filename):
            os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
            with open(self.csv_filename, "w", newline="") as f:
                csv.writer(f).writerow(["rgb_path","mask_path","timestamp",
                                        "throttle","steer","brake","speed"])

    def save_sample (self, timestamp, bgr, mask_rgb, throttle, steer, brake, speed):
        
        rgb_filename  = f"rgb_{self.counter:08d}.png"
        mask_filename = f"mask_{self.counter:08d}.png"
        
        self.counter = self.counter + 1

        cv2.imwrite(os.path.join(self.rgb_path,  rgb_filename),  bgr)
        cv2.imwrite(os.path.join(self.mask_path, mask_filename),
                    cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

        with open(self.csv_filename, "a", newline="") as f:
            csv.writer(f).writerow([
                f"/{self.rgb_foldername}/{rgb_filename}",
                f"/{self.mask_foldername}/{mask_filename}",
                timestamp, throttle, steer, brake, speed
            ])        
     
    def adjust_speed (self, csv_data_filename):

        try:
            self.load_speed_from_csv(
                self.csv_filename,
                csv_data_filename,
                dst_speed_col="speed",
                src_speed_col="speed_m_s"
            )
        except Exception as e:
            print(f"[ERROR] speed align (merge_asof): {e}")

    def load_speed_from_csv(self,
                            dataset_csv: str,
                            speed_csv: str,
                            dst_speed_col: str = "speed",
                            src_speed_col: str = "speed_m_s",
                            src_time_col: str = "sim_time"):

        # 1) Check files
        if not os.path.isfile(dataset_csv):
            print(f"[ERROR] Dataset {dataset_csv} does not exist")
            return
        if not os.path.isfile(speed_csv):
            print(f"[ERROR] Speed CSV does not exist: {speed_csv}")
            return

        # 2) Loading
        df_dst = pd.read_csv(dataset_csv)
        df_src = pd.read_csv(speed_csv)

        # 3) Check necessary columns
        for col in ["timestamp", dst_speed_col]:
            if col not in df_dst.columns:
                print(f"[ERROR] Dataset does not have col '{col}'.")
                return
        for col in [src_time_col, src_speed_col]:
            if col not in df_src.columns:
                print(f"[ERROR] Speed CSV not containing col '{col}'.")
                return
        if df_dst.empty or df_src.empty:
            print("[WARN] Empty Dataset or speed CSV")
            return

        # 4) Number conversion
        df_dst = df_dst.copy()
        df_src = df_src.copy()

        df_dst["timestamp"]      = pd.to_numeric(df_dst["timestamp"], errors="coerce")
        df_src[src_time_col]     = pd.to_numeric(df_src[src_time_col], errors="coerce")
        df_src[src_speed_col]    = pd.to_numeric(df_src[src_speed_col], errors="coerce")

        df_dst = df_dst.dropna(subset=["timestamp"])
        df_src = df_src.dropna(subset=[src_time_col, src_speed_col])

        if df_dst.empty or df_src.empty:
            print("[WARN] NaN data filtered and no data left.")
            return

        # 5) Order by time
        df_dst_sorted = df_dst.sort_values("timestamp").reset_index(drop=False)
        df_src_sorted = df_src.sort_values(src_time_col).reset_index(drop=True)

        # 6) Align using merge_asof
        merged = pd.merge_asof(
            df_dst_sorted,
            df_src_sorted[[src_time_col, src_speed_col]],
            left_on="timestamp",
            right_on=src_time_col,
            direction="nearest"
        )

        # 7) Set aligned speed in the original DataFrame
        df_dst.loc[merged["index"], dst_speed_col] = merged[src_speed_col].values

        # 8) Keep updated data
        df_dst.to_csv(dataset_csv, index=False)

        # 9) Info
        time_diff = np.abs(merged["timestamp"] - merged[src_time_col])
        print(f"[INFO] Speed alignment")
        print(f"  - Nº rows dataset:   {len(df_dst)}")
        print(f"  - Nº rows speed_csv: {len(df_src)}")
        print(f"  - Avg time diff:     {time_diff.mean():.4f} s")
        print(f"  - Max time diff:     {time_diff.max():.4f} s")