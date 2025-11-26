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
        self.semseg_foldername = "semseg"
        self.custom_foldername = "custom_semseg"

        self.rgb_path = os.path.join(self.dataset_path, self.rgb_foldername)
        self.mask_path = os.path.join(self.dataset_path, self.mask_foldername)
        self.semseg_path = os.path.join(self.dataset_path, self.semseg_foldername)
        self.custom_path = os.path.join(self.dataset_path, self.custom_foldername)
        self.csv_filename = os.path.join(self.dataset_path, "dataset.csv")

        self.counter = 0

        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.mask_path, exist_ok=True)
        os.makedirs(self.semseg_path, exist_ok=True)
        os.makedirs(self.custom_path, exist_ok=True)

        print (f"DatasetSaver loaded for {self.dataset_path}")

        if not os.path.exists(self.csv_filename):
            os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
            with open(self.csv_filename, "w", newline="") as f:
                csv.writer(f).writerow(["rgb_path","mask_path","semseg_path","custom_path","timestamp","throttle","steer","brake","speed"])



    def save_sample (self, timestamp, bgr, mask_rgb, semseg_img, custom_img, throttle, steer, brake, speed):
        
        rgb_filename  = f"rgb_{self.counter:08d}.png"
        mask_filename = f"mask_{self.counter:08d}.png"
        semseg_filename = f"semseg_{self.counter:08d}.png"
        custom_filename = f"custom_{self.counter:08d}.png"
        
        self.counter = self.counter + 1

        rgb_str = ""
        mask_str = ""
        semseg_str = ""
        custom_str = ""

        # saves only if is provided
        if bgr is not None:
            cv2.imwrite(os.path.join(self.rgb_path,  rgb_filename),  bgr)
            rgb_str = f"/{self.rgb_foldername}/{rgb_filename}"

        if mask_rgb is not None:
            cv2.imwrite(os.path.join(self.mask_path, mask_filename), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
            mask_str = f"/{self.mask_foldername}/{mask_filename}"

        if semseg_img is not None:
            cv2.imwrite(os.path.join(self.semseg_path, semseg_filename), semseg_img)
            semseg_str = f"/{self.semseg_foldername}/{semseg_filename}"

        if custom_img is not None:
            cv2.imwrite(os.path.join(self.custom_path, custom_filename), custom_img)
            custom_str = f"/{self.custom_foldername}/{custom_filename}"

        # write entry to csv
        with open(self.csv_filename, "a", newline="") as f:
            csv.writer(f).writerow([rgb_str, mask_str, semseg_str, custom_str, timestamp,
                                    throttle, steer, brake, speed])        
     

    
    def adjust_speed (self, csv_data_filename):

        try:
           
            self.load_speed_from_csv(
                self.csv_filename,
                csv_data_filename,
                dst_speed_col="speed",
                src_speed_col="speed_m_s"
            )
        except Exception as e:
            print(f"[ERROR] speed secuential match: {e}")



    def load_speed_from_csv(self, dataset_csv: str, speed_csv: str, dst_speed_col: str = "speed",
                            src_speed_col: str = "speed_m_s",   src_time_col: str = "sim_time"):

        if not os.path.isfile(dataset_csv):
            print(f" Unable to find dataset: {dataset_csv}")
            return
        if not os.path.isfile(speed_csv):
            print(f"Unable to find speed csv: {speed_csv}")
            return

        df_dst = pd.read_csv(dataset_csv)
        df_src = pd.read_csv(speed_csv)

        for col in ["timestamp", dst_speed_col]:
            if col not in df_dst.columns:
                print(f"Dataset does not have col: '{col}'.")
                return
        for col in [src_time_col, src_speed_col]:
            if col not in df_src.columns:
                print(f"CSV speed dataset does not have col: '{col}'.")
                return
        if df_dst.empty or df_src.empty:
            print("Warning: a file has empty data")
            return

        # Convert to num
        dst_ts = pd.to_numeric(df_dst["timestamp"], errors="coerce")
        src_ts = pd.to_numeric(df_src[src_time_col], errors="coerce")
        src_sp = pd.to_numeric(df_src[src_speed_col], errors="coerce")

        # Filter
        valid_src_mask = src_ts.notna() & src_sp.notna()
        if not valid_src_mask.any():
            print("Speed csv has incorrect data")
            return

        # 1) Keep first valid timestamp from speed csv
        first_src_idx = np.where(valid_src_mask.to_numpy())[0][0]
        first_src_time = float(src_ts.iloc[first_src_idx])

        # 2) Search same timestamp in the dataset to align both files 
        if dst_ts.notna().sum() == 0:
            print("Error finding timestamp")
            return

        # Fit index
        diffs = np.abs(dst_ts - first_src_time)
        anchor_dst_idx = int(diffs.idxmin())

        # 3) Prepare data to copy
        src_v = src_sp.iloc[first_src_idx:].to_numpy(dtype=float)

        # 4) Sequential copy
        n_dst = len(df_dst) - anchor_dst_idx
        n_src = len(src_v)
        n = min(n_dst, n_src)

        if n <= 0:
            print("Not enough space to copy speed")
            return

        df_dst.loc[anchor_dst_idx:anchor_dst_idx + n - 1, dst_speed_col] = src_v[:n]
        df_dst.to_csv(dataset_csv, index=False)

        # 5) Info
        print("[INFO] Fitting per timestamp:")
        print(f"  - first_src_time (vel CSV) = {first_src_time:.6f}")
        print(f"  - timestamp(dataset)[anchor] = {dst_ts.iloc[anchor_dst_idx]:.6f} (idx={anchor_dst_idx})")
        print(f"[OK] Cpied {n} speed data in '{dst_speed_col}' from {anchor_dst_idx} (sequential, ignored times from anchor.")
