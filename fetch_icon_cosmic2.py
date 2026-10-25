"""
@Author: Penghao Tian <tph@ustc.edu.cn>
@Data: 2025-12-17 21:27
@Desription:

Jointly select ICON wind field data and COSMIC2 data
Use a temporal window of 20 min and a spatial window of 2.5 degrees

Final data format
year: Year
month: Month
day: Day
hour: Hour, 0-24
time: datetime64
alt: altitude, km, 80-130
lat: latitude, degree, +- mean North and South
lon: longitude, degree, +- mean East and West
lct: local time, 0-24
s4max: 0-5
bx: Geomagnetic field Bx, T
by: By, T
bz: Bz, T
b: Geomagnetic field B, T
cosi: cos(I), I means inclination
sini: sin(I)
cosd: cos(D), D declination
sind: sin(D)

omegai: omega_ion, gyrofrequency, q*B/m, Hz, T**-1

muin: mu_ion_neutral, collision frequency between metallic ions and neutral species, based on the empirical formula: 2.6 * 10**(-9) * total_number_density / np.sqrt(mean_molecular_mass), Hz

r: muin / omegai, dimensionless

dudz_105: du/dz in 105km, u is the zonal wind, Hz
dvdz_105: dv/dz

dwidz_105: dwi/dz, wi is the vertical component of metal ion drift

"""

import os 
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import einops
import configparser
from datetime import datetime

from collections import defaultdict

import igrf

from numba import njit, prange




class Fetch_ICON_COSMIC():
    """
    Extract COSMIC2 and ICON data within the same spatiotemporal window
    Use COSMIC data as the reference: iterate over s4max data and select ICON data within the corresponding spatiotemporal window
    Average multiple ICON observations within the same spatiotemporal grid cell
    
    icon_file: Processed ICON file containing wind fields and other information
    s4max_file: Processed S4max HDF5 file containing spatiotemporal information for S4max
    time_interval: Temporal interval range
    space_interval: Spatial interval range (lat_res, lon_res)
    save_file_name: Location for saving the HDF5 file
    
    """
    def __init__(self, icon_file=None, s4max_file=None, save_file_name=None,
                 ) -> None:
        self.s4max_df = self._load_s4max_file(s4max_file)
        self.icon_df = self._load_icon_file(icon_file)
        
        self.save_file_name = save_file_name
        pass
    
    def _load_s4max_file(self, fname):
        """
        Load the corresponding processed S4max HDF5 file
        
        Handle the time and altitude ranges here
        """
        df = pd.read_hdf(fname)
        
        time_series = df.loc[:, "cosmic_time"]
        alt_series = df.loc[:, "alt"]
        s4max_series = df.loc[:, "s4max"]
        
        # Restrict the altitude to the E region
        sindex = ((alt_series>=80)&(alt_series<=130) &
                  (s4max_series>=0)&(s4max_series<=5) &
                  (time_series>=pd.to_datetime("2019-12-1")) &
                  (time_series<=pd.to_datetime("2022-12-1")))
        
        df = df.loc[sindex, :]
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # df.rename(columns={"time": "cosmic_time"}, inplace=True)
        
        # Select the first 100,000 entries
        # df = df.loc[:100000, :]
        # print(f"s4max df: {len(df)}")
        
        return df
    
    def _load_icon_file(self, fname):
        df = pd.read_hdf(fname)
        df.rename(columns={"time": "icon_time"}, inplace=True)
        
        # Select the first 100,000 entries
        # df = df.loc[:100000, :]
        
        # print(f"icon df: {len(df)}")
        return df
    
    def fetch_all_file(self,):
        s4max_time = self.s4max_df.loc[:, "cosmic_time"].values
        s4max_lat = self.s4max_df.loc[:, "lat"].values
        s4max_lon = self.s4max_df.loc[:, "lon"].values
        
        icon_time = self.icon_df.loc[:, "icon_time"].values
        icon_lat = self.icon_df.loc[:, "latitude"].values
        icon_lon = self.icon_df.loc[:, "longitude"].values
        
        # Convert time to numerical values [ms] for subsequent acceleration with numba
        s4max_time = s4max_time.astype("datetime64[ms]").astype(np.int64)
        icon_time = icon_time.astype("datetime64[ms]").astype(np.int64)
        
        
        # breakpoint()
        
        s4max_index, icon_index = self.fetch_loop(s4max_time, s4max_lat, s4max_lon,
                                                  icon_time, icon_lat, icon_lon)
        
        
        # icon_index2, s4max_index2 = self.fetch_loop(icon_time, icon_lat, icon_lon,
        #                                           s4max_time, s4max_lat, s4max_lon)
        
#         s4max_index1 = np.asarray(s4max_index1)
#         icon_index1 = np.asarray(icon_index1)
#         s4max_index2 = np.asarray(s4max_index2)
#         icon_index2 = np.asarray(icon_index2)
        
        # print(np.array_equal(np.sort(s4max_index1), np.sort(s4max_index2)))
        # print(np.array_equal(np.sort(icon_index1), np.sort(icon_index2)))
        
        # breakpoint()
        
        
        # print(s4max_index.shape)
        # print(np.unique(s4max_index).shape)
        # print(icon_index.shape)
        # print(np.unique(icon_index).shape)
        # Handle cases where one s4max corresponds to multiple icon files, or vice versa
        (df1_agg, 
         df2_agg, mapping) = self.aggregate_by_connected_components(
            self.s4max_df,
            self.icon_df, s4max_index, icon_index)
        # breakpoint()
        
        # Save the data
        con_df = pd.concat([df1_agg, df2_agg], axis=1)
        con_df.reset_index(drop=True, inplace=True)
        con_df.to_hdf(f"{self.save_file_name}", key="data")
        print(con_df.info())
        
        return con_df
    
    @staticmethod
    @njit(parallel=False, nogil=True)
    def fetch_loop(time1, lat1, lon1, time2, lat2, lon2):
        """
        Convert time to ms format 
        Use time.astype("datetime64[ms]").astype(np.int64)

        lat ranges from -90 to 90
        lon -180 180

        """
        # select_index_1 = np.zeros_like(time1)
        # select_index_2 = np.zeros_like(time2)
        # coords_array = np.zeros((len(time1), len(time2)))

        time_threshold = 20  # min
        lat_threshold = 2.5
        lon_threshold = 2.5
        
        # sel1 = np.zeros(time1.shape[0], dtype=np.uint8)
        # sel2 = np.zeros(time2.shape[0], dtype=np.uint8)
        
        sel1 = []
        sel2 = []
        # k = 0
        for i in range(len(time1)):
            for j in range(len(time2)):
                dt = abs(time2[j] - time1[i]) / 60000.0
                if dt > time_threshold:
                    continue

                dlat = abs(lat2[j] - lat1[i])
                if dlat > lat_threshold:
                    continue

                dlon = abs((lon2[j] - lon1[i] + 180.0) % 360.0 - 180.0)
                if dlon > lon_threshold:
                    continue
                
                sel1.append(i)
                sel2.append(j)
                pass

        return np.array(sel1), np.array(sel2)
    

    @staticmethod
    def aggregate_by_connected_components(df1: pd.DataFrame,
                                         df2: pd.DataFrame,
                                         arr1: np.ndarray,
                                         arr2: np.ndarray):
        """
        arr1 and arr2 have equal lengths and represent pairings between df1.index and df2.index (possibly repeated).
        The outputs df1_agg and df2_agg have the same number of rows and correspond one-to-one by component_id.
        """
        arr1 = np.asarray(arr1)
        arr2 = np.asarray(arr2)
        if arr1.shape != arr2.shape:
            raise ValueError("arr1 and arr2 must have the same length / shape.")
            
        dsu = DSU()
        
        # 1) Build the graph and merge using union-find: A: df1 side, B: df2 side
        for a, b in zip(arr1, arr2):
            node_a = ("A", int(a))
            node_b = ("B", int(b))
            dsu.union(node_a, node_b)

        # 2) Collect nodes by connected component
        comp = defaultdict(lambda: {"A": set(), "B": set()})
        for a, b in zip(arr1, arr2):
            ra = dsu.find(("A", int(a)))
            rb = dsu.find(("B", int(b)))
            root = dsu.find(ra)  # Use a common root
            comp[root]["A"].add(int(a))
            comp[root]["B"].add(int(b))

        # 3) Average all columns
        num1 = df1.columns
        num2 = df2.columns

        rows1, rows2, meta = [], [], []

        # Sort components consistently (by the minimum A index, then the minimum B index)
        def sort_key(item):
            a_min = min(item[1]["A"]) if item[1]["A"] else 10**18
            b_min = min(item[1]["B"]) if item[1]["B"] else 10**18
            return (a_min, b_min)

        for cid, (root, nodes) in enumerate(sorted(comp.items(), key=sort_key)):
            a_idx = sorted(nodes["A"])
            b_idx = sorted(nodes["B"])

            # Keep only indices that actually exist in the df index (to avoid KeyError)
            a_exist = [i for i in a_idx if i in df1.index]
            b_exist = [i for i in b_idx if i in df2.index]

            if len(a_exist) == 0 or len(b_exist) == 0:
                # This should not occur (every edge has both A and B), but guard against it
                continue

            mean1 = df1.loc[a_exist, num1].mean(axis=0)
            mean2 = df2.loc[b_exist, num2].mean(axis=0)

            rows1.append(mean1)
            rows2.append(mean2)
            meta.append({
                "component_id": cid,
                "df1_indices": a_idx,
                "df2_indices": b_idx,
                "n_df1": len(a_idx),
                "n_df2": len(b_idx),
            })

        df1_agg = pd.DataFrame(rows1, columns=num1)
        df2_agg = pd.DataFrame(rows2, columns=num2)
        df1_agg.index.name = "component_id"
        df2_agg.index.name = "component_id"

        mapping = pd.DataFrame(meta)
        # .set_index("component_id")

        return df1_agg, df2_agg, mapping


class DSU:
    """
    Handle identical indices
    """
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            


def main1():
    
    icon_file = "./data/icon_wind_2019_2022.h5"
    
    # s4max_file = "./data/cosmic2_s4max_2019_2024.h5"
    # save_file_name = "./data/cosmic2_s4max_icon_wind_2019_2022_20min.h5"
    
    # Process the group contents
    s4max_file = "./config_02/group_a_based_kmeans_final_labels_25_42.h5"
    save_file_name = "./config_02/icon_wind_group_a_based_kmeans_final_labels_25_42.h5"
    
    obj = Fetch_ICON_COSMIC(icon_file=icon_file, 
                            s4max_file=s4max_file, 
                            save_file_name=save_file_name)
    obj.fetch_all_file()
    
    return

def main2():
    
    icon_file = "./data/icon_wind_2019_2022.h5"
    
    # s4max_file = "./data/cosmic2_s4max_2019_2024.h5"
    # save_file_name = "./data/cosmic2_s4max_icon_wind_2019_2022_20min.h5"
    
    # Process the group contents
    s4max_file = "./config_02/group_b_based_kmeans_final_labels_25_42.h5"
    save_file_name = "./config_02/icon_wind_group_b_based_kmeans_final_labels_25_42.h5"
    
    obj = Fetch_ICON_COSMIC(icon_file=icon_file, 
                            s4max_file=s4max_file, 
                            save_file_name=save_file_name)
    obj.fetch_all_file()
    
    return

def main3():
    
    icon_file = "./data/icon_wind_2019_2022.h5"
    
    # s4max_file = "./data/cosmic2_s4max_2019_2024.h5"
    # save_file_name = "./data/cosmic2_s4max_icon_wind_2019_2022_20min.h5"
    
    # Process the group contents
    s4max_file = "./config_02/group_c_based_kmeans_final_labels_25_42.h5"
    save_file_name = "./config_02/icon_wind_group_c_based_kmeans_final_labels_25_42.h5"
    
    obj = Fetch_ICON_COSMIC(icon_file=icon_file, 
                            s4max_file=s4max_file, 
                            save_file_name=save_file_name)
    obj.fetch_all_file()
    
    return

def main4():
    
    icon_file = "./data/icon_wind_2019_2022.h5"
    
    # s4max_file = "./data/cosmic2_s4max_2019_2024.h5"
    # save_file_name = "./data/cosmic2_s4max_icon_wind_2019_2022_20min.h5"
    
    # Process the group contents
    s4max_file = "./config_02/group_d_based_kmeans_final_labels_25_42.h5"
    save_file_name = "./config_02/icon_wind_group_d_based_kmeans_final_labels_25_42.h5"
    
    obj = Fetch_ICON_COSMIC(icon_file=icon_file, 
                            s4max_file=s4max_file, 
                            save_file_name=save_file_name)
    obj.fetch_all_file()
    
    return

def main5():
    
    icon_file = "./data/icon_wind_2019_2022.h5"
    
    # s4max_file = "./data/cosmic2_s4max_2019_2024.h5"
    # save_file_name = "./data/cosmic2_s4max_icon_wind_2019_2022_20min.h5"
    
    # Process the group contents
    s4max_file = "./config_02/group_e_based_kmeans_final_labels_25_42.h5"
    save_file_name = "./config_02/icon_wind_group_e_based_kmeans_final_labels_25_42.h5"
    
    obj = Fetch_ICON_COSMIC(icon_file=icon_file, 
                            s4max_file=s4max_file, 
                            save_file_name=save_file_name)
    obj.fetch_all_file()
    
    return


if __name__ == "__main__":
    """
    main1-5 process the five classified categories
    """
    print(f"Start time is {datetime.now()}")
    print("Script start.")
    print("######################")
    # main1()
    # print("######################")
    # main2()
    # print("######################")
    # main3()
    print("######################")
    main4()
    print("######################")
    main5()
    print("Data pre-process has completed.")
    print(f"Complete time is {datetime.now()}")
    pass





