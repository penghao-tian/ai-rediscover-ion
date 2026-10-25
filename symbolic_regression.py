"""
@Author: Penghao Tian <tph@ustc.edu.cn>
@Date: 2025-12-23 16:54
@Description:
Use the pySR library to fit physical formulas from the corresponding data parameters

"""

import os
import sys
import joblib
import scipy
import netCDF4
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

from datetime import datetime
import time

import argparse

from numba import njit, prange, objmode

from sklearn.metrics import r2_score

import sympy
from pysr import PySRRegressor, TemplateExpressionSpec
# ExpressionSpec



 # 0   cosmic_time         50699 non-null  datetime64[ns]
 # 1   year                50699 non-null  float32       
 # 2   month               50699 non-null  float32       
 # 3   day                 50699 non-null  float32       
 # 4   hour                50699 non-null  float32       
 # 5   lat                 50699 non-null  float32       
 # 6   lon                 50699 non-null  float32       
 # 7   alt                 50699 non-null  float32       
 # 8   lct                 50699 non-null  float32       
 # 9   s4max               50699 non-null  float32       
 # 10  icon_time           50699 non-null  datetime64[ns]
 # 11  latitude            50699 non-null  float64       
 # 12  longitude           50699 non-null  float64       
 # 13  magnetic_latitude   50699 non-null  float64       
 # 14  magnetic_longitude  50699 non-null  float64       
 # 15  local_solar_time    50699 non-null  float64       
 # 16  bx                  50699 non-null  float64       
 # 17  by                  50699 non-null  float64       
 # 18  bz                  50699 non-null  float64       
 # 19  b                   50699 non-null  float64       
 # 20  cosi                50699 non-null  float64       
 # 21  sini                50699 non-null  float64       
 # 22  cosd                50699 non-null  float64       
 # 23  sind                50699 non-null  float64       
 # 24  omegai              50699 non-null  float64       
 # 25  muin                50699 non-null  float64       
 # 26  r                   50699 non-null  float64       
 # 27  dudz_105            50699 non-null  float64       
 # 28  dvdz_105            50699 non-null  float64       
 # 29  dwidz_105           50699 non-null  float64       


# TODO 
# 1. Extracting formulas relating s4max to dudz and dvdz is complex and requires reducing the spatiotemporal resolution
# 2. Define s4max as Es(f | g | h), where f, g, and h represent space, time, and dynamical processes, respectively, to ultimately build an empirical formula model
# 3. For F-region s4max, parameters such as space, time, solar activity, atmospheric temperature, and density may need to be considered


class Process_Data():
    """
    Process the existing data 
    
    pro1: Select E-region dynamical process data
    pro2: Select E-region local time evolution data
    pro3: E-region DOY data
    pro4: E region  DOY localtime
    """
    
    def __init__(self, data_file, process="pro1"):
        
        self.process = process
        if self.process == "pro1":
            self.df = self._process_data_1(data_file)
            pass
        elif self.process == "pro2":
            self.df = self._process_data_2(data_file)
            pass
        elif self.process == "pro3":
            self.df = self._process_data_3(data_file)
            pass
        elif self.process == "pro4":
            self.df = self._process_data_4(data_file)
            pass
        elif self.process == "pro5":
            self.df = self._process_data_5(data_file)
            pass
        elif self.process == "pro6":
            self.df = self._process_data_6(data_file)
            pass
        elif self.process == "pro7":
            self.df = self._process_data_7(data_file)
            pass
        elif self.process == "pro8":
            self.df = self._process_data_8(data_file)
            pass
        else:
            raise
        pass
    
    def _read_file(self, data_file):
        """
        If data file is a single file, return the corresponding dataframe
        If it is a list, return the concatenated dataframe
        """
        if len(data_file) == 1:
            df = pd.read_hdf(data_file[0])
            return df
        
        if len(data_file) > 1:
            df_list = [pd.read_hdf(dd) for dd in data_file]
            df = pd.concat(df_list, axis=0)
            df.reset_index(drop=True, inplace=True)
            return df
        pass
    
    def _process_data_1(self, data_file):
        df = self._read_file(data_file)
        print(len(df))
        # breakpoint()
        return df
    
    def _process_data_2(self, data_file):
        # Process the relationship between lct and s4max in the E region
        # Primarily select cosmic1 data
        # ut = df.loc[:, "cosmic_time"].values
        # s4max = df.loc[:, "s4max"].values
        # doy = df.loc[:, "doy"].values
        # lat = df.loc[:, "lat"].values
        # lon = df.loc[:, "lon"].values
        # mlat = df.loc[:, "mlat"].values
        # alt = df.loc[:, "alt"].values
        # lct = df.loc[:, "lct"].values
        
        df = self._read_file(data_file)
        
        time_series = df.loc[:, "cosmic_time"]
        sindex = (time_series<=pd.to_datetime("2019-12-10"))
        
        df = df.loc[sindex, :]
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        lct = df.loc[:, "lct"].values
        s4max = df.loc[:, "s4max"].values
        
        grid_lct, grid_s4max = self._compute_count_1d_v1(lct, s4max, 
                                                      tag="E-lct-s4max", freq=None,
                                                      slide="1")
        
        ndf = pd.DataFrame({
                           "local_time": grid_lct,
                           "s4max": grid_s4max # grid_s4max 
                          })
        
        return ndf
    
    def _process_data_3(self, data_file):
        # Process the relationship between doy and s4max in the E region
        # 0.196 + \sin{\left(x_{0} \cdot 0.0316 - 0.778 \right)} \left(-0.0260\right)
        
        df = self._read_file(data_file)
        
        time_series = df.loc[:, "cosmic_time"]
        sindex = (time_series<=pd.to_datetime("2019-12-10"))
        
        df = df.loc[sindex, :]
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        lat = df.loc[:, "lat"].values
        lon = df.loc[:, "lon"].values
        s4max = df.loc[:, "s4max"].values
        
        grid_R, grid_C, grid_value, grid_N = self._compute_count_2d(lat, lon, 
                                                                    s4max,
                                                                    tag="lat-lon", slide="1",
                                                                    min_num=0
                                                                    )
        
        # breakpoint()
        
        # Select non-NaN data and flatten to one dimension
        flat_lat = grid_R.flatten()
        flat_lon = grid_C.flatten()
        flat_s4max = grid_value.flatten()
        select_index = (~np.isnan(flat_s4max))
        flat_lat = flat_lat[select_index]
        flat_lon = flat_lon[select_index]
        flat_s4max = flat_s4max[select_index]
        
        ndf = pd.DataFrame({"latitude": flat_lat,
                           "longitude": flat_lon,
                           "s4max": flat_s4max # grid_s4max 
                          })
        # breakpoint()
        
        return ndf
    
    def _process_data_4(self, data_file):
        # Process the relationship between doy, lct, and s4max in the E region
        df = self._read_file(data_file)
        
        time_series = df.loc[:, "time"]
        alt_series = df.loc[:, "alt"]
        s4max_series = df.loc[:, "s4max"]
        
        # Restrict the altitude to the E region
        sindex = (((alt_series>=80)&(alt_series<=130)) &
                  ((s4max_series>=0)&(s4max_series<=5))
                  )
        # time_series<=pd.to_datetime("2023-1-1")
        
        df = df.loc[sindex, :]
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        lct = df.loc[:, "lct"].values
        doy = df.loc[:, "doy"].values
        s4max = df.loc[:, "s4max"].values
        
        grid_doy, grid_lct, grid_s4max, Z = self._compute_count_2d(doy, lct, s4max, 
                                                      tag="E-doy-lct-s4max", 
                                                      slide="1")
        # breakpoint()
        
        flat_doy = grid_doy.flatten()
        flat_lct = grid_lct.flatten()
        flat_s4max = grid_s4max.flatten()
        
        df = pd.DataFrame({
            "doy": flat_doy,
            "local_time": flat_lct,
            "s4max": flat_s4max # grid_s4max 
                          })
        return df
    
    def _process_data_5(self, data_file):
        # Aggregate dwidz and s4max in space and time to reduce the influence of observed nonequilibrium data
        df = self._read_file(data_file)
        print(len(df))
        
        dwidz = df.loc[:, "dwidz_105"].values
        s4max = df.loc[:, "s4max"].values
        
        # grid_dwidz, grid_s4max, Z = self._compute_count_1d(dwidz, s4max, 
        #                                               tag="dwidz_1", 
        #                                               slide="1")
        
        # R_value, P_value = self._cal_r(grid_dwidz, grid_s4max)
        # R_value, P_value = self._cal_r(grid_dwidz, grid_s4max)
        # print(R_value)
        
        # breakpoint()
        
        ndf = pd.DataFrame({
            "dwidz_105": dwidz,
            "s4max": s4max # grid_s4max 
                          })
        # breakpoint()
        
        return ndf
    
    def _process_data_6(self, data_file):
        # Select dwidz and s4max data
        # Compute statistics for the corresponding r, sini, and other parameters using the dwidz grid
        
        df = self._read_file(data_file)
        print(len(df))
        
        dwidz = df.loc[:, "dwidz_105"].values
        s4max = df.loc[:, "s4max"].values
        
        r = df.loc[:, "r"].values
        sini = df.loc[:, "sini"].values
        cosi = df.loc[:, "cosi"].values
        sind = df.loc[:, "sind"].values
        cosd = df.loc[:, "cosd"].values
        
        bx = df.loc[:, "bx"].values
        by = df.loc[:, "by"].values
        bz = df.loc[:, "bz"].values
        b = df.loc[:, "b"].values
        
        dudz = df.loc[:, "dudz_105"].values
        dvdz = df.loc[:, "dvdz_105"].values
        
        
        # Select the corresponding stronger E-region phenomena
        # select_index = (s4max_lat >= 15) & (s4max_lat <= 45)
        # select_index = (s4max_lat >= 15) & (s4max_lat <= 45)
        # select_index = ((s4max_lat >= 20) & (s4max_lat <= 45) )
                        # (s4max_lct>=6) & (s4max_lct<=18) &
                        # (s4max_doy>=153) & (s4max_doy<=245))
        # dwidz = dwidz[select_index]
        # s4max = s4max[select_index]
        
        # Average over space and time, effectively removing nonequilibrium observations
        grid_dwidz, grid_s4max = self._compute_count_1d_v1(dwidz, s4max, 
                                                           tag="dwidz_1", freq=None,
                                                           slide="1")
        grid_dwidz, grid_r = self._compute_count_1d_v1(dwidz, r, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        grid_dwidz, grid_sini = self._compute_count_1d_v1(dwidz, sini, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        grid_dwidz, grid_cosi = self._compute_count_1d_v1(dwidz, cosi, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        grid_dwidz, grid_sind = self._compute_count_1d_v1(dwidz, sind, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        grid_dwidz, grid_cosd = self._compute_count_1d_v1(dwidz, cosd, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        
        grid_dwidz, grid_bx = self._compute_count_1d_v1(dwidz, bx, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        grid_dwidz, grid_by = self._compute_count_1d_v1(dwidz, by, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        grid_dwidz, grid_bz = self._compute_count_1d_v1(dwidz, bz, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        grid_dwidz, grid_b = self._compute_count_1d_v1(dwidz, b, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        
        grid_dwidz, grid_dudz = self._compute_count_1d_v1(dwidz, dudz, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        grid_dwidz, grid_dvdz = self._compute_count_1d_v1(dwidz, dvdz, 
                                                       tag="dwidz_1", freq=None,
                                                       slide="1")
        
        R1, _ = self._cal_r(grid_dwidz, grid_s4max)
        R2, _ = self._cal_r(grid_dudz, grid_s4max)
        R3, _ = self._cal_r(grid_b, grid_s4max)
        R4, _ = self._cal_r(grid_bx, grid_s4max)
        R5, _ = self._cal_r(grid_by, grid_s4max)
        R6, _ = self._cal_r(grid_bz, grid_s4max)
        print("dwidz and s4max:", R1)
        print("dudz and s4max:", R2)
        print("b and s4max:", R3)
        print("bx and s4max:", R4)
        print("by and s4max:", R5)
        print("bz and s4max:", R6)
        
        
        # R_value, P_value = self._cal_r(dwidz, s4max)
        # print(s4max.shape, R_value)

        ndf = pd.DataFrame({
            "r": grid_r,
            "sini": grid_sini,
            "cosi": grid_cosi,
            "sind": grid_sind,
            "cosd": grid_cosd,
            "bx": grid_bx,
            "by": grid_by,
            "bz": grid_bz,
            "b": grid_b,
            "dudz_105": grid_dudz,
            "dvdz_105": grid_dvdz,
            "dwidz_105": grid_dwidz,
            "s4max": grid_s4max # grid_s4max 
                          })
        
        ndf.dropna(inplace=True, ignore_index=True)
        
        # breakpoint()
        
        return ndf

    
    def _process_data_7(self, data_file):
        # Aggregate the other parameters and s4max spatially to reduce the influence of observed nonequilibrium data
        # Fit the data within each lat lon grid cell
        
        df = self._read_file(data_file)
        print(len(df))
        
        lat = df.loc[:, "lat"].values
        lon = df.loc[:, "lon"].values
        
        r = df.loc[:, "r"].values
        sini = df.loc[:, "sini"].values
        cosi = df.loc[:, "cosi"].values
        sind = df.loc[:, "sind"].values
        cosd = df.loc[:, "cosd"].values
        dudz = df.loc[:, "dudz_105"].values
        dvdz = df.loc[:, "dvdz_105"].values
        dwidz = df.loc[:, "dwidz_105"].values
        bx = df.loc[:, "bx"].values
        by = df.loc[:, "by"].values
        bz = df.loc[:, "bz"].values
        b = df.loc[:, "b"].values
        s4max = df.loc[:, "s4max"].values
        
        # grid_dwidz, grid_s4max, Z = self._compute_count_1d(dwidz, s4max, 
        #                                               tag="dwidz_1", 
        #                                               slide="1")
        grid_lat, grid_lon, grid_r, _ = self._compute_count_2d(lat, lon, 
                                                               r,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_sini, _ = self._compute_count_2d(lat, lon, 
                                                               sini,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_cosi, _ = self._compute_count_2d(lat, lon, 
                                                               cosi,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_sind, _ = self._compute_count_2d(lat, lon, 
                                                               sind,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_cosd, _ = self._compute_count_2d(lat, lon, 
                                                               cosd,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_dudz, _ = self._compute_count_2d(lat, lon, 
                                                               dudz,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_dvdz, _ = self._compute_count_2d(lat, lon, 
                                                               dvdz,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_dwidz, _ = self._compute_count_2d(lat, lon, 
                                                               dwidz,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        # geomag
        grid_lat, grid_lon, grid_bx, _ = self._compute_count_2d(lat, lon, 
                                                               bx,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_by, _ = self._compute_count_2d(lat, lon, 
                                                               by,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_bz, _ = self._compute_count_2d(lat, lon, 
                                                               bz,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        grid_lat, grid_lon, grid_b, _ = self._compute_count_2d(lat, lon, 
                                                               b,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        
        grid_lat, grid_lon, grid_s4max, _ = self._compute_count_2d(lat, lon, 
                                                               s4max,
                                                               tag="lat-lon", slide="1",
                                                               min_num=0)
        
        
        R_value, P_value = self._cal_r(grid_dwidz, grid_s4max)
        print(R_value)
        
        
        # Flatten to one dimension
        flat_lat = grid_lat.flatten()
        flat_lon = grid_lon.flatten()
        
        flat_r = grid_r.flatten()
        flat_sini = grid_sini.flatten()
        flat_cosi = grid_cosi.flatten()
        flat_sind = grid_sind.flatten()
        flat_cosd = grid_cosd.flatten()
        flat_dudz = grid_dudz.flatten()
        flat_dvdz = grid_dvdz.flatten()
        flat_dwidz = grid_dwidz.flatten()
        
        flat_bx = grid_bx.flatten()
        flat_by = grid_by.flatten()
        flat_bz = grid_bz.flatten()
        flat_b = grid_b.flatten()
        
        flat_s4max = grid_s4max.flatten()
        
        ndf = pd.DataFrame({
            "lat": flat_lat,
            "lon": flat_lon,
            "r": flat_r,
            "sini": flat_sini,
            "cosi": flat_cosi,
            "sind": flat_sind,
            "cosd": flat_cosd,
            "dudz_105": flat_dudz,
            "dvdz_105": flat_dvdz,
            "dwidz_105": flat_dwidz,
            "bx": flat_bx,
            "by": flat_by,
            "bz": flat_bz,
            "b": flat_b,
            "s4max": flat_s4max # grid_s4max 
                          })
        
        ndf.dropna(inplace=True, ignore_index=True)
        
        # breakpoint()
        return ndf
    
    def _process_data_8(self, data_file):
        # Provide the original formula for calculating dwidz
        
        df = self._read_file(data_file)
        print(len(df))
        
        dwidz = df.loc[:, "dwidz_105"].values
        
        r = df.loc[:, "r"].values
        sini = df.loc[:, "sini"].values
        cosi = df.loc[:, "cosi"].values
        sind = df.loc[:, "sind"].values
        cosd = df.loc[:, "cosd"].values
        dudz = df.loc[:, "dudz_105"].values
        dvdz = df.loc[:, "dvdz_105"].values
        
        
        ndf = pd.DataFrame({
            "r": r,
            "sini": sini,
            "cosi": cosi,
            "sind": sind,
            "cosd": cosd,
            "dudz_105": dudz,
            "dvdz_105": dvdz,
            "dwidz_105": dwidz,
                          })
        
        ndf.dropna(inplace=True, ignore_index=True)
        
        breakpoint()
        
        return ndf
    
        
    
        
    @staticmethod
    def _cal_r(array1, array2):
        """
        Compute the correlation between two sequences using non-NaN values
        """
        array1 = array1.flatten()
        array2 = array2.flatten()
        sindex = (~np.isnan(array1)) & (~np.isnan(array2))
        used_array1 = array1[sindex]
        used_array2 = array2[sindex]
        r, p = scipy.stats.spearmanr(used_array1, used_array2)
        return r, p
    
    @staticmethod
    @njit(parallel=True)
    def _compute_count_1d_v1(coords, value, tag="time", freq="5D", slide="1"):
        """
        Summarize the data distribution on a one-dimensional grid
        
        Compute the mean value within each grid cell
        
        Parameters:
        coords : 1D array
            Coordinate array
        value : 1D array
            Array of corresponding S4max values
        freq : float, optional; available only when tag is time
            Grid frequency (step size), default 0.05
        slide: Sliding step size; include all data within this step

        Returns:
        grid_coords : 1D array
            Grid coordinates
        grid_density : 1D array
            Mean value within each grid cell
        """
        with objmode(grid_coords="float64[:]", delta="float64"):
            if tag == "time":
                time_range = pd.date_range("2007-1-1", "2025-9-10", freq=freq)
                grid_coords = time_range.to_numpy().astype(np.float64)
                pass
            elif tag == "dudz_1":
                grid_coords = np.arange(-40., 40.01, 0.3)
                pass
            elif tag == "dudz_2":
                grid_coords = np.arange(-40, 40.01, 1.5)
                pass
            
            elif tag == "dvdz_1":
                grid_coords = np.arange(-40, 40.01, 0.3)
                pass
            elif tag == "dvdz_2":
                grid_coords = np.arange(-40, 40.01, 1.5)
                pass
            elif tag == "dwidz_1":
                grid_coords = np.arange(-1., 1., 0.01)
                pass
            elif tag == "dwidz_2":
                grid_coords = np.arange(-1., 1., 0.05)
                
            elif tag == "dwidz_3":
                grid_coords = np.arange(-5., 5., 0.01)
                pass
            elif tag == "dwidz_4":
                grid_coords = np.arange(-1., 1., 0.005)
                
            elif tag == "s4max-s4max":
                grid_coords = np.arange(0.025, 2, 0.05)
                pass
            
            elif tag == "E-alt-s4max":
                # alt 80~130 km
                # grid_coords = np.arange(80.5, 130, 1)
                grid_coords = np.arange(81., 130, 2)
                pass
            elif tag == "E-lct-s4max":
                # grid_coords = np.arange(0.5, 24, 1)
                grid_coords = np.arange(0.1, 24, 0.2)
                pass
            
            
            elif tag == "E-doy-s4max":
                # grid_coords = np.arange(0.5, 24, 1)
                grid_coords = np.arange(1., 365, 1.)
                pass
            elif tag == "E-mlat-s4max":
                grid_coords = np.arange(-88.,90,4)
                # grid_glat = np.arange(-89.5,90,1)
                # grid_glat = np.arange(-90+(1.5/2),90,1.5)
                # grid_glat = np.arange(-90+1., 90., 2)
                pass
            else:
                pass
            
            if slide == "1":
                delta = grid_coords[1] - grid_coords[0]
                pass
            elif slide == "2":
                delta = (grid_coords[1] - grid_coords[0]) * 2.
                pass
            else:
                pass
            pass

        
        
        # Total count
        N = np.zeros_like(grid_coords)
        Z = np.zeros_like(grid_coords)
        
        for i in prange(Z.shape[0]):
            for di in prange(len(value)):
                coordsi = coords[di]
                valuei = value[di]
                grid_coordsi = grid_coords[i]
                
                if (coordsi>=grid_coordsi-(delta/2))&(coordsi<grid_coordsi+(delta/2)):
                    N[i] += 1
                    Z[i] += valuei
                    pass
                pass
            pass
        
        # Compute the mean for each time interval
        grid_density = np.zeros_like(Z)
        for i in prange(Z.shape[0]):
            if (N[i] > 0):
                grid_density[i] = Z[i] / N[i]
            else:
                grid_density[i] = np.nan  # Set to NaN if no data are available
        
        return (grid_coords, grid_density)
    
    @staticmethod
    @njit(parallel=True)
    def _compute_count_1d_v2(coords, value, tag="time", freq="5D", slide="1"):
        """
        Note: This method requires excessive time (double) and memory and may cause an out-of-memory error 
              Not recommended unless analyzing boxplots of the data in each grid cell
              
              
        Summarize the data distribution on a one-dimensional grid
        
        Compute the mean value in each grid cell and also return all data in each cell
        for subsequent statistical analysis

        Parameters:
        coords : 1D array
            Coordinate array
        value : 1D array
            Array of corresponding S4max values
        freq : float, optional; available only when tag is time
            Grid frequency (step size), default 0.05
        slide: Sliding step size; include all data within this step

        Returns:
        grid_coords : 1D array
            Grid coordinates
        grid_density : 1D array
            Mean value within each grid cell
        Z: 2D array
            All data within each grid cell; NaN indicates invalid data
        
        """
        with objmode(grid_coords="float64[:]", delta="float64"):
            if tag == "time":
                time_range = pd.date_range("2007-1-1", "2025-9-10", freq=freq)
                grid_coords = time_range.to_numpy().astype(np.float64)
                pass
            elif tag == "dudz_1":
                grid_coords = np.arange(-40., 40.01, 0.3)
                pass
            elif tag == "dudz_2":
                grid_coords = np.arange(-40, 40.01, 1.5)
                pass
            
            elif tag == "dvdz_1":
                grid_coords = np.arange(-40, 40.01, 0.3)
                pass
            elif tag == "dvdz_2":
                grid_coords = np.arange(-40, 40.01, 1.5)
                pass
            elif tag == "dwidz_1":
                grid_coords = np.arange(-1., 1., 0.01)
                pass
            elif tag == "dwidz_2":
                grid_coords = np.arange(-1., 1., 0.05)
                
            elif tag == "dwidz_3":
                grid_coords = np.arange(-5., 5., 0.01)
                pass
            elif tag == "dwidz_4":
                grid_coords = np.arange(-1., 1., 0.005)
                
            elif tag == "s4max-s4max":
                grid_coords = np.arange(0.025, 2, 0.05)
                pass
            
            elif tag == "E-alt-s4max":
                # alt 80~130 km
                # grid_coords = np.arange(80.5, 130, 1)
                grid_coords = np.arange(81., 130, 2)
                pass
            elif tag == "E-lct-s4max":
                # grid_coords = np.arange(0.5, 24, 1)
                grid_coords = np.arange(0.1, 24, 0.2)
                pass
            
            
            elif tag == "E-doy-s4max":
                # grid_coords = np.arange(0.5, 24, 1)
                grid_coords = np.arange(1., 365, 1.)
                pass
            elif tag == "E-mlat-s4max":
                grid_coords = np.arange(-88.,90,4)
                # grid_glat = np.arange(-89.5,90,1)
                # grid_glat = np.arange(-90+(1.5/2),90,1.5)
                # grid_glat = np.arange(-90+1., 90., 2)
                pass
            else:
                pass
            
            if slide == "1":
                delta = grid_coords[1] - grid_coords[0]
                pass
            elif slide == "2":
                delta = (grid_coords[1] - grid_coords[0]) * 2.
                pass
            else:
                pass
            pass

        # Initialize the Z array to store all value entries in each grid cell
        Z = np.zeros((grid_coords.shape[0], len(value)))

        for i in prange(Z.shape[0]):
            k = 0  # The second axis stores each matching data point
            for di in prange(len(value)):
                coordsi = coords[di]
                valuei = value[di]
                grid_coordsi = grid_coords[i]

                # Store the matching data in Z
                if (coordsi >= grid_coordsi-(delta / 2)) & (coordsi < grid_coordsi+(delta / 2)):
                    Z[i, k] = valuei
                    k += 1

        # Since the number of valid data points in each grid cell is unknown, 
        # replace zeros in Z with NaN and then compute the mean using nanmean
        for i in prange(Z.shape[0]):
            for j in prange(Z.shape[1]):
                if Z[i, j] == 0:
                    Z[i, j] = np.nan

        # Initialize the grid_density array to store the mean of each grid cell
        grid_density = np.zeros((Z.shape[0]))
        for i in prange(Z.shape[0]):
            slice_data = Z[i, :]
            if np.all(np.isnan(slice_data)):
                grid_density[i] = np.nan  # Set to a missing value
            else:
                grid_density[i] = np.nanmean(slice_data)  # Compute the mean of non-NaN values

        return (grid_coords, grid_density, Z)
    
    @staticmethod
    @njit(parallel=True, cache=False)
    def _compute_count_2d(row_coords, col_coords, value, tag=None, slide="1", min_num=10):
        """
        Summarize the mean data distribution on a two-dimensional grid
        
        tag: Determine the grid range based on tag 
        slide: Determine whether the transitions between grid cells are smooth based on slide
        min_num: Minimum number of data points required for a grid cell to be considered valid
        
        return
        row_grid, col_grid: Gridded coordinates that can be plotted using pcolormesh
        average_Z: Mean values within grid cells; cells with fewer than min_num points are set to NaN
        N: Number of data points in each grid cell, usable for counting; the input value can be arbitrary in this case
        
        
        The grid is oriented intuitively
        The upper left is (row_coords[0,0], col_coords[0,0])
        The lower left is (row_coords[imax,0], col_coords[imax,0])
        """
        assert len(row_coords.shape) == 1
        assert len(col_coords.shape) == 1
        assert len(value.shape) == 1
        with objmode(row_grid="float64[:,:]", col_grid="float64[:,:]", 
                     delta_row="float64", delta_col="float64"):
            if tag == "E-doy-lct-s4max":
                row_grid, col_grid = np.meshgrid(np.linspace(1, 365, 200),
                                                 np.linspace(0, 24, 48),
                                                 indexing="ij")
                pass
            
            elif tag=="alt-lct":
                row_grid, col_grid = np.meshgrid(np.linspace(80,400,100),
                                                 np.linspace(0,24,100),
                                                 indexing="ij")
                pass
            elif tag=="alt-doy":
                row_grid, col_grid = np.meshgrid(np.linspace(80,400,100),
                                                 np.linspace(0,365,100),
                                                 indexing="ij")
                pass
            elif tag=="alt-lat-1":
                row_grid, col_grid = np.meshgrid(np.linspace(80,130,20),
                                                 np.linspace(-90,90,25),
                                                 indexing="ij")
                pass
            elif tag=="alt-lat-2":
                row_grid, col_grid = np.meshgrid(np.linspace(80,400,100),
                                                 np.linspace(-90,90,100),
                                                 indexing="ij")
                pass
            elif tag=="alt-lon":
                row_grid, col_grid = np.meshgrid(np.linspace(80,400,100),
                                                 np.linspace(-180,180,100),
                                                 indexing="ij")
                pass
            
            elif tag == "lat-lon":
                row_grid, col_grid = np.meshgrid(np.arange(-90.,90.1, 2.5),
                                                 np.arange(-180.,180.1, 5.),
                                                 indexing="ij")
                pass
            else:
                pass
            
            if slide == "1":
                delta_row = np.diff(row_grid, axis=0)[0,0]
                delta_col = np.diff(col_grid, axis=1)[0,0]
                pass
            elif slide == "2":
                delta_row = np.diff(row_grid, axis=0)[0,0] * 2.
                delta_col = np.diff(col_grid, axis=1)[0,0] * 2.
                pass
            else:
                pass
            pass
        Z = np.zeros_like(row_grid)
        N = np.zeros_like(row_grid)
        for i in prange(Z.shape[0]):
            for j in prange(Z.shape[1]):
                for data_index in prange(len(value)):
                    rr = row_coords[data_index]
                    cc = col_coords[data_index]
                    vv = value[data_index]
                    
                    rp = row_grid[i, j]
                    cp = col_grid[i, j]
                    if ((rr>=rp-(delta_row/2))&(rr<rp+(delta_row/2))&
                        (cc>=cp-(delta_col/2))&(cc<cp+(delta_col/2))):
                        N[i, j] += 1
                        Z[i, j] += vv
                        pass
                    pass
                pass
            pass
        
        # Compute the mean for each time interval
        average_Z = np.zeros_like(Z)
        for i in prange(Z.shape[0]):
            for j in prange(Z.shape[1]):
                if (N[i, j] > min_num):
                    average_Z[i, j] = Z[i, j] / N[i, j]
                else:
                    average_Z[i, j] = np.nan  # Set to NaN if no data are available
                    
        return (row_grid, col_grid, average_Z, N)
    
    pass


class Symbolic_Regression():
    """
    Use the previously available data
    to generate the corresponding physical formulas
    
    dataframe: Processed data dataframe
    variable: Select different parameters to use
    expression: Select different specific expressions
    """
    
    def __init__(self, dataframe, variable="var1", expression="exp1"):
        
        self.variable = variable
        self.expression = expression
        
        if self.variable == "var1":
            
            x_used_columns = ["r", "sini", "cosi", "sind", "cosd", "dudz_105", "dvdz_105"]
            
            y_used_columns = ["dwidz_105"]
            pass
        
        elif self.variable == "var2":
            x_used_columns = ["r", "sini", "cosi", "sind", "cosd", "dudz_105", "dvdz_105"]
            
            y_used_columns = ["s4max"]
            pass
        
        elif self.variable == "var3":
            # Using dwidz requires aggregation over spatiotemporal windows
            x_used_columns = ["dwidz_105"]
            y_used_columns = ["s4max"]
            
        elif self.variable == "var4":
            x_used_columns = ["r", "sini", "cosi", "sind", "cosd", "dwidz_105"]
            y_used_columns = ["s4max"]
        
        elif self.variable == "var5":
            x_used_columns = ["sini", "cosi", "sind", "cosd", "bx", "by", "bz", "b", "dwidz_105"]
            y_used_columns = ["s4max"]
            
        elif self.variable == "var6":
            x_used_columns = ["r", "sini", "b", "dwidz_105"]
            y_used_columns = ["s4max"]
            
        elif self.variable == "var7":
            x_used_columns = ["latitude", "longitude"]
            y_used_columns = ["s4max"]
        
        elif self.variable == "var8":
            x_used_columns = ["r", "sini", "cosi", "sind", "cosd", "bx", "by", "bz", "b", "dudz_105", "dvdz_105"]
            y_used_columns = ["s4max"]
          
            
        else:
            raise
        
        # x_used_columns = ["year", "month", "day", "hour",
        #                   "lat", "lon", "alt", "dwidz_105"]
        
        # y_used_columns = ["s4max"]
        
        self.X = dataframe.loc[:, x_used_columns].values
        self.Y = dataframe.loc[:, y_used_columns].values
        
        pass
    
    def get_expression(self,):
        # The variable order in varibale names matches that in self.X
        # In the results, #1 and #2 refer to argument positions in functions f and g, not positions in variable_names
        
        if self.expression == "exp1":
            # Separate dudz and dvdz to ensure dimensional consistency
            template = TemplateExpressionSpec(
                        expressions=["f", "g"],
                        variable_names=["r", "sinI", "cosI", 
                                        "sinD", "cosD", 
                                        "dudz", "dvdz"],
                        combine="f(r, sinI, cosI, sinD, cosD)*dudz + g(r, sinI, cosI, sinD, cosD)*dvdz",
                    )
            
            # template = TemplateExpressionSpec(
            #             expressions=["f", "g", "h", "j"],
            #             variable_names=["r", "sinI", "cosI", 
            #                             "sinD", "cosD", 
            #                             "dudz", "dvdz"],
            #             combine="h(f(r, sinI, cosI, sinD, cosD), dudz) + j(g(r, sinI, cosI, sinD, cosD), dvdz)",
            #         )
            
            
            pass
        
        elif self.expression == "exp2":
            template = TemplateExpressionSpec(
                        expressions=["f"],
                        variable_names=["r", "sinI", "cosI", 
                                        "sinD", "cosD", 
                                        "dudz", "dvdz"],
                        combine="f(r, sinI, cosI, sinD, cosD, dudz, dvdz)/(1 + r^2)",
                    )
            pass
        
        elif self.expression == "exp3":
            template = TemplateExpressionSpec(
                        expressions=["f", "g"],
                        variable_names=["r", "sinI", "cosI", 
                                        "sinD", "cosD", 
                                        "dudz", "dvdz"
                            ],
                        combine="f(r, sinI, cosI, sinD, cosD)*dudz/(1+r^2) + g(r, sinI, cosI, sinD, cosD)*dvdz/(1+r^2)",
                    )
            pass
        
        elif self.expression == "exp4":
            # f1x^-3 + f2x^-2+ f3x^-1 + f4 + f5x + f6x^2 + f7x^3
            template = TemplateExpressionSpec(
            expression=["f1","f2","f3", "f4", "f5","f6", "f7",],
            variable_names=["dwidz"],
            combine="f1*dwidz^(-3) + f2*dwidz^(-2)+ f3*dwidz^(-1) + f4 + f5*dwidz + f6*dwidz^(2) + f7*dwidz^(3)",
            )
            pass
        
        elif self.expression == "exp5":
            template = TemplateExpressionSpec(
                        expressions=["f", "g"],
                        variable_names=["sinI", "cosI", "sinD", "cosD", 
                                        "Bx", "By", "Bz", "B", "dwidz"],
                        combine="f(Bz)*g(dwidz)",
                    )
            pass
        
        elif self.expression == "exp6":
            template = TemplateExpressionSpec(
                        expressions=["f"],
                        variable_names=["sinI", "cosI", "sinD", "cosD", 
                                        "Bx", "By", "Bz", "B", "dwidz"],
                        combine="f(Bx, By, Bz, B)",
                    )
            pass
        
        elif self.expression == "exp7":
            template = TemplateExpressionSpec(
                        expressions=["f"],
                        variable_names=["sinI", "cosI", "sinD", "cosD", 
                                        "Bx", "By", "Bz", "B", "dwidz"],
                        combine="f(dwidz)",
                    )
            pass
        
        elif self.expression == "exp8":
            template = TemplateExpressionSpec(
                        expressions=["f"],
                        variable_names=["sinI", "cosI", "sinD", "cosD", 
                                        "Bx", "By", "Bz", "B", "dwidz"],
                        combine="f(Bz)",
                    )
            pass
        
        elif self.expression == "exp9":
            template = TemplateExpressionSpec(
                        expressions=["f", "g"],
                        variable_names=["sinI", "cosI", "sinD", "cosD", 
                                        "Bx", "By", "Bz", "B", "dwidz"],
                        combine="f(Bz)+g(dwidz)",
                    )
            pass
                
        
        else:
            template = None
            pass
        
        return template
    
    def train(self):
        # Define known possible expression forms
        expression_template = self.get_expression()
        
        
        # Define the model
        model = PySRRegressor(
            # batching=True,
            # batch_size=100,
            
            # Custom expression
            expression_spec=expression_template, 
            
            niterations=300,          # Start with a smaller value; it can be increased to 1000+ later
            populations=30,
            population_size=50,
            model_selection="best",   # Select the best single formula
            maxsize=30,               # Control formula complexity
            # maxdepth=6,
            
            # Relationship between two parameters
            binary_operators=["+", "-", "*", 
                              "/",
                             ],
            
            # Single-parameter relationship
            unary_operators=[
                # "square",     # x^2
                # "sin",
                # "cos",
                # "exp",
            # "sqrt",       # sqrt(x)
            # "log",        # log(x)
            # "exp",        # exp(x)
            # "sin",
            # "cos"
                            ],
            
            # Limit formula complexity
            constraints={"^": (9, 1),   # For pow, the maximum complexity is 9 for the left argument and 1 for the right argument
                         }, 
            nested_constraints={
                # "sin": {"sin": 0, "cos": 0, "exp": 0},  # sin cannot contain nested sin or cos
                # "cos": {"sin": 0, "cos": 0, "exp": 0},  # cos cannot contain nested sin or cos
                # "exp": {"exp": 0, "sin":0, "cos":0, "square":0}, # exp can only be nested once
                # "sqrt": {"sqrt": 0},
                # "square": {"square": 0},      # square can only be nested once
                                },
            
            
            # Use squared error as the loss function
            elementwise_loss="L2DistLoss()",
            
            output_directory="./symbolic_regression_outputs",
            
            # Set the seed to ensure consistent results
            # The seed can be set in serial mode
            # The seed has no effect in multithreading mode
            parallelism="multithreading",
            # random_state=42,
            # deterministic=True,
            
            # This parameter has no effect if parallel=serial
            # procs=32,  
            
            # extra_sympy_mappings=True
            
    )
        
        model.fit(self.X, self.Y)
        
        
        equations = model.equations_
        
        # breakpoint()
        
        
        # for idx, row in equations.iterrows():
        #     expr = row["sympy_format"]      # This column already contains SymPy expressions
        #     latex_str = sympy.latex(expr)      # Convert to a LaTeX string
        #     print(f"Equation {idx}, complexity={row['complexity']}, loss={row['loss']}:")
        #     print(latex_str)
        #     print()
        # print(model)              # Print all formulas on the Pareto front
        
        # Evaluate
        
        y_pred = model.predict(self.X)
        print("Use the folowing model:")
        # print(model.latex())
        R, P = self._cal_r(self.Y, y_pred)
        print("R2 on test:", R)
        
        plt.plot(self.X, self.Y, marker="o",c="k")
        plt.plot(self.X, y_pred, marker="*", c="red")
        plt.savefig("./test.png")
        
        pass
    
    @staticmethod
    def _cal_r(array1, array2):
        """
        Compute the correlation between two sequences using non-NaN values
        """
        array1 = array1.flatten()
        array2 = array2.flatten()
        sindex = (~np.isnan(array1)) & (~np.isnan(array2))
        used_array1 = array1[sindex]
        used_array2 = array2[sindex]
        r, p = scipy.stats.spearmanr(used_array1, used_array2)
        return r, p
    


if __name__ == "__main__":
    """
    python symbolic_regression.py arg1 arg2 arg3 arg4
    arg1. Specify the data file
    arg2. Specify the data file processing method
    arg3. Specify the variables to use
    arg4. Specify the concrete expressions for the variables
    """
    
    data_arg = str(sys.argv[1]) # data1 data2
    process_arg = str(sys.argv[2]) # pro1 pro2
    variable_arg = str(sys.argv[3]) # var1 var2
    expression_arg = str(sys.argv[4]) # none exp1 exp2
    
    if data_arg == "data1":
        data_file = ["./data/cosmic2_s4max_icon_wind_2019_2022_20min.h5"]
        pass
    elif data_arg == "data2":
        data_file = ["./data/cosmic2_s4max_2019_2024.h5"]
        pass
    elif data_arg == "data3":
        data_file = ["./config_02/icon_wind_group_a_based_kmeans_final_labels_25_42.h5"]
        
    elif data_arg == "data4":
        data_file = ["./config_02/icon_wind_group_b_based_kmeans_final_labels_25_42.h5"]
        
    elif data_arg == "data5":
        data_file = ["./config_02/icon_wind_group_c_based_kmeans_final_labels_25_42.h5"]
        
    elif data_arg == "data6":
        # Joint observation data from icon and cosmic2
        data_file = ["./config_02/icon_wind_group_b_based_kmeans_final_labels_25_42.h5",
                     "./config_02/icon_wind_group_c_based_kmeans_final_labels_25_42.h5"]
        
    elif data_arg == "data7":
        data_file = ["./config_02/icon_wind_group_a_based_kmeans_final_labels_25_42.h5",
                     "./config_02/icon_wind_group_b_based_kmeans_final_labels_25_42.h5",
                     "./config_02/icon_wind_group_c_based_kmeans_final_labels_25_42.h5"]
        
    elif data_arg == "data8":
        data_file = [
                     "./config_02/icon_wind_group_d_based_kmeans_final_labels_25_42.h5",
                     "./config_02/icon_wind_group_e_based_kmeans_final_labels_25_42.h5"]
        
    elif data_arg == "data9":
        data_file = [
                     "./config_02/group_b_based_kmeans_final_labels_25_42.h5",
                     "./config_02/group_c_based_kmeans_final_labels_25_42.h5"]
    
        
    else:
        raise
    
    # Preprocess the data
    obj_1 = Process_Data(data_file=data_file, process=process_arg)
    used_df = obj_1.df
    
    # breakpoint()
    
    
    # Run symbolic regression
    obj_2 = Symbolic_Regression(dataframe=used_df,
                                variable=variable_arg, 
                                expression=expression_arg)
    obj_2.train()
    # breakpoint()
    
    