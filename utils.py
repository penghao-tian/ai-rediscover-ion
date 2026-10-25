# Small utility functions

import shutil

import requests
import tarfile
import chardet

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from timezonefinder import TimezoneFinder

from numba import njit, prange, objmode

import igrf

class Aux1():
    """
    Helper functions for processing geomagnetic quantities and local time
    
    local time -> universal time
    universal time -> local time
    
    Convert to magnetic latitude
    Compute the global magnetic field configuration
    """
    @staticmethod
    def lct2ut(time, lat=-35.32, lon=149.0):
        tzf = TimezoneFinder()
        local_zone = tzf.timezone_at(lng=lon, lat=lat)
        local_time = time.tz_localize(local_zone, 
                                      ambiguous=True,
                                      nonexistent="shift_forward")
        ut_time = local_time.tz_convert(None)

        return ut_time
    
    @staticmethod
    def ut2lct(time, lat=-35.32, lon=149.0):
        tzf = TimezoneFinder()
        ut_time = time.tz_localize("UTC")
        local_zone = tzf.timezone_at(lng=lon, lat=lat)
        local_time = ut_time.tz_convert(local_zone)

        # The dt operation cannot be used with the local time zone attached
        # Remove the local time zone
        local_time = pd.Timestamp(year=local_time.year,
                                  month=local_time.month,
                                  day=local_time.day,
                                  hour=local_time.hour,
                                  minute=local_time.minute,
                                  second=local_time.second)
        return local_time
    
    @staticmethod
    def datetime64_to_hours_of_day(utc_dt64: np.ndarray) -> np.ndarray:
        """
        Convert a numpy datetime64[*] array to hours within the day (0–24).
        Pure NumPy with C-level vectorization, suitable for tens of millions of observations.
        """
        # Remove the date component and retain only the time offset within the day (timedelta64)
        tod = utc_dt64 - utc_dt64.astype('datetime64[D]')
        # Convert to seconds, then hours (use 'ms'/'us' if required by the desired precision)
        hours = tod / np.timedelta64(1, 's') / 3600.0
        return hours.astype(np.float64, copy=False)

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def utc_hours_and_lon_to_local_time(utc_hours: np.ndarray, lon_deg: np.ndarray):
        """
        Compute local time (local solar time, 0–24) from the UTC hour of the day and longitude.
        Returns: local_time (0–24), day_offset (day offset relative to UTC: ..., -1, 0, +1, ...)
        """
        n = utc_hours.shape[0]
        lct = np.zeros(n, dtype=np.float64)
        day_offset = np.zeros(n, dtype=np.float64)
        
        
        for i in prange(n):
            
            # Indexing the original array yields (float64, 1d, C)
            # The result cannot be stored in the array after subsequent operations; first use item to obtain a scalar
            uh = utc_hours[i].item()   # Explicitly extract a scalar
            lon = lon_deg[i].item()      # Explicitly extract a scalar

            # Normalize longitude to [-180, 180)
            lon = ((lon + 180.0) % 360.0) - 180.0
            
            t = uh + lon / 15.0          # Local time that may cross a day boundary (before modulo)
            q = np.floor(t / 24.0)     # Whole-day offset (may be negative)
            
            # print(q)
            # print(t)
            # print(t-q*24.0)
            
            lct[i] += (t - q * 24.0)         # Map to [0, 24)
            
            day_offset[i] += q
            pass
        
        return lct, day_offset
    
    @staticmethod
    @njit(parallel=True)
    def glat2diplat(glat, glon, alt):
        """
        Convert the given longitude, latitude, and altitude to dip latitude
        """
        assert len(glat.shape)==1
        assert len(glon.shape)==1
        assert len(alt.shape)==1
        assert glat.shape==alt.shape
        
        # Convert geographic longitude and latitude to dip latitude
        year = 2020
        with objmode(g="float64[:,:]", h="float64[:,:]"):
            g, h = igrf.ReadCoeffs(filename="./igrf14coeffs.txt").get_coeffs(year)
            pass
        dip_lat = np.zeros_like(glat)
        for i in prange(len(glat)):
            (dip, 
             gd, gi, 
             bh, bx, by, bz, bf) = igrf.igrf_value(g, h, 
                                                   glat[i], glon[i], alt[i], year)
            # Dip latitude
            dip_lat[i] = dip
            pass
        return dip_lat
    
    @staticmethod
    @njit(parallel=True)
    def compute_global_magnetic():
        """
        Compute global magnetic dip latitude and magnetic declination
        
        At an altitude of 100 km for the year 2014
        """
        with objmode(glat="float64[:,:]", glon="float64[:,:]"):
            glat, glon = np.meshgrid(np.arange(-90,90,0.25)[1::2], 
                                     np.arange(-180,180,0.25)[1::2],
                                     indexing="ij")
            pass
        dip_lat = np.zeros_like(glat)
        decl = np.zeros_like(glat)
        alt = 100
        year = 2020
        
        with objmode(g="float64[:,:]", h="float64[:,:]"):
            g, h = igrf.ReadCoeffs(filename="./igrf14coeffs.txt").get_coeffs(year)
            pass
        
        for i in prange(glat.shape[0]):
            for j in prange(glat.shape[1]):
                lat = glat[i, j]
                lon = glon[i, j]
                dip, gd, gi, bh, bx, by, bz, bf = igrf.igrf_value(g, h, lat, lon, alt, year)
                dip_lat[i, j] = dip
                decl[i, j] = gd
                pass
            pass
        return (glat, glon, dip_lat, decl)
    

    pass

class Aux2():
    """
    Download the NRLMSIS2.0 neutral atmosphere numerical model
    """
    @staticmethod
    def download_file(url, dest_path):
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check whether the request succeeded

        # Save the file locally
        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pass
            pass
        pass
    
    @staticmethod
    def extract_tar_gz(tar_path, extract_path):
        # Extract the tar.gz file
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
            pass
        pass
    
    @staticmethod
    def main():
        # File URL and save path
        url = 'https://map.nrl.navy.mil/map/pub/nrl/NRLMSIS/NRLMSIS2.0/NRLMSIS2.0.tar.gz'
        dest_path = './NRLMSIS2.0.tar.gz'
        extract_path = './NRLMSIS2.0'

        # Download the tar.gz file
        # Aux2.download_file(url, dest_path)
        print(f"Downloaded file to {dest_path}")

        # Extract the file
        os.makedirs(extract_path, exist_ok=True)
        Aux2.extract_tar_gz(dest_path, extract_path)
        print(f"Extracted files to {extract_path}")
        pass
    pass


class Aux3():
    """
    Convert files to UTF8 encoding so they can be read
    """

    @staticmethod
    def convert_to_utf8(file_path):
        with open(file_path, 'rb') as f:
            raw_data = f.read()

        # Detect the file encoding
        result = chardet.detect(raw_data)
        current_encoding = result['encoding']

        if current_encoding is None:
            print(f"Unable to detect the encoding of {file_path}")
            return

        # Re-encode the file contents as UTF-8
        try:
            decoded_data = raw_data.decode(current_encoding)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(decoded_data)
            print(f"Converted {file_path} to UTF-8 encoding")
        except Exception as e:
            print(f"Error converting {file_path}: {e}")
            pass
        pass
    
    @staticmethod
    def batch_convert_to_utf8(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.f90'):
                    file_path = os.path.join(root, file)
                    Aux3.convert_to_utf8(file_path)
                    pass
                pass
            pass
        pass
    
    @staticmethod
    def main():
        # Specify the target directory
        target_directory = './NRLMSIS2.0'

        # Batch-convert file encoding to UTF-8
        Aux3.batch_convert_to_utf8(target_directory)
        pass
    
    pass


class Aux4():
    """
    Convert raw ICON files into files suitable for straightforward plotting
    
    Raw files
    1. Remove unnecessary variables for faster reading and plotting
    2. Perform additional longitude processing
    3. Remove unnecessary altitude information
    """
    
    @staticmethod
    def transform_longitude(longitude_matrix):
        # Convert longitude values from 0 to 360 to -180 to 180
        converted_matrix = np.where(longitude_matrix > 180, longitude_matrix - 360, 
                                    longitude_matrix)
        return converted_matrix
    
    @staticmethod
    def main():
        file_path = Path("./data/MIGHTI_WIND_2019_2021_ion_drift.nc")
        save_file_path = Path("./data/MIGHTI_WIND_2019_2021_ion_drift_26_Alt.nc")
        
        ds = xr.open_dataset(file_path)
        ds.close()
        
        # Select the first 26 altitudes
        nds = ds.isel(ICON_L22_Altitude=slice(26))
        
        # Convert longitude
        long = nds.ICON_L22_Longitude.data
        trans_long = Aux4.transform_longitude(long)
        nds.ICON_L22_Longitude.data = trans_long
        
        # Convert time
        # time_array = pd.to_datetime(nds.ICON_L22_UTC_Time.data).tz_localize(None)
        # nds.ICON_L22_UTC_Time.data = time_array.to_numpy()
        
        # lat_array = nds.ICON_L22_UTC_Time.data
        # time_array, lat_array, lon_array
        print(nds)
        nds.to_netcdf(save_file_path, mode="w", format="NETCDF4", engine="netcdf4")
        pass
    pass


class Aux5():
    """
    Set the current working directory in main
    Set the directory name prefix to clean up in startswith
    
    Running this clears the contents of all directories under this directory, retaining only config.ini files
    """
    
    @staticmethod
    def delete_non_config_files(base_dir):
        # Convert base_dir to a Path object
        base_path = Path(base_dir)
        print(base_path.resolve())
        
        # breakpoint()
        
        # Get all subdirectories starting with 'config_'
        # or 'symbolic_regression_'
        subdirectories = [subdir for subdir in base_path.iterdir() if subdir.is_dir() and subdir.name.startswith('symbolic_regression_')]

        for subdir in subdirectories:
            # Subdirectory path
            print(f"Processing directory: {subdir}")

            # Get all files in the subdirectory
            for file in subdir.iterdir():
                if file.is_file():
                    # For files, keep ini files and delete the others
                    if file.name != 'config.ini':
                        print(f"Deleting {file}")
                        file.unlink()  # Delete the file
                    else:
                        print(f"Keeping {file}")
                        pass
                    pass
                if file.is_dir():
                    # For directories, delete all contents
                    print(f"Remove dir {file}")
                    shutil.rmtree(file)


            # Delete the entire subdirectory if there is no config.ini file
            # if not (subdir / 'config.ini').exists():
            #     print(f"Deleting empty directory {subdir}")
                # subdir.rmdir()  # Delete the empty directory

    # Usage example:
    @staticmethod
    def main():
        base_directory = "./"  # Replace with the actual path
        Aux5.delete_non_config_files(base_directory)
        pass
    
    pass


class Aux6():
    """
    Count files with a specific name in the corresponding directory
    """
    
    @staticmethod
    def main():
        ff = open("./data/utils_Aux6.log", "a")
        Aux6.seek_cosmic_1(ff)
        Aux6.seek_cosmic_2(ff)
        ff.close()
    
    @staticmethod
    def seek_cosmic_1(ff):
        file_list = []
        seek_path = Path("/database9/tianph/ion_dataset/cosmic1/scnLv1")
        assert seek_path.is_dir()
        seek_file_name = "scnLv1_*_nc"
        
        for file in seek_path.rglob(seek_file_name):
            if file.stat().st_size != 0:
                file_list.append(file)
            pass
        print("###################")
        print(f"Seek path: {seek_path.resolve()}", file=ff)
        print(f"seek file: {seek_file_name}", file=ff)
        print(f"Number: {len(file_list)}", file=ff)
        pass
    
    @staticmethod
    def seek_cosmic_2(ff):
        file_list = []
        seek_path = Path("/database9/tianph/ion_dataset/cosmic2/podTc2")
        assert seek_path.is_dir()
        seek_file_name = "podTc2_*_nc"
        
        for file in seek_path.rglob(seek_file_name):
            if file.stat().st_size != 0:
                file_list.append(file)
            pass
        print(f"Seek path: {seek_path.resolve()}", file=ff)
        print(f"seek file: {seek_file_name}", file=ff)
        print(f"Number: {len(file_list)}", file=ff)
        pass
    pass

class Aux7():
    """
    Download the corresponding near-Earth solar wind parameters from the omniweb website
    
    1. Select the content and time range to download at https://omniweb.gsfc.nasa.gov/form/dx1.html and select create file
    2. Download the information from the URL locally, then process it into an h5 file
    """
    @staticmethod
    def download_file(url, dest_path):
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check whether the request succeeded

        # Save the file locally
        with open(dest_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pass
            pass
        pass
    
    @staticmethod
    def process_file(dest_path):
        
        df = pd.read_csv(dest_path, sep="\s+", header=None,
                        names=["year", 'doy', 'hour', 'r','f107','lyman_alpha'])

        for i in range(len(df)):
            year = df.loc[i, "year"]
            doy = df.loc[i, "doy"]
            time = pd.to_datetime(f"{year}-{doy}", format="%Y-%j")
            df.loc[i, "time"] = time
            pass

        df.to_hdf("./data/omni_daily_solar_activity.h5", key="data")
    
    @staticmethod
    def main():
        # File URL and save path
        url = "https://omniweb.gsfc.nasa.gov/staging/omni2_daily_bOIFbzVmG1.lst"
        dest_path = './data/omni_daily_solar_activity.lst'

        # Download the file
        # Aux7.download_file(url, dest_path)
        
        # Process into an hdf file
        Aux7.process_file(dest_path)
        print(f"Downloaded file to {dest_path}")
        
        pass
    
    

if __name__ == "__main__":
    # Aux4.main()
    
    # clear dir and files
    Aux5.main()
    
    # Aux6.main()
    # Aux7.main()
    pass
