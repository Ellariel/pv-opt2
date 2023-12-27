import os
import time
import datetime
import requests
import hashlib
import pandas as pd
import functools
from faas_cache_dict import FaaSCacheDict
from faas_cache_dict.file_faas_cache_dict import FileBackedFaaSCache

PVGIS_ENDPOINT = "https://re.jrc.ec.europa.eu/"
PVGIS_DEFAULT_CACHE_FILE = "pvgis_cache"
PVGIS_DEFAULT_CACHE_DIR = "./"

def _get_hash(s):
    return hashlib.md5(s.encode()).hexdigest()

def _format_datetime(s):
    return datetime.datetime.strptime(s, "%Y%m%d:%H%M")

def _request_PVGIS(datatype='hourly', pvtechchoice='CIS', slope=0, azimuth=0, mountingplace='building', 
                   system_loss=14, lat=52.373, lon=9.738, startyear=2016, endyear=2016, 
                   retry_timeout_sec=3, max_retries=3, optimal_angle=False, optimal_both=False):
    # https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html
    # https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en
    # INPUT:
    # peakpower in kW, 1 by default
    # pvtechchoice: "crystSi", "CIS", "CdTe" and "Unknown"
    # mountingplace: "free" or "building"
    # aspect (-180° -- 180°): the orientation angle / azimuth, or orientation, is the angle of the PV modules relative to the direction due South. -90° is East, 0° is South and 90° is West.
    # angle (0° -- 90°): the inclination angle or slope of the PV modules from the horizontal plane, for a fixed (non-tracking) mounting.
    # OUTPUT:
    # P_W per 1 kW peak -> {"P": {"description": "PV system power", "units": "W"} ...
    # optimalinclination	Calculate the optimum inclination angle. Value of 1 for "yes". All other values (or no value) mean "no"
    # optimalangles Calculate the optimum inclination AND orientation angles. Value of 1 for "yes". All other values (or no value) mean "no".
    
    if optimal_angle:
        print('optimal_angle', optimal_angle)
    
    if datatype=='hourly':
      req = f"{PVGIS_ENDPOINT}api/seriescalc?outputformat=json&pvcalculation=1&peakpower=1&mountingplace={mountingplace}"+\
            f"&lat={lat}&lon={lon}&pvtechchoice={pvtechchoice}&loss={system_loss}&angle={slope}&aspect={azimuth}"+\
            f"&raddatabase=PVGIS-SARAH&startyear={startyear}&endyear={endyear}"
    else:
      raise NotImplementedError(datatype)
    
    try:
        r = requests.get(req)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(str(e))
        for i in range(0, max_retries):
            print(f"another try: {i+1}")
            try:
                time.sleep(retry_timeout_sec)
                r = requests.get(req)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                print(str(e))      

class PVGIS(object):
    def __init__(self, local_cache_dir=PVGIS_DEFAULT_CACHE_DIR, verbose=False):
        
        self.verbose = verbose
        self.cache_root_dir = local_cache_dir
        self.key_name = PVGIS_DEFAULT_CACHE_FILE
        if local_cache_dir:
            os.makedirs(self.cache_root_dir, exist_ok=True)
            self.cache = FileBackedFaaSCache.init(key_name=self.key_name, root_path=self.cache_root_dir)
        else:
            self.cache = FaaSCacheDict()
            
    @functools.cache
    def get_radiation_data(self, slope=0, azimuth=0, pvtech='CIS', 
                           lat=52.373, lon=9.738, system_loss=14, datayear=2016, datatype='hourly',
                           optimal_angle=False, optimal_both=False):  

        api_parameters = ', '.join([f"{k}:{v}" for k, v in sorted(locals().items(), key=lambda item: item[0]) if k not in ['self']])     
        request_key = _get_hash(api_parameters)
        if request_key not in self.cache:
            if self.verbose:
                print(f'requesting data from PVGIS, {api_parameters}')
            self.cache[request_key] = _request_PVGIS(slope=slope,
                                 azimuth=azimuth,
                                 pvtechchoice=pvtech,
                                 system_loss=system_loss,
                                 lat=lat, lon=lon,
                                 startyear=datayear,
                                 endyear=datayear,
                                 datatype=datatype,
                                 optimal_angle=optimal_angle,
                                 optimal_both=optimal_both)
        else:
            if self.verbose:
                print(f'getting cached PVGIS data, {api_parameters}')
        return self.cache[request_key]

    @functools.cache
    def get_production_timeserie(self, slope=0, azimuth=0, pvtech='CIS', 
                                 lat=52.373, lon=9.738, system_loss=14, 
                                 datayear=2016, datatype='hourly', name='production', 
                                 optimal_angle=False, optimal_both=False):
        # makes float64 timeserie with DatetimeIndex, Name: production, Length: 8784, dtype: float64
        data_json = self.get_radiation_data(slope=slope, azimuth=azimuth, 
                                            pvtech=pvtech, lat=lat, lon=lon,
                                            system_loss=system_loss, 
                                            datayear=datayear, datatype=datatype,
                                            optimal_angle=optimal_angle, optimal_both=optimal_both)
        production = pd.Series({_format_datetime(i['time']) : i['P'] for i in data_json['outputs'][datatype]}, name=name)
        info = {}
        return production, info
        
if __name__ == "__main__":
    print(PVGIS(local_cache_dir=None, verbose=True).get_production_timeserie())
