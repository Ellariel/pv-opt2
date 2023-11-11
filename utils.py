import json, time, requests, uuid, os, datetime, jsonpickle, pickle, shutil, glob
import pandas as pd#, numpy as np
from pandas import json_normalize
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import hashlib
import cv2
import base64
from faas_cache_dict.file_faas_cache_dict import FileBackedFaaSCache



class SyncronizedCache(FileBackedFaaSCache):
    def _self_to_disk(self):
        try:
            with self.file_lock:
                with open(self.file_path, 'rb') as f:
                    return self._do_pickle_file_load(f)
        except (EOFError, pickle.UnpicklingError):
            try:
                with self.file_lock:
                    with open(self.old_path, 'rb') as f:
                        return self._do_pickle_file_load(f)
            except (EOFError, FileNotFoundError):
                raise FileNotFoundError
        except FileNotFoundError:
            pass
        super()._self_to_disk()

def get_hash(obj):
    return hashlib.md5(obj.to_string().encode()).hexdigest()

def hash_dict(obj):
    return hashlib.md5(''.join([f"{k}{v}" for k, v in sorted(obj.items())]).encode()).hexdigest()

def get_encoded_img(image_path):
    img = cv2.imread(image_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (100, 100))
    _, buf = cv2.imencode('.png', img)
    image = base64.b64encode(buf).decode('ascii')
    return image

def make_timeserie_figure(data, file_name):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    data.resample('W').mean().plot(ax=ax)
    fig.text(0.2, 0.22, f"Yearly total: {data.sum()/1000:.1f} kWh", ha='left')
    #loc = plticker.MultipleLocator(base=30.0)
    #ax.xaxis.set_major_locator(loc)
    plt.ylabel('Power produced/consumed, Wh')
    plt.xticks(rotation=90)
    plt.tight_layout()    
    fig.savefig(file_name)
    plt.close(fig)
    
def make_metrics_figure(data, file_name):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    ax.scatter(data['solution_energy_costs'], data['genossenschaft_value'], s=45, alpha=0.5)
    
    for i, d in data.iterrows():
        ax.annotate(f"{i+1} ({d['genossenschaft_payback_perod']})", (d['solution_energy_costs'], d['genossenschaft_value']))
    
    fig.text(0.1, 0.0, f"Discounted payback perod in (), years. Alternative energy costs: {d['alternative_energy_costs']:.2f}, €/year", ha='left')
    plt.xlabel('Solution energy costs, €/year')
    plt.ylabel('Genossenschaft value, €/year')
    plt.tight_layout()    
    fig.savefig(file_name)
    plt.close(fig)

def save_pickle(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f) 
        
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
  
def from_storage(data_key, storage='./'):
        file_name = os.path.join(storage, data_key)+'.pickle'
        if os.path.exists(file_name):
            return load_pickle(file_name)        

def to_storage(data_key, data, storage='./'):
        file_name = os.path.join(storage, data_key)+'.pickle'
        save_pickle(data, file_name)  
    
def move_files(src_dir, dst_dir, mask='*.json'):
    file_list = glob.glob(os.path.join(src_dir, mask))
    for file_name in file_list:
        if os.path.isfile(file_name):
            shutil.move(file_name, os.path.join(dst_dir, os.path.basename(file_name)))

def is_empty(x):
    if isinstance(x, (list, dict, str, pd.DataFrame, pd.Series)):
        return len(x) == 0
    return pd.isna(x)

def to_json(obj):
        return jsonpickle.encode(obj)

def from_json(json_data):
        return jsonpickle.decode(json_data)

def _timestamp(x):
    return time.mktime(datetime.datetime.strptime(x, "%Y%m%d:%H%M").timetuple())

def _datetime(x):
    return datetime.datetime.strptime(x, "%Y%m%d:%H%M")

def _serie(x, datatype='hourly', name='production'):
    v = [(_datetime(i['time']), i['P']) for i in x[datatype]]
    v = pd.DataFrame(v).rename(columns={0:'timestamp', 1:name})
    v['timestamp'] = pd.DatetimeIndex(v['timestamp'])
    return v.set_index('timestamp')

def request_PVGIS(datatype='hourly', pvtechchoice='CIS', slope=0, azimuth=0, loss=14, lat=52.373, lon=9.738, startyear=2016, endyear=2016, timeout=3):
    # https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html
    # https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en
    # pvtechchoice	"crystSi", "CIS", "CdTe" and "Unknown".
    # aspect	(azimuth) angle of the (fixed) plane, 0=south, 90=west, -90=east. Not relevant for tracking planes.
    # {"P": {"description": "PV system power", "units": "W"}
    
    '''
    Inclination angle or slope
    This is the angle of the PV modules from the horizontal plane, for a fixed (non-tracking) mounting.
    For some applications the slope and orientation angles will already be known, 
    for instance if the PV modules are to be built into an existing roof. However, 
    if you have the possibility to choose the slope and/or azimuth (orientation), 
    this application can also calculate for you the optimal values for slope and orientation 
    (assuming fixed angles for the entire year).
    
    Orientation angle or azimuth
    The azimuth, or orientation, is the angle of the PV modules 
    relative to the direction due South. -90° is East, 0° is South and 90° is West.
    For some applications the slope and azimuth angles will already be known, 
    for instance if the PV modules are to be built into an existing roof. 
    However, if you have the possibility to choose the inclination and/or orientation, 
    this application can also calculate for you the optimal values for inclination 
    and orientation (assuming fixed angles for the entire year).
    '''   
    
    if datatype=='hourly':
      req = r"https://re.jrc.ec.europa.eu/api/seriescalc?outputformat=json&pvcalculation=1&peakpower=1&mountingplace=building"+\
            f"&lat={lat}&lon={lon}&pvtechchoice={pvtechchoice}&loss={loss}&angle={slope}&aspect={azimuth}"+\
            f"&raddatabase=PVGIS-SARAH&startyear={startyear}&endyear={endyear}"
    else:
      raise NotImplementedError(datatype)
    try:
        time.sleep(50)
        r = requests.get(req)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        time.sleep(timeout)
        r = requests.get(req)
        r.raise_for_status()
    return r.json()

class PVGIS:
    def __init__(self, storage='./'):
        self.inputs_storage = os.path.join(storage, 'pv_inputs.csv')
        self.outputs_storage = os.path.join(storage, 'pv_outputs')
        os.makedirs(self.outputs_storage, exist_ok=True)
        self.inputs = pd.read_csv(self.inputs_storage, sep=';') if os.path.exists(self.inputs_storage) else None
        
    def from_storage(self, data_key):
        with open(os.path.join(self.outputs_storage, data_key['outputs']+'.json'), 'r') as fp:
            return json.load(fp)
    
    def to_storage(self, data_key, data):
        with open(os.path.join(self.outputs_storage, data_key['outputs'].iloc[0]+'.json'), 'w') as fp:
            json.dump(data, fp)
        self.inputs = pd.concat([self.inputs, data_key], ignore_index=True)
        self.inputs.to_csv(self.inputs_storage, sep=';', index=False)    
    
    def get_nominal_pv(self, slope=0, azimuth=0, pvtech='CIS', loss=14, lat=52.373, lon=9.738, datayear=2016, datatype='hourly', request_if_none=True, save_if_none=True):
        # Watt per 1 kWp {"P": {"description": "PV system power", "units": "W"}
        filtered = self.inputs.query(
                f"`location.latitude` == {lat} & `location.longitude` == {lon} & "+\
                f"`data.type` == '{datatype}' & `data.year` == {datayear} & "+\
                f"`mounting_system.fixed.slope.value` == {slope} & "+\
                f"`mounting_system.fixed.azimuth.value` == {azimuth} & "+\
                f"`pv_module.technology` == '{pvtech}' & `pv_module.system_loss` == {loss}") if not is_empty(self.inputs) else None
        if is_empty(filtered) and request_if_none:
            print(f'requesting data from PVGIS, lat: {lat}, lon: {lon}, slope: {slope}, azimuth: {azimuth}')
            pv_raw_data = request_PVGIS(slope=slope,
                                 azimuth=azimuth,
                                 pvtechchoice=pvtech,
                                 loss=loss,
                                 lat=lat,
                                 lon=lon,
                                 startyear=datayear,
                                 endyear=datayear,
                                 datatype=datatype)
            if save_if_none:
                data_key = json_normalize(pv_raw_data['inputs'])
                data_key['outputs'] = uuid.uuid4().hex
                data_key['data.year'] = datayear       
                data_key['data.type'] = datatype
                data_key['data.timestamp'] = time.time()
                self.to_storage(data_key, pv_raw_data['outputs'])
            return _serie(pv_raw_data['outputs'], datatype=datatype)
        else:
            # filtered = filtered.sort_values(by='data.timestamp', ascending=False)
            #print(f'getting data from cache, lat: {lat}, lon: {lon}, angle: {angle}, aspect: {aspect}')
            return _serie(self.from_storage(filtered.iloc[0]), datatype=datatype)      
        
class Cache:
    def __init__(self, storage='./', use_pickle=True):
        #print('!!!!!!!!', storage)
        self.storage_file = os.path.join(storage, f"cache{'.pickle' if use_pickle else '.json'}")
        self.module = pickle if use_pickle else json
        self.storage = self.load()

    def load(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r' if '.json' in self.storage_file else 'rb') as fp:
                return self.module.load(fp)
        return {}
        
    def save(self):
        _storage = self.load()
        _storage.update(self.storage)
        with open(self.storage_file, 'w' if '.json' in self.storage_file else 'wb') as fp:
            self.module.dump(_storage, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
    def get_cached_solution(self, key, calc_if_none_method=None):
        if key in self.storage:
            return self.storage[key]
        elif calc_if_none_method:
            self.storage[key] = calc_if_none_method()
            return self.storage[key]
        