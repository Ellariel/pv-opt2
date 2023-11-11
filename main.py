import time, os, glob#, pickle, json, uuid
import pandas as pd, numpy as np
from humanfriendly import format_timespan

#from ppretty import ppretty
from ast import literal_eval
from operator import itemgetter
from collections import OrderedDict
import numbers
import copy
from uuid import uuid4
import gc
import asyncio
import functools
import time
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor


import equip
#from equip import Building, Equipment, Location, Battery
from utils import save_pickle, load_pickle, move_files, from_storage, to_storage, is_empty
from solver import ConstraintSolver#, total_building_energy_costs, total_installation_costs, genossenschaft_profit, _update_building

#from pvutils import PVGIS
from pvgis import PVGIS


base_dir = './'
config = {
            'city_selling_price': 0.20,
            'grid_selling_price': 0.30,
            'grid_buying_price': 0.08,
            'autonomy_period_days': 0.0, # 0.05~1h
            'installation_coef': 1.0, 
            'miscellaneous_coef': 0.5,
            'discount_rate': 0.03, 
            'discount_horison': 5,   
            'max_investments': 250000,
            'max_payback_perod': 30,
            'min_payback_perod': 5,
            'max_equipment_count': 150,
            'min_equipment_count': 10, 
            'shown_solutions_limit': 10,
            'battery_capacity_uplimit': 2.0,
            'use_ray': 0,
            'ray_rate': 1.0,
            'num_cpus': 4,    
        }

components = {}
buildings = {}
data_tables = {'location_data': None,
              'equipment_data': None, 
              'battery_data': None,
              'building_data': None,
#              'consumption_data': None,
#              'production_data': None,
              'solution_data': None,
}  

consumption_dir = os.path.join(base_dir, 'consumption')
production_dir = os.path.join(base_dir, 'production')
solution_dir = os.path.join(base_dir, 'solution')

os.makedirs(consumption_dir, exist_ok=True)
os.makedirs(production_dir, exist_ok=True)
os.makedirs(solution_dir, exist_ok=True)

log = ''
def _print(value, clear=False):
    global log
    if clear:
        log = ''
    log += '\n' + value
    print(value)    
    
    

    
def update_config(new_config):
    global config
    config.update(new_config)
    
def init_components(base_dir, upload_dir=None):
    global components, buildings, data_tables, config
    
    _print(f'base_dir: {base_dir}', clear=True)
    
    if not upload_dir:
        data_tables = load_pickle(os.path.join(base_dir, 'components.pickle'))       
    else:
        move_files(upload_dir['consumption_file'], consumption_dir)
        move_files(upload_dir['production_file'], production_dir)
        excel_file = glob.glob(os.path.join(upload_dir['excel_file'], '*.xlsx'))
        if len(excel_file) and os.path.exists(excel_file[0]):
            excel_file = excel_file[0] 
            for k in data_tables.keys():  
                try:    
                    print(f'attempt to load {k} from {excel_file}')      
                    df = pd.read_excel(excel_file, sheet_name=k.split('_')[0], converters={#'size_WxHm': literal_eval,
                                                                                           #'pv_size_mm': literal_eval,
                                                                                           'uuid': str,
                                                                                           'building_uuid': str,
                    })
                    df.index = range(1, len(df)+1)
                    #print(df.info())
                    #print(df)
                    data_tables[k] = df
                    data_tables[k].index = data_tables[k].uuid.copy()
                    
                    if 'battery_price' in data_tables[k].columns:
                        data_tables[k]['battery_price_per_kWh'] = (data_tables[k]['battery_price'] / data_tables[k]['battery_energy_kWh']).fillna(0)
                        #data_tables[k].loc[pd.isna(data_tables[k]['battery_energy_kWh']), ['battery_price_per_kWh']] = 0
                        #print(data_tables[k]['battery_price_per_kWh'])
                        
                    if 'pv_price' in data_tables[k].columns:
                        data_tables[k]['pv_price_per_Wp'] = (data_tables[k]['pv_price'] / data_tables[k]['pv_watt_peak']).fillna(0)
                    
                    print(f'loaded data length: {len(df)}')
                except Exception as e:
                    print(f'error: {e}')
            os.remove(excel_file)
            #del data_tables['consumption_data']
            #del data_tables['production_data']
            save_pickle(data_tables, os.path.join(base_dir, 'components.pickle'))
   
    components['location'] = data_tables['location_data'].to_dict(orient='index')
    components['equipment'] = data_tables['equipment_data'].to_dict(orient='index')
    components['battery'] = data_tables['battery_data'].to_dict(orient='index')
    #buildings = data_tables['building_data'].iloc[0:1].to_dict(orient='index')
    buildings = data_tables['building_data'].to_dict(orient='index')
    try:   
        for k, v in buildings.items():
            if v['production_profile_key'] != None and not isinstance(v['production'], pd.Series):
                    equip.load_production_profile(v, production_storage=production_dir)
            if v['consumption_profile_key'] != None and not isinstance(v['consumption'], pd.Series):
                    equip.load_consumption_profile(v, consumption_storage=consumption_dir) 
    except Exception as e:
        print(e)
    
    #print(data_tables['building_data'])
    #print(buildings) 

    _print('data loading:')  
    _print(f"    buildings: {len(data_tables['building_data'])}, locations: {len(components['location'])},")
    _print(f"    equipment: {len(components['equipment'])}, batteries: {len(components['battery'])},")
    _print(f"    stored solutions: {len(data_tables['solution_data'])}") 
    
def calculate(base_dir):   
    global components, buildings, data_tables, config
    
    results = {}
    _print(f"config: {config}")
    start_time = time.time()
    
    if not config['use_ray']:
        pvgis = PVGIS(verbose=True)
        for uuid, b in buildings.items():
            results[uuid] = []
            print(f"solving building: {uuid}")
            try:
                _start_time = time.time()
                _config = config.copy()
                _config['autonomy_period_days'] = 0
                solver = ConstraintSolver(b, components, pvgis, config=_config)
                results[uuid] = solver.get_solutions()
                del solver
                print(f"{uuid} solving time: {format_timespan(time.time() - _start_time)}")
                _print(f"possible solutions for building {uuid}: {len(results[uuid])}")  
                
                if config['autonomy_period_days'] != 0:
                    _start_time = time.time()
                    print(f"attempt to select batteries for building {uuid}")
                    _config['autonomy_period_days'] = config['autonomy_period_days']
                    modified_solutions = {}
                    results[uuid] = sorted(results[uuid], reverse=False, key=lambda x: x['solution_energy_costs'])
                    for s in results[uuid][:config['shown_solutions_limit']]:#[:1]:
                        if uuid not in modified_solutions:
                            modified_solutions[uuid] = []
                        s['components']['batteries'] = OrderedDict()
                        solver = ConstraintSolver(b, components, pvgis, config=_config, fixed_solution=s)
                        modified_solutions[uuid] += solver.get_solutions() 
                        _print(f"possible solutions with batteries for building {uuid}: {len(modified_solutions[uuid])}")       
                        del solver          
                        #collected = gc.collect()
                        #if collected:
                        #    print("Garbage collector: collected %d objects" % collected)     
                    results[uuid] = modified_solutions[uuid]
                    print(f"{uuid} solving time: {format_timespan(time.time() - _start_time)}")
                    
            except Exception as e:
                print(f"some errors occured while calculating building {uuid}: {str(e)}")
    else:
        import ray
        ray.init(ignore_reinit_error=True)#, num_cpus=4) # log_to_driver=False

        @ray.remote
        def solve(building, components, config):
            solutions = []
            try:
                print(f"solving building: {building['uuid']}")
                pvgis = PVGIS(verbose=True)
                _start_time = time.time()
                solver = ConstraintSolver(building, components, pvgis, config=config)
                solutions = solver.get_solutions()  
                del solver 
                print(f"{building['uuid']} solving time: {format_timespan(time.time() - _start_time)}")
            except Exception as e:
                print(f"some errors occured while calculating building {building['uuid']}: {str(e)}")
            return solutions 

        if config['autonomy_period_days'] == 0:
            split_key = 'equipment'
        else:
            if len(components['equipment']) >= len(components['battery']):
                split_key = 'equipment'
            else:
                split_key = 'battery'        
        split_list = list(components[split_key].keys())
        split_count = len(split_list)
        chunk_size = int(config['ray_rate'] * split_count)
        if chunk_size < 1:
            chunk_size = 1
        print(f"splitting by: {split_key}, chunk size: {chunk_size}")
        
        ray_instances = {}
        for chunk in range(0, split_count, chunk_size):
            print(split_list[chunk:chunk + chunk_size])
            _components = components.copy()#copy.deepcopy(components)
            _components[split_key] = {k : v for k, v in components[split_key].items() if k in split_list[chunk:chunk + chunk_size]}
            for uuid, b in buildings.items():
                if uuid not in ray_instances:
                    ray_instances[uuid] = []
                ray_instances[uuid] += [solve.remote(b, _components, config)] #.copy() copy.deepcopy(b)
        for uuid, b in buildings.items():
            results[uuid] = []
            for s in ray.get(ray_instances[uuid]):
                results[uuid] += s
            print(f"possible solutions for building {uuid}: {len(results[uuid])}")
        ray.shutdown()
    
    _print(f'total solving time: {format_timespan(time.time() - start_time)}')
    for uuid, b in buildings.items():
        if uuid in results:
            if len(results[uuid]):
                 _print(f"top solutions for building {uuid}:")
                 
                 #ray_results[uuid] = sorted(ray_results[uuid], reverse=True, key=lambda x: x['genossenschaft_value'])
                 results[uuid] = sorted(results[uuid], reverse=False, key=lambda x: x['solution_energy_costs'])
                 
                 solutions = results[uuid][:config['shown_solutions_limit']]
                 for s in solutions:
                     s.update({'metrics' : equip.get_soultion_metrics(s), 'config' : config})
                     _print(f"{s['metrics']}")
                 data_tables['solution_data'] = save_solutions(data_tables['solution_data'], solutions, storage=solution_dir)
    save_pickle(data_tables, os.path.join(base_dir, 'components.pickle'))

    
def save_solutions(solution_data, solutions, timestamp=None, uuid=None, storage='./'):
        data_key = equip.Solution.copy()
        data_key['uuid'] = uuid if uuid != None else uuid4().hex
        data_key['timestamp'] = timestamp if timestamp != None else time.time()
        data_key['solutions_profile_key'] = uuid4().hex
        data_key['config'] = str(solutions[0]['config'])
        del data_key['components']
        del data_key['building']
        del data_key['metrics']
        data = pd.DataFrame([{**i['metrics'], **data_key} for i in solutions])
        data['n'] = data.index + 1
        for c in data.columns:
            if data[c].dtype == 'object':
                data[c] = data[c].apply(lambda x: str(x))     
        to_storage(data_key['solutions_profile_key'], solutions, storage=storage)
        return pd.concat([solution_data, data], ignore_index=True)
    
'''
def save_solutions(solution_data, solutions, timestamp=None, uuid=None, storage='./'):
        data_key = equip.Solution.copy()
        data_key['uuid'] = uuid if uuid != None else uuid4().hex
        data_key['timestamp'] = timestamp if timestamp != None else time.time()
        data_key['building_uuid'] = solutions[0]['building']['uuid']
        data_key['metrics'] = str(solutions[0]['metrics'])
        data_key['config'] = str(solutions[0]['config'])
        data_key['solutions_profile_key'] = uuid4().hex
        #data_key.update(solutions[0]['metrics'])
        del data_key['components']
        del data_key['building']
        to_storage(data_key['solutions_profile_key'], solutions, storage=storage)
        return pd.concat([solution_data, pd.DataFrame.from_dict({0: data_key}, orient='index')], ignore_index=True)
'''
    
def load_solutions(solution_data, uuid=None, timestamp=None, storage='./'):
        query = f"`uuid` == '{uuid}'"
        query += f" & `timestamp` == {timestamp}" if timestamp != None else ''
        filtered = solution_data.query(query)
        if not is_empty(filtered):
            filtered = filtered.sort_values(by=['timestamp'], ascending=False)
            return filtered.iloc[0], from_storage(filtered.iloc[0]['solutions_profile_key'], storage=storage)