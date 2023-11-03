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
import ray
import gc

import equip
#from equip import Building, Equipment, Location, Battery
from utils import save_pickle, load_pickle, move_files, make_figure, from_storage, to_storage, is_empty
from solver import ConstraintSolver#, total_building_energy_costs, total_installation_costs, genossenschaft_profit, _update_building

from pvutils import PVGIS
#from pvgis import PVGIS


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
            'max_equipment_count': 10,
            'min_equipment_count': 10, 
            'shown_solutions_limit': 5,
            'battery_capacity_uplimit': 2.0,
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
'''
_rename = {'A': 'location_uuid',
           'B': 'equipment_uuid',
           'C': 'equipment_count',
           'D': 'battery_uuid',
           'E': 'battery_count',
           }

def _ren(s):
    global components
    def _match(k, v):
        if 'uuid' in k:
            k = k.split('_')[0]
            if isinstance(v, (int, str)):
                return components[k][v]['uuid']
            elif isinstance(v, tuple):
                return [components[k][i]['uuid'] for i in v]
        return v
    _r = {}
    for k, v in s.items():
        if k in _rename:   
            k = _rename[k]
            #v = _match(k, v)
            _r.update({k: v})
    return _r
'''        
'''
def _get_soultion_metrics(building):
    metrics = {
        'building': building.uuid,
        'total_production': building.production['production'].sum(),
        'total_consumption': building.consumption['consumption'].sum(),
        'total_solar_energy_consumption': building.total_solar_energy_consumption,
        'total_solar_energy_underproduction': building.total_solar_energy_underproduction,
        'total_solar_energy_overproduction': building.total_solar_energy_overproduction,
        'total_building_energy_costs': total_building_energy_costs(building, **config),
        'locations_involved': len(building._locations),
        'total_renting_costs': building.total_renting_costs,
        'equipment_units_used': sum([eq['pv_count'] for loc in building._locations for eq in loc['_equipment']]),
        'total_equipment_costs': building.total_equipment_costs,
        'baterry_units_used': sum([bt['battery_count'] for bt in building._battery]),
        'total_battery_costs': building.total_battery_costs,
        'total_installation_costs': total_installation_costs(building, **config),
        'genossenschaft_profit': genossenschaft_profit(building, **config),
    }
    #_print(str(metrics))
    return metrics
'''    
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
    #try:   
    for k, v in buildings.items():
        if v['production_profile_key'] != None and not isinstance(v['production'], pd.Series):
                equip.load_production_profile(v, production_storage=production_dir)
        if v['consumption_profile_key'] != None and not isinstance(v['consumption'], pd.Series):
                equip.load_consumption_profile(v, consumption_storage=consumption_dir) 
    #except Exception as e:
    #    print(e)
    
    #print(data_tables['building_data'])
    #print(buildings) 

    _print('data loading:')  
    _print(f"    buildings: {len(data_tables['building_data'])}, locations: {len(components['location'])},")
    _print(f"    equipment: {len(components['equipment'])}, batteries: {len(components['battery'])},")
    _print(f"    stored solutions: {len(data_tables['solution_data'])}") 
    
    #print(data_tables['production_data'])
    '''
    for idx, item in data_tables['building_data'].iterrows():
        #try:        
            b = equip.Building(**item.to_dict())
            b.load_production(data_tables['production_data'], storage=production_dir)
            b.load_consumption(data_tables['consumption_data'], storage=consumption_dir)
            for idx, item in data_tables['location_data'][data_tables['location_data']['building_uuid'] == b.uuid].iterrows():
                loc = equip.Location.copy()
                loc.update(item.to_dict())
                b._locations.append(loc)
            b.updated(update_production=False)
            #print(b.production['production'].sum())
            building_objects.append(b)
        #except Exception as e:
        #    print(f'error loading building {b.uuid}: {str(e)}')
    '''
'''
def dict_to_building(building_dict):
        global components, buildings, data_tables, config
        b = equip.Building(**building_dict)
        b.load_production(data_tables['production_data'], storage=production_dir)
        b.load_consumption(data_tables['consumption_data'], storage=consumption_dir)
        for idx, item in data_tables['location_data'][data_tables['location_data']['building_uuid'] == b.uuid].iterrows():
                loc = equip.Location.copy()
                loc.update(item.to_dict())
                b._locations.append(loc)
        return b
'''
def calculate(base_dir):   
    global components, buildings, data_tables, config

    #components['location'] = OrderedDict((k, {**v, 'area_sqm' : v['size_sqm'], 'area_used_sqm':0}) for i, (k, v) in enumerate(components['location'].items()))
    #components['equipment'] = OrderedDict((k, {**v, 'pv_system_loss' : v['pv_loss'], 'pv_count': 1}) for i, (k, v) in enumerate(components['equipment'].items()))
    
    '''
    loc = OrderedDict((k, {**v, 'area_sqm' : v['size_sqm'], 'area_used_sqm':0}) for i, (k, v) in enumerate(components['location'].items()) if i < 3)
    eq = OrderedDict((k, {**v, 'pv_system_loss' : v['pv_loss'], 'pv_count': 100}) for i, (k, v) in enumerate(components['equipment'].items()) if i < 2)
    bt = OrderedDict()
    print(loc)
    print()
    print(eq)
    print()
    solution = equip.Solution.copy()
    solution['building'] = equip.Building.copy()
    solution['building']['consumption_profile_key'] = 'G1'
    equip.load_consumption_profile(solution['building'], consumption_dir)
    solution['components']['locations'] = loc
    solution['components']['equipment'] = eq
    solution['components']['batteries'] = bt
    print(solution)
    
    equip.calc_equipment_allocation(solution, pvgis, calc_production=True)
    
    print(solution)
    print(equip.get_soultion_metrics(solution))
    '''
    ray.init(ignore_reinit_error=True)#, num_cpus=4) # log_to_driver=False
    
    #pvgis_id = ray.put(pvgis)
    
    @ray.remote
    def solve(building, components, config):
        solutions = []
        #try:
        print(f"solving building: {building['uuid']}")
        pvgis = PVGIS(verbose=True)
        start_time = time.time()
        solver = ConstraintSolver(building, components, pvgis, config=config)
        solutions = solver.get_solutions()   
        print(f"{building['uuid']} solving time: {format_timespan(time.time() - start_time)}")
        #except Exception as e:
        #print(f"error calculating building {building['uuid']}: {str(e)}")
        return solutions 
    
    _print(f"config: {config}")
    
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
    start_time = time.time()
    
    ray_instances = {}
    for chunk in range(0, split_count, chunk_size):
        print(split_list[chunk:chunk + chunk_size])
        _components = components.copy()#copy.deepcopy(components)
        _components[split_key] = {k : v for k, v in components[split_key].items() if k in split_list[chunk:chunk + chunk_size]}
        for uuid, b in buildings.items():
            if uuid not in ray_instances:
                ray_instances[uuid] = []
            ray_instances[uuid] += [solve.remote(b, _components, config)] #.copy() copy.deepcopy(b)
    
    ray_results = {}
    for uuid, b in buildings.items():
        ray_results[uuid] = []
        for s in ray.get(ray_instances[uuid]):
            ray_results[uuid] += s
        print(f"possible solutions for building {uuid}: {len(ray_results[uuid])}")
        #if len(solutions) > 1:
            #solutions = itemgetter(*np.argsort(costs))(solutions) 
            #costs = itemgetter(*np.argsort(costs))(costs)
            ###solutions = solutions[:config['shown_solutions_limit']]
            #costs = costs[:config['top_limit']]
            #print(solutions)
        #    pass
        

    _print(f'total solving time: {format_timespan(time.time() - start_time)}')
    ray.shutdown()
    #print(ray_results)
    
    for uuid, b in buildings.items():
        if uuid in ray_results:
            if len(ray_results[uuid]):
                 _print(f"top solutions for building {uuid}:")
                 
                 #ray_results[uuid] = sorted(ray_results[uuid], reverse=True, key=lambda x: x['genossenschaft_value'])
                 ray_results[uuid] = sorted(ray_results[uuid], reverse=False, key=lambda x: x['solution_energy_costs'])
                 
                 solutions = ray_results[uuid][:config['shown_solutions_limit']]
                 for s in solutions:
                     s.update({'metrics' : equip.get_soultion_metrics(s), 'config' : config})
                     _print(f"{s['metrics']}")
                 data_tables['solution_data'] = save_solutions(data_tables['solution_data'], solutions, storage=solution_dir)
    
    save_pickle(data_tables, os.path.join(base_dir, 'components.pickle'))
                 
            #for solution in ray_results[uuid]:#[:3]:
                #metrics = equip.get_soultion_metrics(solution)
                #metrics.update({'allocated' : solution['allocated']})
                #solution.update(metrics)
                
                #if not solution['allocated']:
                #solutions = solutions[:config['shown_solutions_limit']]
                
                ##del solution['building']['production']
                #del solution['building']['consumption']
                #_print(f"{metrics}")
        #break
    '''
    for _, b in data_tables['building_data'].iterrows(): #.iloc[:2]
        if b['uuid'] in ray_results:
            building = dict_to_building(b.to_dict())
            s = ray_results[b['uuid']][0][0]
            _update_building(building, components, s, use_roof_sq=config['use_roof_sq'])
            s = _ren(s)
            s.update(get_soultion_metrics(building))
            _print(f"solution for building {b['uuid']}: {s}")
            data_tables['solution_data'] = ConstraintSolver(building, components, config=config).save_solution(data_tables['solution_data'], building, s, storage=solution_dir)
            if config['save_opt_production']:
                data_tables['production_data'] = building.save_production(data_tables['production_data'], storage=production_dir)

    save_pickle(data_tables, os.path.join(base_dir, 'components.pickle'))  
    
    Solution = dict(
        uuid = None,
        timestamp = 1692989661.8293242,
        building = None,
        #selected = 0,
        solution_profile_key = '5c75deb8d33045cd86bb3ee9b7e98c25',
        components = dict(
            locations = OrderedDict(),
            equipment = OrderedDict(),
            batteries = OrderedDict(),
        ),
    
    '''
    
def save_solutions(solution_data, solutions, timestamp=None, uuid=None, storage='./'):
        data_key = equip.Solution.copy()
        data_key['uuid'] = uuid if uuid != None else uuid4().hex
        data_key['timestamp'] = timestamp if timestamp != None else time.time()
        data_key['building_uuid'] = solutions[0]['building']['uuid']
        data_key['metrics'] = str(solutions[0]['metrics'])
        data_key['config'] = str(solutions[0]['config'])
        data_key['solutions_profile_key'] = uuid4().hex
        del data_key['components']
        del data_key['building']
        to_storage(data_key['solutions_profile_key'], solutions, storage=storage)
        return pd.concat([solution_data, pd.DataFrame.from_dict({0: data_key}, orient='index')], ignore_index=True)
    
def load_solutions(self, solution_data, building_uuid, timestamp=None, uuid=None, storage='./'):
        query = f"`building_uuid` == {building_uuid}"
        query += f" & `uuid` == {uuid}" if uuid != None else ''
        query += f" & `timestamp` == {timestamp}" if timestamp != None else ''
        filtered = solution_data.query(query)
        if not is_empty(filtered):
            filtered = filtered.sort_values(by=['timestamp'], ascending=False)
            return filtered.iloc[0], from_storage(filtered.iloc[0]['solutions_profile_key'], storage=storage)