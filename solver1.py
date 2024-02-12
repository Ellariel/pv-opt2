import time, os, copy, pickle, time, random, sys#, datetime, random, requests, uuid, json, 
import pandas as pd, numpy as np
import numbers
import hashlib
from tqdm import tqdm
#from tqdm.notebook import tqdm
#from pandas import json_normalize
import constraint, itertools, functools
from faas_cache_dict import FaaSCacheDict
from faas_cache_dict.file_faas_cache_dict import FileBackedFaaSCache
from uuid import uuid4
from collections import OrderedDict
from itertools import islice
from pprint import pprint
import gc

from utils import hash_dict, SyncronizedCache, Cache, is_empty, rescale
#from equip import Building, Equipment, Location, Battery, _calc_battery_costs

import utils
import equip
SOLVER_DEFAULT_CACHE_FILE = "solver_cache"
SOLVER_DEFAULT_CACHE_DIR = "./"
CACHED_CONGIF_KEYS = {
            'city_selling_price': True,
            'grid_selling_price': True,
            'grid_buying_price': True,
            'autonomy_period_days': True, 
            'installation_coef': True, 
            'miscellaneous_coef': True,
            'discount_rate': True, 
            'discount_horison': True, 
            'optimal_angle': True,
            'optimal_both': True,
            
            'max_investments': False,
            'max_kwattpeak': False,
            'max_payback_perod': False,
            'max_equipment_count': False,
            'min_equipment_count': False, 
            'shown_solutions_limit': False,
            'battery_capacity_uplimit': False,
            'use_ray': False,
            'ray_rate': False,
            'num_cpus': False,     
}

# IC – installation costs / kWp and installation costs / kWh
# solar panel overproduction SPO = max{0, (building solar production - building consumption)}
# solar energy consumption SEC = min{building solar production, building consumption}
# solar panel underproduction SPU = max{0, (building consumption - building solar production)}
# grid buying price BP: price at which the respective energy provider is buying energy
# grid selling price SP: price at which the respective energy provider is selling energy
# city price CP: price at which the Genossenschaft is selling energy to the city # city_solar_energy_price
# max: Genossenschaft profit = (SPO * BP) + (SEC * CP) – IC - roof renting costs
# min: building energy cost = (SEC * CP) + (SPU * SP) + roof renting costs

#@functools.cache
def calc_combinations(a): 
    _comb = set()
    for j in list(itertools.combinations_with_replacement(range(0, len(a)+1), len(a))) :
        c = []
        for i in j:
            if i > 0 and i not in c:
                c += [i]
        if len(c):
            _comb.add(tuple(sorted([a[i-1] for i in c])))
    return list(_comb)

@functools.cache
def calc_range(min_count=10, max_count=100):
    def _primes(_from, _to):
        out = list()
        sieve = [True] * (_to+1)
        for p in range(_from, _to+1):
            if sieve[p]:
                out.append(p)
                for i in range(p, _to+1, p):
                    sieve[i] = False
        return out
    _out = set(list(range(1, min_count+1)) + _primes(min_count+1, max_count))
    _out.add(min_count)
    _out.add(max_count)
    return [(i,) for i in _out]

#@functools.cache
def get_replaced(combination, items):
    return OrderedDict((i, items[i]) for i in combination if i in items) #copy.deepcopy(

#@functools.cache
def _hash_config(config):
    return hashlib.md5(''.join([f"{k}{v}" for k, v in sorted(config.items()) if k in CACHED_CONGIF_KEYS and CACHED_CONGIF_KEYS[k]]).encode()).hexdigest()


class ConstraintSolver:
    def __init__(self, building, components, pvgis=None, local_cache_dir=SOLVER_DEFAULT_CACHE_DIR, config={}, fixed_solution=None):
        self.key_name = SOLVER_DEFAULT_CACHE_FILE
        self.local_cache_dir = local_cache_dir
        #if self.local_cache_dir:
        #    os.makedirs(self.local_cache_dir, exist_ok=True)
        #    #self.cache = FileBackedFaaSCache.init(key_name=self.key_name, root_path=self.local_cache_dir)
        #    self.cache = SyncronizedCache.init(key_name=self.key_name, root_path=self.local_cache_dir)
        #else:
        #    self.cache = {} #FaaSCacheDict()
        self.cache = {}    
            
        #self.fixed_solution = fixed_solution
        self.components = components
        self.building = building
        self.config = config
        self.pvgis = pvgis
        
        
        self.filtered_locations = {k : copy.deepcopy(v) for k, v in self.components['location'].items() if v['building_uuid'] == self.building['uuid']}
        self.filtered_equipment = copy.deepcopy(self.components['equipment']) #{k : v for k, v in self.components['equipment'].items()}
        self.filtered_batteries = [v for k, v in copy.deepcopy(self.components['battery']).items() if k != 'NONE']  #{k : v for k, v in self.components['battery'].items()}
        if len(self.filtered_batteries):
            self.filtered_batteries = sorted(self.filtered_batteries, key=lambda x: x['battery_price_per_kWh'] if 'battery_price_per_kWh' in x else np.nan)
        
        #pprint(self.filtered_batteries)
        
        self.solutions = []
        self.initial_solutions = []
     

        
        #sys.exit()
        return
        
        
        
        
        self.filtered_locations = {k : v for k, v in self.components['location'].items() if v['building_uuid'] == building['uuid']}   
        self.location_combinations = calc_combinations(tuple(sorted(self.filtered_locations.keys())))
        self.equipment_combinations = [(k,) for k, v in self.components['equipment'].items()]
        self.battery_combinations = [(k,) for k, v in self.components['battery'].items()]
        print(f"location_combinations_count: {len(self.location_combinations)}")
        print(f"equipment_combinations: {self.equipment_combinations}")
        print(f"battery_combinations: {self.battery_combinations}")               
        max_equipment_count = int(np.max([equip.get_max_equipment_count_for_location(loc, eq, **self.config) 
                                        for i, loc in self.filtered_locations.items() 
                                            for j, eq in self.components['equipment'].items()]))
        print(f"estimated_max_equipment_count: {max_equipment_count}, config_max_equipment_count: {self.config['max_equipment_count']}")
        max_equipment_count = min(max_equipment_count, self.config['max_equipment_count'])
        self.equipment_count_range = calc_range(self.config['min_equipment_count'], max_equipment_count)
        self.battery_count_range = self.equipment_count_range.copy()
        print(f"equipment_count_range: {self.equipment_count_range}")
        print(f"battery_count_range: {self.battery_count_range}")
        if self.fixed_solution:
            print(f"fixed solution determined:")
            if len(self.fixed_solution['components']['locations'].items()):
                self.location_combinations = [tuple([k for k, v in self.fixed_solution['components']['locations'].items()])]
            if len(self.fixed_solution['components']['equipment'].items()):
                self.equipment_combinations = [(k, ) for k, v in self.fixed_solution['components']['equipment'].items()]
                self.equipment_count_range = [(v['pv_count'] ,) for k, v in self.fixed_solution['components']['equipment'].items()]
            if len(self.fixed_solution['components']['batteries'].items()):
                self.battery_combinations = [(k, ) for k, v in self.fixed_solution['components']['batteries'].items()]
                self.battery_count_range = [(v['battery_count'] ,) for k, v in self.fixed_solution['components']['batteries'].items()]
            print(f"location_combinations: {self.location_combinations}")
            print(f"equipment_combinations: {self.equipment_combinations} {self.equipment_count_range}")
            print(f"battery_combinations: {self.battery_combinations} {self.battery_count_range}")
                
        self.problem = constraint.Problem()

        self.problem.addVariable('A', self.location_combinations) # locations involved
        self.problem.addVariable('B', self.equipment_combinations) # equipment involved  
        self.problem.addVariable('C', self.equipment_count_range) # equipment count
             
        if self.config['autonomy_period_days'] == 0:
            self.problem.addVariable('D', [('NONE',)]) # none of batteries
            self.problem.addVariable('E', [(0,)]) # battery count 
        else:
            self.problem.addVariable('D', self.battery_combinations) # batteries involved
            self.problem.addVariable('E', [(0,)] + self.battery_count_range)
                
        self.problem.addConstraint(self.equipment_area_needed_constraint, "ABCDE")
        self.problem.addConstraint(self.battery_capacity_constraint, "ABCDE") 
        self.problem.addConstraint(self.max_investments_constraint, "ABCDE") 
        self.problem.addConstraint(self.max_kwattpeak_constraint, "ABCDE") 
        self.problem.addConstraint(self.max_payback_perod_constraint, "ABCDE")        



    #@functools.cache
    def equipment_area_needed_constraint(self, A, B, C, D, E):
        solution = self.get_cached((A, B, C, D, E), 'calc_equipment_allocation')
        #if not solution['allocated']:
        #    print((A, B, C, D, E))
        #print(f"equipment_area_needed_constraint: {solution['allocated']}")
        return solution['allocated']

    #@functools.cache
    def battery_capacity_constraint(self, A, B, C, D, E):            
        if self.config['autonomy_period_days'] == 0 :
            return True
        energy_storage_capacity = self.get_cached((A, B, C, D, E), 'get_energy_storage_capacity')
        energy_storage_needed = self.get_cached((A, B, C, D, E), 'get_energy_storage_needed')
        #if pd.isna(energy_storage_needed):
        #    return False
        #print(f"battery_capacity_constraint: {energy_storage_needed} <= {energy_storage_capacity} <= {energy_storage_needed * self.config['battery_uplimit']}")
        return energy_storage_capacity >= energy_storage_needed and\
                energy_storage_capacity <= energy_storage_needed * self.config['battery_capacity_uplimit']

    #@functools.cache
    def max_investments_constraint(self, A, B, C, D, E):
        installation_costs = self.get_cached((A, B, C, D, E), 'get_installation_costs')
        #print(f"max_investments_constraint: {installation_costs} <= {self.config['max_investments']}")
        return installation_costs <= self.config['max_investments']
    
    #@functools.cache
    def max_kwattpeak_constraint(self, A, B, C, D, E):
        solution = self.get_cached((A, B, C, D, E), 'calc_equipment_allocation')
        total_kwattpeak = 0
        for eq in solution['components']['equipment'].values():
            total_kwattpeak += eq['pv_watt_peak'] * eq['pv_count'] / 1000
        #print(f"max_kwattpeak_constraint: {total_kwattpeak} <= {self.config['max_kwattpeak']}")
        return total_kwattpeak <= self.config['max_kwattpeak']
    
    def max_payback_perod_constraint(self, A, B, C, D, E):
        payback_perod = self.get_cached((A, B, C, D, E), 'get_genossenschaft_payback_perod')
        payback_perod = int(payback_perod[1:]) + 1 if str(payback_perod)[0] == '>' else int(payback_perod)
        #print(f"max_payback_perod_constraint: {payback_perod} <= {self.config['max_payback_perod']}")
        return payback_perod <= self.config['max_payback_perod'] and\
                payback_perod >= self.config['min_payback_perod']
    
    #@functools.cache
    def build_solution(self, items): # A, B, C, D, E
        solution = equip.Solution.copy()
        solution['building'] = self.building#copy.deepcopy(self.building) #########
        solution['components']['locations'] = get_replaced(items[0], self.filtered_locations)#.copy() #########
        equipment = get_replaced(items[1], self.components['equipment'])#.copy() ###########
        #print(f"items[1]: {items[1]}")
        equipment_count = min(len(equipment), len(items[2]))
        #print(f"equipment: {equipment}")
        #print(f"items[2]: {items[2]}")
        #print(f"equipment_count: {equipment_count}")
        equipment = OrderedDict(itertools.islice(equipment.items(), equipment_count))
        #print(f"equipment: {OrderedDict(equipment)}")
        batteries = get_replaced(items[3], self.components['battery'])#.copy() ##########
        batteries_count = min(len(batteries), len(items[4]))
        #print(f"batteries: {batteries}")
        #print(f"batteries_count: {batteries_count}")
        batteries = OrderedDict(itertools.islice(batteries.items(), batteries_count))
        #print(f"{equipment}, {items[2][:equipment_count]}")
        for (k, v), c in zip(equipment.items(), items[2][:equipment_count]):
            v['pv_count'] = c
        for (k, v), c in zip(batteries.items(), items[4][:batteries_count]):
            v['battery_count'] = c
        solution['components']['equipment'] = equipment
        solution['components']['batteries'] = batteries 
        #print(f"equipment: {solution['components']['equipment']}")
        #print(f"batteries: {solution['components']['batteries']}")
        #print(solution['components']['equipment'])
        return solution

    @functools.cache
    def get_cached(self, items, method):
        calc_key = items + (method,) + (_hash_config(self.config), )
        #if items[2][0] == 107:
        #    print(calc_key)
        if calc_key in self.cache:
        #    if items[2][0] == 107:
        #        print(self.cache[calc_key][0])
            return self.cache[calc_key]#[0]
        if method == 'build_solution':
            solution = self.build_solution(items)
        elif method == 'calc_equipment_allocation':
            solution = copy.deepcopy(self.get_cached(items, 'build_solution'))
            solution = equip.calc_equipment_allocation(solution, **self.config)
        elif method == 'calc_equipment_allocation_and_production':
            solution = copy.deepcopy(self.get_cached(items, 'build_solution'))
            solution = equip.calc_equipment_allocation(solution, pvgis=self.pvgis, calc_production=True, **self.config)    
        elif method == 'get_energy_storage_capacity':
            solution = self.get_cached(items, 'build_solution')
            solution = equip.get_energy_storage_capacity(solution, **self.config)
        elif method == 'get_energy_storage_needed':
            solution = self.get_cached(items, 'calc_equipment_allocation_and_production')
            solution = equip.get_energy_storage_needed(solution, **self.config)    
        elif method == 'get_installation_costs':
            solution = self.get_cached(items, 'calc_equipment_allocation')
            solution = equip.get_installation_costs(solution, **self.config)               
        elif method == 'get_solution_energy_costs':
            solution = self.get_cached(items, 'calc_equipment_allocation_and_production')
            solution = equip.get_solution_energy_costs(solution, **self.config) 
        elif method == 'get_genossenschaft_value':
            solution = self.get_cached(items, 'calc_equipment_allocation_and_production')
            solution = equip.get_genossenschaft_value(solution, **self.config) 
        elif method == 'get_genossenschaft_payback_perod':
            solution = self.get_cached(items, 'calc_equipment_allocation_and_production')
            solution = equip.get_genossenschaft_payback_perod(solution, **self.config) 
        else:
            pass
        self.cache[calc_key] = copy.deepcopy(solution) #copy.deepcopy((solution,))
        del solution

      
        
        #if items[2][0] == 107:
        #    print(self.cache[calc_key][0])
        return self.cache[calc_key]#[0]  
    
    def get_initial_solutions(self):        
        self.initial_solutions = []
        for loc_idx, loc in tqdm(self.filtered_locations.items()):
            _list = []
            _loc = loc.copy()
            for eq_idx, eq in self.filtered_equipment.items():
                _eq = eq.copy()
                production, info = equip._get_nominal_production(_loc, _eq, 
                                                                 pvgis=self.pvgis, 
                                                                 lat=self.building['lat'], 
                                                                 lon=self.building['lon'],
                                                                 **self.config)
                if isinstance(production, pd.Series):
                    _eq['pv_count'] = 1
                    _eq.update(info)
                    production = production * (_eq['pv_watt_peak'] / 1000)
                    installation_costs = equip._get_installation_costs(_loc, _eq,
                                                                       **self.config)
                    _max_ratio = production.sum() / installation_costs
                    _list.append((_loc, _eq, production, installation_costs, _max_ratio))
            _list = sorted(_list, reverse=True, key=lambda x: x[4])
            if len(_list):
                self.initial_solutions.append((loc_idx, _list))
        if len(self.initial_solutions):
            self.initial_solutions = sorted(self.initial_solutions, reverse=True, key=lambda x: x[1][0][4])  
        return self.initial_solutions
          
    def get_solutions(self):
        self.solutions = []
        self.get_initial_solutions()
        
        city_selling_price = self.config.get('city_selling_price', 0.20)
        grid_selling_price = self.config.get('grid_selling_price', 0.30) 
        grid_buying_price = self.config.get('grid_buying_price', 0.08)
        
        for i in tqdm(list(range(self.config['shown_solutions_limit']))): # split equipment
            for j in tqdm(list(range(self.config['shown_solutions_limit']))): # split locations

                if j >= len(self.initial_solutions):
                    break

                _solutions = []
                investments_budget = self.config['max_investments']
                kwattpeak_budget = self.config['max_kwattpeak']
                for loc_idx, _list in copy.deepcopy(self.initial_solutions[j:]):
                    
                    if i >= len(_list):
                        break
                    
                    for loc, eq, production, installation_costs, _ in _list[i:]:
                            kwattpeak_installed = eq['pv_watt_peak'] / 1000
                            pv_count = min([equip._get_max_equipment_count_for_location(loc, eq, 
                                                                                            **self.config),
                                                np.floor(investments_budget / installation_costs),
                                                np.floor(kwattpeak_budget / kwattpeak_installed),
                                                self.config['max_equipment_count'],
                            ])
                            area_used_sqm = eq['pv_size_Wmm'] * (eq['pv_size_Hmm'] + eq['pv_dist']) * pv_count / (10 ** 6)

                            if (pv_count <= 0) or (loc['area_used_sqm'] + area_used_sqm > loc['area_sqm']):
                                break
                            
                            eq['pv_count'] = pv_count
                            loc['area_used_sqm'] += area_used_sqm
                            investments_budget -= installation_costs * pv_count
                            kwattpeak_budget -= kwattpeak_installed * pv_count
                            _solutions.append((loc, eq, production * pv_count, installation_costs * pv_count))
                        
                    if (investments_budget <= 0) or (kwattpeak_budget <= 0):
                            break
                        
                if len(_solutions):
                    components = []
                    total_production = 0    
                    total_renting_costs = 0
                    total_installation_costs = 0       
                    total_consumption = self.building['consumption'].fillna(0)    
                    for loc, eq, production, installation_costs in _solutions:
                        components.append((loc, eq))
                        total_production += production
                        total_installation_costs += installation_costs
                        total_renting_costs += equip._get_renting_costs(loc, eq, **self.config)
                        
                    total_production = utils.rescale(total_production)
                    equip.adjust_year(total_production, total_consumption)
                    
                    _building = copy.deepcopy(self.building)
                    _building['production'] = total_production
                    
                    solar_energy_consumption = equip._get_solar_energy_consumption(total_production, 
                                                                                        total_consumption,
                                                                                        **self.config)
                    solar_energy_underproduction = equip._get_solar_energy_underproduction(total_production, 
                                                                                                total_consumption,
                                                                                                **self.config)
                    solar_energy_overproduction = equip._get_solar_energy_overproduction(total_production, 
                                                                                            total_consumption,
                                                                                            **self.config)                
                    solution_energy_costs = solar_energy_consumption.sum() / 1000  * city_selling_price +\
                                            solar_energy_underproduction.sum() / 1000 * grid_selling_price  
                                                
                    alternative_energy_costs = total_consumption.sum() / 1000 * grid_selling_price
                    
                    genossenschaft_value = solar_energy_overproduction.sum() / 1000 * grid_buying_price +\
                                        solar_energy_consumption.sum() / 1000 * city_selling_price -\
                                        total_renting_costs
                    
                    solution = {
                        'building_uuid': self.building['uuid'],
                        'building': _building,
                        'equpment': components,
                        'config': self.config,
                        'batteries' : [],
                        'metrics': {
                            'total_production': total_production.sum(),
                            'total_consumption': total_consumption.sum(),
                            'solar_energy_consumption': solar_energy_consumption.sum(),
                            'solar_energy_underproduction': solar_energy_underproduction.sum(),
                            'solar_energy_overproduction': solar_energy_overproduction.sum(),
                            #'battery_capacity_limit': battery_capacity_limit,
                            'total_installation_costs': total_installation_costs,
                            'total_renting_costs': total_renting_costs,
                            'solution_energy_costs': solution_energy_costs,
                            'alternative_energy_costs': alternative_energy_costs,
                            'genossenschaft_value': genossenschaft_value,
                            'genossenschaft_payback_perod': equip._get_genossenschaft_payback_perod(genossenschaft_value, 
                                                                                                total_installation_costs),
                            'solution' : [{'location' : loc['uuid'],
                                        'equipment' : eq['uuid'],
                                        'equipment_count' : eq['pv_count']} for loc, eq in components]
                            }
                    }               
                    self.solutions.append(copy.deepcopy(solution))
                    
                    consumption_delta = total_consumption.sum() - solar_energy_consumption.sum()
                    battery_capacity_limits = np.minimum(utils.percentile(solar_energy_overproduction), 
                                                        utils.percentile(solar_energy_underproduction))
                    battery_capacity_limits = np.clip(battery_capacity_limits, 0, self.config['battery_capacity_uplimit'] * 1000)
                    
                    print(battery_capacity_limits)
                    if battery_capacity_limits.sum() > 0:    
                        for capacity in set(battery_capacity_limits):
                            if capacity <= 0:
                                continue
                            _solution = copy.deepcopy(solution)
                            battery_capacity_budget = capacity / 1000
                            investments_budget = self.config['max_investments'] - total_installation_costs
                        
                            batteries = []
                            for bt in self.filtered_batteries[:self.config['shown_solutions_limit']]:
                                _bt = bt.copy()
                                _bt['battery_count'] = 1
                                battery_installation_costs = equip._get_battery_installation_costs(_bt)
                                bt_count = min([np.floor(battery_capacity_budget / _bt['battery_effective_energy_kWh']),
                                                np.floor(investments_budget / battery_installation_costs)
                                                ])
                                _bt['battery_count'] = bt_count
                                if (bt_count > 0):
                                    batteries.append(_bt)
                                battery_capacity_budget -= _bt['battery_effective_energy_kWh'] * bt_count
                                investments_budget -= battery_installation_costs * bt_count
                                
                                if (battery_capacity_budget <= 0) or (investments_budget <= 0):
                                    break
                        
                            if len(batteries):
                                _solution['batteries'] = batteries
                                total_battery_capacity = 0
                                total_battery_costs = 0
                                for bt in batteries:
                                    total_battery_capacity += bt['battery_count'] * bt['battery_effective_energy_kWh'] * 1000
                                    total_battery_costs += equip._get_battery_installation_costs(bt)
                                
                                stored_overproduction = np.clip(solar_energy_overproduction, 0, total_battery_capacity).sum()
                                stored_overproduction = min([solar_energy_underproduction.sum(), stored_overproduction, consumption_delta])  
                                if stored_overproduction > 0:                
                                    _solar_energy_overproduction = solar_energy_overproduction.sum() - stored_overproduction
                                    _solar_energy_underproduction = solar_energy_underproduction.sum() - stored_overproduction
                                    _solution_energy_costs = (solar_energy_consumption.sum() + stored_overproduction) / 1000  * city_selling_price +\
                                                        _solar_energy_underproduction / 1000 * grid_selling_price  
                                    _genossenschaft_value = _solar_energy_overproduction / 1000 * grid_buying_price +\
                                                    (solar_energy_consumption.sum() + stored_overproduction) / 1000 * city_selling_price -\
                                                    total_renting_costs                
                                    _solution['metrics']['solar_energy_consumption'] = solar_energy_consumption.sum() + stored_overproduction      
                                    _solution['metrics']['solar_energy_underproduction'] = _solar_energy_underproduction
                                    _solution['metrics']['solar_energy_overproduction'] = _solar_energy_overproduction
                                    _solution['metrics']['total_installation_costs'] += total_battery_costs 
                                    _solution['metrics']['total_battery_capacity'] = total_battery_capacity
                                    _solution['metrics']['solution_energy_costs'] = _solution_energy_costs
                                    _solution['metrics']['genossenschaft_value'] = _genossenschaft_value
                                    _solution['metrics']['genossenschaft_payback_perod'] = equip._get_genossenschaft_payback_perod(_genossenschaft_value, 
                                                                                                                                solution['metrics']['total_installation_costs'])
                                    _solution['metrics']['solution'] += [str({'battery' : bt['uuid'],
                                                                        'battery_count' : bt['battery_count']}) for bt in batteries],
                                    self.solutions.append(_solution)

        return self.solutions
    

        if len(self.solutions) == 0 or recalc:
            for s in self.problem.getSolutions():
                items = (s['A'], s['B'], s['C'], s['D'], s['E'])
                solution = self.get_cached(items, 'calc_equipment_allocation_and_production')
                
                if len([k for k, v in solution['components']['locations'].items() if v['area_used_sqm'] == 0]):
                    continue #remove cloned solutions with zero area used locations
                
                solution_energy_costs = self.get_cached(items, 'get_solution_energy_costs')
                genossenschaft_value = self.get_cached(items, 'get_genossenschaft_value')
                
                #if not solution['allocated']:
                ##    print(items)
                #    print(solution)
                #    
                #_s = self.get_cached(items, 'calc_equipment_allocation')
                #if not solution['allocated']:
                #    print(items)
                #    print(_s)
                
                   

                solution.update({'solution_energy_costs': solution_energy_costs,
                                 'genossenschaft_value': genossenschaft_value})
                self.solutions.append(solution)
        return self.solutions