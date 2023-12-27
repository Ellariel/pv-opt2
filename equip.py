import pandas as pd
import numpy as np
from uuid import uuid4
import os
import pickle
import time
from datetime import datetime
import utils
import copy
from collections import OrderedDict
import constraint, itertools, functools
#from pvgis import PVGIS

#pv_gis = PVGIS()#utils.PVGIS()

# https://www.youtube.com/watch?v=OPNBWaBZvjc&ab_channel=%D0%94%D0%B5%D1%80%D0%B5%D0%B2%D0%B5%D0%BD%D1%81%D0%BA%D0%B8%D0%B9%D1%84%D0%BE%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84
# https://www.youtube.com/watch?v=Oriqr7K9kAc&ab_channel=%D0%94%D0%B5%D1%80%D0%B5%D0%B2%D0%B5%D0%BD%D1%81%D0%BA%D0%B8%D0%B9%D1%84%D0%BE%D1%82%D0%BE%D0%B3%D1%80%D0%B0%D1%84
# потребление - 1 кВт 8 часов в сутки
# аккумулятор - 12 В х 60 Ah = 720 Втч * 0,7 = 504 Втч - полчас
Battery = dict(
        uuid = None,
        battery_count = 1,
        type = 'LiFePO4',
        battery_price = 100.0,
        #battery_capacity_Ah = 100,
        battery_energy_kWh = 4.8,
        #battery_voltage = 48,
        battery_discharge_factor = 0.7,
        battery_price_per_kWh = 9.738,
)
Equipment = dict(
        uuid = None,
        pv_count = 1,
        type = 'CIS',
        pv_price = 100.0,
        pv_size_Wmm = 2176,
        pv_size_Hmm = 1098,
        #pv_size_mm = (2176, 1098),
        pv_efficiency = 18,
        pv_watt_peak = 500,
        pv_price_per_Wp = 0.90,
        pv_system_loss = 14,
        #pv_voltage = 48,
)
Location = dict(
        uuid = None,
        building_uuid = 1,
        slope = 0,
        azimuth = 0,
        size_Wm = 10,
        size_Hm = 10,
        area_sqm = 100,
        price_per_sqm = 1,
        area_used_sqm = 0,
        #pv_area_used = 0,
        #lat = 52.373,
        #lon = 9.738,
        #_equipment = []
)
Building = dict(
        uuid = None, 
        name = 'noname',
        address = 'Hannover',
        lat = 52.373,
        lon = 9.738,
        production_profile_key = None,
        production = None,
        consumption_profile_key = 'mook',
        consumption = None,
)
'''
Production_profile = dict(
        uuid = None,
        timestamp = 1692989661.8293242,
        year = 2022,
        building_uuid = 1,
        stored = '5c75deb8d33045cd86bb3ee9b7e98c25',
        production = None,
)
Consumption_profile = dict(
        uuid = None,
        timestamp = 1692989661.8293242,
        year = 2022,
        building_uuid = 1,
        stored = '5c75deb8d33045cd86bb3ee9b7e98c25',
        consumption = None,
)
'''
Solution = dict(
        uuid = None,
        timestamp = 1692989661.8293242,
        building = None,
        #selected = 0,
        metrics = None,
        solutions_profile_key = '5c75deb8d33045cd86bb3ee9b7e98c25',
        components = dict(
            locations = OrderedDict(),
            equipment = OrderedDict(),
            batteries = OrderedDict(),
        ),
)

def get_soultion_metrics(solution, **kwargs):
    return {
        'building_uuid': solution['building']['uuid'],
        'total_production': solution['building']['production'].sum(),
        'total_consumption': solution['building']['consumption'].sum(),
        'solar_energy_consumption': get_solar_energy_consumption(solution, **kwargs),
        'solar_energy_underproduction': get_solar_energy_underproduction(solution, **kwargs),
        'solar_energy_overproduction': get_solar_energy_overproduction(solution, **kwargs),
        'solution_energy_costs': get_solution_energy_costs(solution, **kwargs),
        'alternative_energy_costs': get_alternative_energy_costs(solution, **kwargs),
        'genossenschaft_value': get_genossenschaft_value(solution, **kwargs),
        'genossenschaft_payback_perod': get_genossenschaft_payback_perod(solution, **kwargs),
        'locations_involved': {k : v['area_used_sqm'] for k, v in solution['components']['locations'].items()}, 
        'renting_costs': get_renting_costs(solution, **kwargs),
        'pv_equipment_used': {k : v['pv_count'] for k, v in solution['components']['equipment'].items()},
        'pv_costs': get_pv_costs(solution, **kwargs),
        'baterry_units_used': {k : v['battery_count'] for k, v in solution['components']['batteries'].items()},
        'batteries_costs': get_battery_costs(solution, **kwargs),
        'total_installation_costs': get_installation_costs(solution, **kwargs),
    }

# min: building energy cost = (SEC * CP) + (SPU * SP) + roof renting costs
# grid buying price BP: price at which the respective energy provider is buying energy per kwh
# grid selling price SP: price at which the respective energy provider is selling energy per kwh
# city price CP: price at which the Genossenschaft is selling energy to the city # city_selling_price
def get_solution_energy_costs(solution, city_selling_price=0.20, grid_selling_price=0.30, **kwargs):
    return get_solar_energy_consumption(solution, **kwargs) / 1000  * city_selling_price +\
           get_solar_energy_underproduction(solution, **kwargs) / 1000 * grid_selling_price +\
           get_renting_costs(solution, **kwargs)
           
def get_alternative_energy_costs(solution, grid_selling_price=0.30, **kwargs):
    return solution['building']['consumption'].sum() / 1000 * grid_selling_price
           
# max: Genossenschaft profit = (SPO * BP) + (SEC * CP) – IC - roof renting costs   
# grid buying price BP: price at which the respective energy provider is buying energy per kwh
# city price CP: price at which the Genossenschaft is selling energy to the city # city_selling_price 
def get_genossenschaft_value(solution, grid_buying_price=0.08, city_selling_price=0.20, **kwargs):
    return get_solar_energy_overproduction(solution, **kwargs) / 1000 * grid_buying_price +\
           get_solar_energy_consumption(solution, **kwargs) / 1000 * city_selling_price -\
           get_renting_costs(solution, **kwargs)
           
def get_genossenschaft_payback_perod(solution, discount_rate=0.03, discount_horison=5, **kwargs):
    genossenschaft_value = get_genossenschaft_value(solution, **kwargs)
    installation_costs = get_installation_costs(solution, **kwargs)
    payback_period = int(max(1, np.ceil(installation_costs / genossenschaft_value))) + discount_horison
    npv = 0
    for i in range(1, payback_period):
        npv += genossenschaft_value / (1 + discount_rate) ** i
        if npv + 1 > installation_costs:
            return i
    return f'>{payback_period}'

# solar energy consumption SEC = min{building solar production, building consumption}
def get_solar_energy_consumption(solution, **kwargs):
    #return min(solution['building']['production'].sum(), solution['building']['consumption'].sum())
    result = np.minimum(solution['building']['consumption'].resample('D').sum().fillna(0),
                        solution['building']['production'].resample('D').sum().fillna(0))
    return result.sum()

# solar panel underproduction SPU = max{0, (building consumption - building solar production)}
def get_solar_energy_underproduction(solution, **kwargs):
    #return max(0, solution['building']['consumption'].sum() - solution['building']['production'].sum())
    diff = solution['building']['consumption'].resample('D').sum().fillna(0) -\
           solution['building']['production'].resample('D').sum().fillna(0)
    diff = np.where(diff < 0, 0, diff)
    return diff.sum()
       
# solar panel overproduction SPO = max{0, (building solar production - building consumption)}
def get_solar_energy_overproduction(solution, **kwargs):
    #return max(0, solution['building']['production'].sum() - solution['building']['consumption'].sum())
    diff = solution['building']['production'].resample('D').sum().fillna(0) -\
           solution['building']['consumption'].resample('D').sum().fillna(0)
    diff = np.where(diff < 0, 0, diff)
    return diff.sum()

def get_energy_storage_needed(solution, autonomy_period_days=-1, **kwargs):
    if autonomy_period_days == 0:
        return 0
    peak_daily_consumption = solution['building']['consumption'].resample('D').sum().max()
    if autonomy_period_days < 0:
        #if not isinstance(solution['building']['production'], pd.Series):
        #    return np.nan
        avg_daily_production = solution['building']['production'].resample('D').sum().mean()
        autonomy_period_days = np.ceil(peak_daily_consumption / avg_daily_production)
    return peak_daily_consumption * autonomy_period_days

def get_installation_costs(solution, installation_coef=1.0, miscellaneous_coef=0.5, **kwargs):
    return get_pv_costs(solution, **kwargs) * (installation_coef + miscellaneous_coef + 1) + get_battery_costs(solution, **kwargs)

def get_renting_costs(solution, **kwargs):
    _costs = 0
    for k, v in solution['components']['locations'].items():
        _costs += v['area_used_sqm'] * v['price_per_sqm']
    _costs += len(solution['components']['locations']) / 100 # add little costs to reduce number of locations involved
    return _costs 
    
def get_pv_costs(solution, **kwargs): 
    _costs = 0
    for k, v in solution['components']['equipment'].items():
        _costs += v['pv_watt_peak'] * v['pv_price_per_Wp'] * v['pv_count']
    return _costs 
    
def get_battery_costs(solution, **kwargs): 
    _costs = 0
    for k, v in solution['components']['batteries'].items():
        _costs += v['battery_energy_kWh'] * v['battery_price_per_kWh'] * v['battery_count']
    return _costs

def get_energy_storage_capacity(solution, **kwargs): 
    _storage_capacity = 0
    for k, v in solution['components']['batteries'].items():
        _storage_capacity += v['battery_energy_kWh'] * v['battery_discharge_factor'] * v['battery_count'] * 1000 
    return _storage_capacity   

def get_max_equipment_count_for_location(loc, eq, **kwargs):
    return np.floor((loc['area_sqm'] - loc['area_used_sqm']) * 10 ** 6 / (eq['pv_size_Wmm'] * eq['pv_size_Hmm']))

def calc_equipment_allocation(solution, pvgis=None, calc_production=False, **kwargs):
    #print(solution)
    #solution = copy.deepcopy(solution)
    def get_nominal_production(eq, loc): # Watt per 1 kWp
        if pvgis:
            return pvgis.get_production_timeserie(slope=loc['slope'],
                                              azimuth=loc['azimuth'], 
                                              pvtech=eq['type'], 
                                              system_loss=eq['pv_system_loss'], 
                                              lat=solution['building']['lat'], 
                                              lon=solution['building']['lon'])    
    total_production = None
    allocation_area_used = {}
    allocation_equipment = {}
    equipment = copy.deepcopy([v for k, v in solution['components']['equipment'].items()])#copy.deepcopy()#.copy()########
    locations = copy.deepcopy([v for k, v in solution['components']['locations'].items()])#)#.copy()########copy.deepcopy(
    while len(locations) and len(equipment):
        loc, eq = locations[0], equipment[0]     
        max_equipment_count = get_max_equipment_count_for_location(loc, eq, **kwargs)
        if eq['pv_count'] < max_equipment_count:
                eq_count = eq['pv_count']
                area_used = eq['pv_size_Wmm'] * eq['pv_size_Hmm'] * eq_count / (10 ** 6)
                loc['area_used_sqm'] += area_used
                #allocation_equipment.append(eq.copy())
                del equipment[0]
        elif eq['pv_count'] > max_equipment_count:
                eq_count = max_equipment_count
                eq['pv_count'] -= eq_count
                area_used = eq['pv_size_Wmm'] * eq['pv_size_Hmm'] * eq_count / (10 ** 6)
                #loc['area_used_sqm'] = area_used#
                #allocation_equipment.append(eq.copy())
                del locations[0]
        else:
                eq_count = eq['pv_count']
                area_used = eq['pv_size_Wmm'] * eq['pv_size_Hmm'] * eq_count / (10 ** 6)
                #loc['area_used_sqm'] = area_used#
                #allocation_equipment.append(eq.copy())
                del equipment[0]
                del locations[0]
        allocation_area_used[loc['uuid']] = allocation_area_used[loc['uuid']] + area_used if loc['uuid'] in allocation_area_used else area_used
        allocation_equipment[eq['uuid']] = allocation_equipment[eq['uuid']] + eq_count if eq['uuid'] in allocation_equipment else eq_count
        
        if calc_production:
            production = get_nominal_production(eq, loc) * (eq['pv_watt_peak'] / 1000) * eq_count    
            total_production = total_production + production if isinstance(total_production, pd.Series) else production
    solution['building']['production'] = total_production
    
    #solution['components']['locations'] = copy.deepcopy(solution['components']['locations'])
    
    #print(allocation_equipment)
    #if len(equipment) > 0 or len(allocation_equipment) == 0:
    #    return False
    for k, v in solution['components']['locations'].items():
        if k in allocation_area_used:
            v['area_used_sqm'] = allocation_area_used[k]
        else:
            v['area_used_sqm'] = 0
            
    for k, v in solution['components']['equipment'].items():
        if k in allocation_equipment:
            v['pv_count'] = allocation_equipment[k]
        else:
            v['pv_count'] = 0            
            
    #
    #return True
    solution['allocated'] = len(equipment) == 0 and len(allocation_equipment) > 0
    #    return False
    #print(solution)
    #if calc_production:
    #    solution.update({'after_calc_production' : True})
    check_solution_year(solution)
    return solution

def from_storage(key, storage):
    file_name = os.path.join(storage, key) + '.csv'
    if os.path.exists(file_name):
        return pd.read_csv(file_name, parse_dates=True, header=None, index_col=0).iloc[:,0]
      
def to_storage(key, data, storage):
    file_name = os.path.join(storage, key) + '.csv'
    data.to_csv(file_name, header=None, index=True)

def load_production_profile(building, production_storage='./'):
    building['production'] = from_storage(building['production_profile_key'], storage=production_storage).rename('production')
    
def load_consumption_profile(building, consumption_storage='./'):
    building['consumption'] = from_storage(building['consumption_profile_key'], storage=consumption_storage).rename('consumption')
    
def save_production_profile(building, production_storage='./'):
    building['production_profile'] = uuid4().hex
    to_storage(building['production_profile_key'], building['production'], storage=production_storage)
    
def save_consumption_profile(building, consumption_storage='./'):
    building['consumption_profile'] = uuid4().hex
    to_storage(building['consumption_profile_key'], building['consumption'], storage=consumption_storage)
    
def check_solution_year(solution):
    if isinstance(solution['building']['production'], pd.Series) and isinstance(solution['building']['consumption'], pd.Series):
        production_year = solution['building']['production'].index[0].year
        consumtion_year = solution['building']['consumption'].index[0].year
        diff = consumtion_year - production_year
        if diff != 0:
            solution['building']['production'].index = solution['building']['production'].index + pd.offsets.DateOffset(years=diff)
        return diff
