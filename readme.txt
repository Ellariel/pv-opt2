gui.exe is executable



# max: Genossenschaft profit = (SPO * BP) + (SEC * CP) – IC - roof renting costs   
# min: building energy cost = (SEC * CP) + (SPU * SP) + roof renting costs
# grid buying price BP: price at which the respective energy provider is buying energy per kwh
# grid selling price SP: price at which the respective energy provider is selling energy per kwh
# city price CP: price at which the Genossenschaft is selling energy to the city # city_selling_price
# grid buying price BP: price at which the respective energy provider is buying energy per kwh
# city price CP: price at which the Genossenschaft is selling energy to the city # city_selling_price

genossenschaft_value =
    solar_energy_overproduction / 1000 * grid_buying_price +
    solar_energy_consumption / 1000 * city_selling_price - 
    renting_costs
           
genossenschaft_payback_perod = installation_costs / discounted genossenschaft_value