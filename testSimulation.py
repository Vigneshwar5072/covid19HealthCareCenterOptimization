
#%%
import matplotlib.pylab as plt
from itertools import product
from math import sqrt
from math import exp
import gurobipy as gp
from gurobipy import GRB
import random
from numpy.core.fromnumeric import cumsum
import pandas as pd
from scipy import stats
# tested with Gurobi v9.1.0 and Python 3.7.0
# %%
def compute_distance(loc1, loc2):
    
    # This function determines the Euclidean distance between a facility and a county centroid.
    
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return sqrt(dx*dx + dy*dy)


# %%
def solve_covid19_facility(c_coordinates, cv_19_demand, sick_demand,gamma, alpha, timePeriod,e_capacity, t_capacity,total_permanent_facility_cost,total_temporary_facility_cost):
    
    #####################################################
    #                    Data
    #####################################################
    counties = [*range(1,10)]
    
    # Indices for the facilities
    facilities = [*range(1,24)]
    
    # Indices for the counties
    if timePeriod== 0:
        
        # Create a dictionary to capture the coordinates of an existing facility and capacity of treating COVID-19 patients

        existing,  e_coordinates, e_capacity  = gp.multidict({
            1: [(1, 2), 500],
            2: [(2.5, 1), 500],
            3: [(5, 1), 500],
            4: [(6.5, 3.5), 500],
            5: [(1, 5), 500],
            6: [(3, 4), 500],
            7: [(5, 4), 500],
            8: [(6.5, 5.5), 500],
            9: [(1, 8.5), 500], 
            10: [(1.5, 9.5), 500],
            11: [(8.5, 6), 500],
            12: [(5, 8), 500],
            13: [(3, 9), 500],
            14: [(6, 9), 500]
        })
        
        # Create a dictionary to capture the coordinates of a temporary facility and capacity of treating COVID-19 patients
        
        temporary, t_coordinates, t_capacity  = gp.multidict({
            15: [(1.5, 1), 100],
            16: [(3.5, 1.5), 100],
            17: [(5.5, 2.5), 100],
            18: [(1.5, 3.5), 100],
            19: [(3.5, 2.5), 100],
            20: [(4.5, 4.5), 100],
            21: [(1.5, 6.5), 100],
            22: [(3.5, 6.5), 100],
            23: [(5.5, 6.5), 100]
        })
        
    # Cost of driving 10 miles
    dcost = 5
    
    # Cost of building a temporary facility with capacity of 100 COVID-19
    tfcost = 500000
    

    temporary = [15, 16, 17, 18, 19, 20, 21, 22, 23]
    t_coordinates={15: (1.5, 1), 16: (3.5, 1.5), 17: (5.5, 2.5), 18: (1.5, 3.5), 19: (3.5, 2.5), 20: (4.5, 4.5), 21: (1.5, 6.5), 22: (3.5, 6.5), 23: (5.5, 6.5)}
    existing = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    e_coordinates= {1: (1, 2), 2: (2.5, 1), 3: (5, 1), 4: (6.5, 3.5), 5: (1, 5), 6: (3, 4), 7: (5, 4), 8: (6.5, 5.5), 9: (1, 8.5), 10: (1.5, 9.5), 11: (8.5, 6), 12: (5, 8), 13: (3, 9), 14: (6, 9)}

    # Compute key parameters of MIP model formulation
    f_coordinates = {}
    for e in existing:
        f_coordinates[e] = e_coordinates[e]
    
    for t in temporary:
        f_coordinates[t] = t_coordinates[t]
    
    # Cartesian product of counties and facilities
    cf = []
    
    for c in counties:
        for f in facilities:
            tp = c,f
            cf.append(tp)
        
    # Compute distances between counties centroids and facility locations
    distance = {(c,f): compute_distance(c_coordinates[c], f_coordinates[f]) for c, f in cf}
    
    #####################################################
    #                    MIP Model Formulation
    #####################################################

    m = gp.Model('covid19_temporary_facility_location')
    # Build temporary facility
    y = m.addVars(temporary, vtype=GRB.BINARY, name='temporary')
    
    # Assign COVID-19 patients of county to facility
    x = m.addVars(cf, vtype=GRB.CONTINUOUS, name='Assign')
    
    # Add capacity to temporary facilities
    #z = m.addVars(temporary, vtype=GRB.CONTINUOUS, name='addCap' )
    
    # Objective function: Minimize total distance to drive to a COVID-19 facility
    
    # Big penalty for adding capacity at a temporary facility
    #bigM = 1e9

    m.setObjective(gp.quicksum(dcost*distance[c,f]*x[c,f] for c,f in cf) 
                   + tfcost*y.sum(), GRB.MINIMIZE)
    
    # Counties demand constraints

    demandConstrs = m.addConstrs((gp.quicksum(x[c,f] for f in facilities) == (cv_19_demand[c]+sick_demand[c]) for c in counties), 
                                     name='demandConstrs')        
    # Existing facilities capacity constraints
    existingCapConstrs = m.addConstrs((gp.quicksum(x[c,e]  for c in counties) <= e_capacity[e]+gamma+alpha for e in existing ), 
                                      name='existingCapConstrs')
    
    # temporary facilities capacity constraints
    temporaryCapConstrs = m.addConstrs((gp.quicksum(x[c,t]  for c in counties)
                                        <= t_capacity[t]*y[t] for t in temporary ),
                                       name='temporaryCapConstrs')
    # Run optimization engine
    m.optimize()
    m.write('modeltest.lp')

    #####################################################
    #                    Output Reports
    #####################################################
    
    # Total cost of building temporary facility locations
    temporary_facility_cost = 0
    
    print(f"\n\n_____________Optimal costs______________________")
    for t in temporary:
        if (y[t].x > 0.5):
            temporary_facility_cost += tfcost*round(y[t].x)
        
    patient_allocation_cost = 0
    for c,f in cf:
        if x[c,f].x > 1e-6:
            patient_allocation_cost += dcost*round(distance[c,f]*x[c,f].x)
            
    print(f"The total cost of building COVID-19 temporary healhtcare facilities is ${temporary_facility_cost:,}") 
    print(f"The total cost of allocating COVID-19 patients to healtcare facilities is ${patient_allocation_cost:,}")  
    total_temporary_facility_cost.append(temporary_facility_cost)
    total_permanent_facility_cost.append(patient_allocation_cost)
    # Build temporary facility at location
    
    

    print(f"\n_____________Available facilities______________________")

    print(e_capacity)
    print(t_capacity)
    #temp_e_capacity = e_capacity
    #temp_t_capacity = t_capacity


    #myList = e_capacity.items()
    #myList = sorted(myList) 
    #x, y = zip(*myList) 
    #plt.plot(x, y)
    #plt.show()
    
    print(f"\n_____________Plan for temporary facilities______________________")
    for t in temporary:
        if (y[t].x > 0.5):
            print(f"Build a temporary facility at location {t}")
            
    # Extra capacity at temporary facilities
    #print(f"\n_____________Plan to increase Capacity at temporary Facilities______________________")
    #for t in temporary:
    #    if (z[t].x > 1e-6):
    #        print(f"Increase  temporary facility capacity at location {t} by {round(z[t].x)} beds")

    # Demand satisfied at each facility
    f_demand = {}
    
            #print(t_capacity[f])


    print(f"\n_____________Allocation of county patients to COVID-19 healthcare facility______________________")
    for f in facilities:
        temp = 0
        for c in counties:
            #print(round(x[c,f].x))

            allocation = round(x[c,f].x)
            if allocation > 0:
                print(f"{allocation} patients from county {c} are treated at facility {f} ")
            temp += allocation
        f_demand[f] = temp
        print(f"{temp} is the total number of patients that are treated at facility {f}. ")
        if(f<=14):
            e_capacity[f] = e_capacity[f] - f_demand[f]
            print(f"{e_capacity[f]} capacity is available at facility{f} ")
        else:
            t_capacity[f] = t_capacity[f] - f_demand[f]
            print(f"{t_capacity[f]} capacity is available at facility{f} ")
        print(f_demand[f])
        print(f"\n________________________________________________________________________________")
        

        #print(e_capacity, t_capacity)

        

    # Test total demand = total demand satisfied by facilities
    total_demand = 0
    
    for c in counties:
        total_demand += (cv_19_demand[c]+ sick_demand[c])

    demand_satisfied = 0
    for f in facilities:
        demand_satisfied += f_demand[f]
        
    print(f"\n_____________Test demand = supply______________________")
    print(f"Total demand is: {total_demand:,} patients")
    print(f"Total demand satisfied is: {demand_satisfied:,} beds")
    print(f"\n_____________The NEW ASSIGNMENT FOR THE TIME-PERIOD: {timePeriod:,} BEGINS HERE___________")

    return(e_capacity,t_capacity,total_permanent_facility_cost,total_temporary_facility_cost) 



# %%
#x = []
k = 9
capacities = pd.DataFrame()
p = [0.012]
import numpy as np
ydist = np.random.normal(1.15, 0.3, 30).tolist()


for t in range(0, 11):
    existing_capacity = []

    if t == 0:
        


        counties, time_period, coordinates, cv_19_forecast  = gp.multidict({
            1: [t,(1, 1.5), round(ydist[t]*exp(k*p[0]*t+1))],
            2: [t, (3, 1), round(ydist[t]*exp(k*p[0]*t+1))],
            3: [t, (5.5, 1.5), round(ydist[t]*exp(k*p[0]*t+1))],
            4: [t, (1, 4.5 ), round(ydist[t]*exp(k*p[0]*t+1))],
            5: [t, (3, 3.5), round(ydist[t]*exp(k*p[0]*t+1))],
            6: [t, (5.5, 4.5),round(ydist[t]*exp(k*p[0]*t+1))],
            7: [t, (1, 8), round(ydist[t]*exp(k*p[0]*t+1))],
            8: [t, (3, 6), round(ydist[t]*exp(k*p[0]*t+1))],
            9: [t,(4.5, 8),round(ydist[t]*exp(k*p[0]*t+1))]   
        })


        counties, time_period, coordinates, sick_forecast  = gp.multidict({
            1: [t,(2, 1.5), random.randint(50,75)],
            2: [t, (3, 2), random.randint(50,75)],
            3: [t, (5.7, 1.5), random.randint(50,75)],
            4: [t, (2.3, 4.5 ), random.randint(50,75)],
            5: [t, (3.3, 3.5), random.randint(50,75)],
            6: [t, (5.6, 4.5),random.randint(50,75)],
            7: [t, (8, 8),random.randint(50,75)],
            8: [t, (3.5, 6), random.randint(50,75)],
            9: [t,(4.5, 8), random.randint(50,75)]   
        })
        gamma = 120
        alpha = 80
        e_capacity = 0
        t_capacity = 0
        total_temporary_facility_cost = []
        total_permanent_facility_cost = []

        # find the optimal solution of the base scenario
        [e_capacity, t_capacity,total_permanent_facility_cost,total_temporary_facility_cost] =  solve_covid19_facility(coordinates, cv_19_forecast,sick_forecast,gamma, alpha,t,e_capacity, t_capacity,total_permanent_facility_cost,total_temporary_facility_cost)
        for i in range(1,24):
            if(i <= 14):
                existing_capacity.append(e_capacity[i])
            else:
                existing_capacity.append(t_capacity[i])
        capacities[t] = existing_capacity

    else:
        counties, time_period, coordinates, cv_19_forecast  = gp.multidict({
            1: [t,(1, 1.5), round(ydist[t]*exp(k*p[0]*t+1))],
            2: [t, (3, 1), round(ydist[t]*exp(k*p[0]*t+1))],
            3: [t, (5.5, 1.5), round(ydist[t]*exp(k*p[0]*t+1))],
            4: [t, (1, 4.5 ), round(ydist[t]*exp(k*p[0]*t+1))],
            5: [t, (3, 3.5), round(ydist[t]*exp(k*p[0]*t+1))],
            6: [t, (5.5, 4.5), round(ydist[t]*exp(k*p[0]*t+1))],
            7: [t, (1, 8), round(ydist[t]*exp(k*p[0]*t+1))],
            8: [t, (3, 6), round(ydist[t]*exp(k*p[0]*t+1))],
            9: [t,(4.5, 8), round(ydist[t]*exp(k*p[0]*t+1))]   
        })


        counties, time_period, coordinates, sick_forecast  = gp.multidict({
            1: [t,(2, 1.5), random.randint(50,75)],
            2: [t, (3, 2), random.randint(50,75)],
            3: [t, (5.7, 1.5), random.randint(50,75)],
            4: [t, (2.3, 4.5 ), random.randint(50,75)],
            5: [t, (3.3, 3.5), random.randint(50,75)],
            6: [t, (5.6, 4.5), random.randint(50,75)],
            7: [t, (8, 8), random.randint(50,75)],
            8: [t, (3.5, 6), random.randint(50,75)],
            9: [t,(4.5, 8), random.randint(50,75)]   
        })


        [e_capacity, t_capacity,total_permanent_facility_cost,total_temporary_facility_cost] =  solve_covid19_facility(coordinates, cv_19_forecast,sick_forecast,gamma, alpha,t, e_capacity, t_capacity,total_permanent_facility_cost,total_temporary_facility_cost)
        for i in range(1,24):
            if(i <= 14):
                existing_capacity.append(e_capacity[i])
            else:
                existing_capacity.append(t_capacity[i])
        capacities[t] = existing_capacity




# %%
sum_over_capacities = []
sum_over_capacities.append(7900)

for column in capacities:
    sum_over_capacities.append(capacities[column].sum())

x = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
y = np.array(sum_over_capacities)
plt.plot(x,y)


# %%

x = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

y1 = [7900, 7305, 6728, 6116, 5506, 4900, 4276, 3600, 2886, 2187, 1424,634]

y2 = [7900, 7328, 6720, 6120, 5485, 4848, 4246, 3621, 2968, 2273, 1606, 952]

y3 = [7900, 7312, 6718, 6142, 5553, 4951, 4296, 3724, 3190, 2585, 1969, 1352]

lines = [y1,y2,y3]
colors  = ['r','g','b']
labels  = ['0.024','0.018','0.012']

# fig1 = plt.figure()
for i,c,l in zip(lines,colors,labels):  
    plt.plot(x,i,c,label='l')
    plt.legend(labels)    
plt.xlabel("Time Periods")
plt.ylabel("Total Available Capacities")
plt.show()
# %%

y0 = np.cumsum(total_permanent_facility_cost)

x = np.array([1,2,3,4,5,6,7,8,9,10,11])
y1 = [ 3220,  6310,  9630, 13065, 17110, 21760, 26610, 32910, 39130,
       47575, 61395]

y2 = [ 2980,  6240,  9415, 12630, 16885, 21300, 26415, 32345, 38070,
       46490, 56815]

y3 = [ 3025,  5955,  9070, 12530, 16365, 20275, 25180, 30055, 35325,
       40995, 50180]

lines = [y1,y2,y3]
colors  = ['r','g','b']
labels  = ['0.024','0.018','0.012']

# fig1 = plt.figure()
for i,c,l in zip(lines,colors,labels):  
    plt.plot(x,i,c,label='l')
    plt.legend(labels)    
plt.xlabel("Time Periods")
plt.ylabel("Total Cost")
plt.show()



# %%
