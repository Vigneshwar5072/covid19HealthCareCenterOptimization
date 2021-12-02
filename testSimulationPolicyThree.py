
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
def solve_covid19_facility(c_coordinates, cv_19_demand, sick_demand,gamma, alpha, timePeriod,e_capacity,total_permanent_facility_cost):
    
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
            14: [(6, 9), 500],
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
        
        # Create a dictionary to capture the coordinates of a temporary facility and capacity of treating COVID-19 patients
                
    # Cost of driving 10 miles
    dcost = 5
    
    # Cost of building a temporary facility with capacity of 100 COVID-19
    tfcost = 500000
    
    existing = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15, 16, 17, 18, 19, 20, 21, 22, 23]
    e_coordinates= {1: (1, 2), 2: (2.5, 1), 3: (5, 1), 4: (6.5, 3.5), 5: (1, 5), 6: (3, 4), 7: (5, 4), 8: (6.5, 5.5), 9: (1, 8.5), 10: (1.5, 9.5), 11: (8.5, 6), 12: (5, 8), 13: (3, 9), 14: (6, 9),
                    15: (1.5, 1), 16: (3.5, 1.5), 17: (5.5, 2.5), 18: (1.5, 3.5), 19: (3.5, 2.5), 20: (4.5, 4.5), 21: (1.5, 6.5), 22: (3.5, 6.5), 23: (5.5, 6.5)}

    # Compute key parameters of MIP model formulation
    f_coordinates = {}
    for e in existing:
        f_coordinates[e] = e_coordinates[e]
    
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
    
    # Assign COVID-19 patients of county to facility
    x = m.addVars(cf, vtype=GRB.CONTINUOUS, name='Assign')
    

    m.setObjective(gp.quicksum(dcost*distance[c,f]*x[c,f] for c,f in cf), GRB.MINIMIZE)
    
    # Counties demand constraints

    demandConstrs = m.addConstrs((gp.quicksum(x[c,f] for f in facilities) == (cv_19_demand[c]+sick_demand[c]) for c in counties), 
                                     name='demandConstrs')        
    # Existing facilities capacity constraints
    existingCapConstrs = m.addConstrs((gp.quicksum(x[c,e]  for c in counties) <= e_capacity[e] for e in existing ), 
                                      name='existingCapConstrs')
    
    # Run optimization engine
    m.optimize()
    m.write('modeltest.lp')

    #####################################################
    #                    Output Reports
    #####################################################
        
    print(f"\n\n_____________Optimal costs______________________")
        
    patient_allocation_cost = 0
    for c,f in cf:
        if x[c,f].x > 1e-6:
            patient_allocation_cost += dcost*round(distance[c,f]*x[c,f].x)
            
    print(f"The total cost of allocating COVID-19 patients to healtcare facilities is ${patient_allocation_cost:,}")  
    total_permanent_facility_cost.append(patient_allocation_cost)
    # Build temporary facility at location
    
    

    print(f"\n_____________Available facilities______________________")

    print(e_capacity)
    f_demand = {}

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
        if(f<=23):
            e_capacity[f] = e_capacity[f] - f_demand[f]
            print(f"{e_capacity[f]} capacity is available at facility{f} ")
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

    return(e_capacity,total_permanent_facility_cost) 



# %%
#x = []
k = 9
capacities = pd.DataFrame()
p = [0.012]
import numpy as np
ydist = np.random.normal(1.12, 0.3, 30).tolist()


for t in range(0, 11):
    existing_capacity = []
    gamma = random.randint(10,50)
    alpha = random.randint(10,25)

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
        e_capacity = 0
        t_capacity = 0
        total_permanent_facility_cost = []

        # find the optimal solution of the base scenario
        [e_capacity,total_permanent_facility_cost] =  solve_covid19_facility(coordinates, cv_19_forecast,sick_forecast,gamma, alpha,t,e_capacity,total_permanent_facility_cost)
        for i in range(1,24):
            existing_capacity.append(e_capacity[i])
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


        [e_capacity,total_permanent_facility_cost] =  solve_covid19_facility(coordinates, cv_19_forecast,sick_forecast,gamma, alpha,t, e_capacity,total_permanent_facility_cost)
        for i in range(1,24):
            existing_capacity.append(e_capacity[i])
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

y1 = [7900, 7337, 6700, 6085, 5449, 4818, 4174, 3488, 2809, 2044, 1262, 533]

y2 = [7900, 7291, 6711, 6147, 5557, 4964, 4353, 3722, 3096, 2441, 1718, 1024]

y3 = [7900, 7295, 6691, 6092, 5472, 4853, 4236, 3592, 2985, 2351, 1715, 1052]

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


y1 = [ 4502460,  5250,  8470, 11840, 15715, 20170, 25500, 32030, 39790,
       53755, 71065]

y2 = [ 4502490,  5130,  7880, 11040, 14835, 18880, 23755, 29400, 35875,
       46400, 59830]

y3 = [ 4502535,  5220,  8210, 11620, 15375, 19515, 24645, 30175, 36510,
       46605, 59815]

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
