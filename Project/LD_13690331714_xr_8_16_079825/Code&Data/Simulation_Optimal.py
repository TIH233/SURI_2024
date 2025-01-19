# coding: utf-8

# # Simulation for Optimal Model under (r,Q) policy

# ## Import

import simpy
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# For progress tracking
from tqdm import tqdm

# To track the CPU usage and test potential problem (mostly in parallel computing)
import logging
import psutil
import time

# For setting random seeds to ensure reproducibility
import random

# for csv file writing and time tracking
import csv
from datetime import datetime

from scipy.spatial.distance import euclidean
from collections import deque
import os
from multiprocessing import Pool
import multiprocessing
from threading import Lock

# To ensure solutions will not be evaluated for efficiency
from scipy.spatial.distance import euclidean

CACHE_LOCK = Lock()

# ## Helper Methods

# log in configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

SIM_DAY = 5000
SCALE = 1


def normalize_02n(array, n):
    min_val = np.min(array)
    max_val = np.max(array)
    return n * (array - min_val) / (max_val - min_val)


class InventorySystem:
    def __init__(self, env, r, Q, mu, L, m, K, h, c, w, p, demand_params):
        self.env = env
        self.r = r
        self.Q = Q
        self.mu = mu
        self.L = L
        self.m = m
        self.K = K
        self.h = h
        self.c = c
        self.w = w
        self.p = p
        self.demand_params = demand_params

        self.inventory_level = r
        self.inventory_position = r
        self.outdated_items = 0
        self.lost_sales = 0

        self.shelf_life = {}
        self.outstanding_orders = []
        self.reorder_count = 0
        self.waste_count = 0
        self.total_cost = 0

        self.weekly_reorder_frequency = 0
        self.weekly_waste_frequency = 0
        self.cumulative_lost_sales = 0
        self.cumulative_outdated_items = 0
        self.total_inventory_level = 0
        self.average_inventory_level = 0

        self.demands = []  # New list to store demands
        self.gen_demands = np.load('simulation_demand.npy')
        self.gen_demands = normalize_02n(self.gen_demands, 2)  # [0, 2]

        self.env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(1)  # Time step
            self.check_order_arrivals()
            self.remove_perished_items()
            self.demand_occurrence()
            self.review_inventory()
            self.update_costs()
            self.update_statistics()

    def check_order_arrivals(self):
        arrived_orders = [order for order in self.outstanding_orders if order['arrival_time'] <= self.env.now]
        for order in arrived_orders:
            self.inventory_level += order['quantity']
            self.inventory_position += order['quantity']
            self.outstanding_orders.remove(order)
            for i in range(int(order['quantity'])):
                self.shelf_life[self.env.now] = self.env.now + self.m

    def remove_perished_items(self):
        perished = sum(1 for time, expiry in list(self.shelf_life.items()) if expiry <= self.env.now)
        if perished > 0:
            self.inventory_level -= perished
            self.inventory_position -= perished
            self.outdated_items += perished
            self.waste_count += 1
            for time in list(self.shelf_life.keys()):
                if self.shelf_life[time] <= self.env.now:
                    del self.shelf_life[time]

    def demand_occurrence(self):
        demand = self.generate_demand()
        self.demands.append(demand)  # Store each generated demand

        # demand = self.demands[self.env.now-1][0]

        if demand > self.inventory_level:
            self.lost_sales += demand - self.inventory_level
            self.inventory_level = 0
            self.inventory_position = max(0, self.inventory_position - demand)
        else:
            self.inventory_level -= demand
            self.inventory_position -= demand
        # Remove used items from shelf_life
        for _ in range(int(min(demand, len(self.shelf_life)))):
            oldest = min(self.shelf_life.keys())
            del self.shelf_life[oldest]

    def review_inventory(self):
        if self.inventory_position <= self.r:
            self.place_order()
            self.reorder_count += 1

    def place_order(self):
        order = {'quantity': self.Q, 'arrival_time': self.env.now + self.L}
        self.outstanding_orders.append(order)
        self.inventory_position += self.Q

    def update_costs(self):
        holding_cost = self.h * self.inventory_level
        ordering_cost = self.K if self.inventory_position <= self.r else 0
        outdating_cost = self.w * (self.outdated_items - sum(order['quantity'] for order in self.outstanding_orders))
        shortage_cost = self.p * self.lost_sales
        self.total_cost += holding_cost + ordering_cost + outdating_cost + shortage_cost
        # TODO
        # self.total_cost /= SCALE

    def update_statistics(self):
        # Calculate weekly statistics
        if self.env.now % 7 == 0:  # Every 7 time units (representing a week)
            self.weekly_reorder_frequency = self.reorder_count / 7
            self.weekly_waste_frequency = self.waste_count / 7

        # Reset counters for the next week
        self.reorder_count = 0
        self.waste_count = 0

        # Cumulative statistics
        self.cumulative_lost_sales = self.lost_sales
        self.cumulative_outdated_items = self.outdated_items

        # Average inventory level
        self.average_inventory_level = self.total_inventory_level / self.env.now if self.env.now > 0 else 0

        # Update total inventory level for average calculation
        self.total_inventory_level += self.inventory_level

    def generate_demand(self):
        # means = self.demand_params['means']
        # stds = self.demand_params['stds']
        # weights = self.demand_params['weights']

        # component = np.random.choice(2, p=weights)
        # demand = np.random.normal(means[component], stds[component])

        mean = self.gen_demands[self.env.now-1][0]
        demand = np.random.normal(mean, 0.01)  # TODO random

        return max(0, demand)


def run_simulation(args):
    r, Q, params, sim_time, replications = args
    results = []
    for _ in range(replications):
        env = simpy.Environment()
        system = InventorySystem(env, r, Q, **params)
        env.run(until=sim_time)
        results.append(system.total_cost)
    return np.mean(results)


# Create a new function to restore demand into a file in order to rerun using proposed model
def save_demands_to_file(all_demands, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Replication', 'Time', 'Demand'])
        for rep, demands in enumerate(all_demands, 1):
            for time, demand in enumerate(demands):
                writer.writerow([rep, time, demand])


# log in function (format login message and define behavior)
def log_cpu_and_progress(scenario, max_scenarios):
    cpu_usage = psutil.cpu_percent(interval=1)
    progress = (scenario + 1) / max_scenarios * 100
    logging.info(f"Scenario {scenario + 1}/{max_scenarios} ({progress:.2f}%) - CPU Usage: {cpu_usage}%")


# ## Optimizer
def objective_function(x, params):
    r, Q = x
    return run_simulation((r, Q, params, SIM_DAY, 5))


def generate_initial_solutions(bounds, n_solutions=10):
    solutions = [
        bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * 0.25,  # 25% quantile
        bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * 0.5,  # 50% quantile (median)
        bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * 0.75  # 75% quantile
    ]
    solutions.extend([np.random.uniform(bounds[:, 0], bounds[:, 1]) for _ in range(n_solutions - 3)])
    return solutions


def scatter_search(solutions, bounds, objective_function, pool, cache):
    new_solutions = []
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            if np.random.random() < 0.2:
                new_solution = np.random.uniform(bounds[:, 0], bounds[:, 1])
            else:
                new_solution = (solutions[i] + solutions[j]) / 2
                perturbation_range = (bounds[:, 1] - bounds[:, 0]) * 0.05
                perturbation = np.random.uniform(-perturbation_range, perturbation_range)
                new_solution += perturbation
            new_solution = np.clip(new_solution, bounds[:, 0], bounds[:, 1])
            CACHE_LOCK.acquire()
            if cache.get(new_solution) is None:
                new_solutions.append(new_solution)
            CACHE_LOCK.release()
            time.sleep(0.000001)

    evaluated_solutions = list(
        zip(new_solutions, pool.starmap(objective_function_wrapper, [(x, params, cache) for x in new_solutions])))

    for sol, cost in evaluated_solutions:
        CACHE_LOCK.acquire()
        cache.add(sol, cost)
        CACHE_LOCK.release()
        time.sleep(0.000001)

    evaluated_solutions.sort(key=lambda x: x[1])
    return [sol for sol, _ in evaluated_solutions[:5]]


def tabu_search(current_solution, tabu_list, bounds, objective_function):
    best_neighbor = None
    best_neighbor_cost = float('inf')

    step_sizes = (bounds[:, 1] - bounds[:, 0]) * 0.01  # 1% of the range
    moves = [
        np.array([step_sizes[0], 0]), np.array([-step_sizes[0], 0]),  # Change r
        np.array([0, step_sizes[1]]), np.array([0, -step_sizes[1]])  # Change Q
    ]

    for move in moves:
        neighbor = np.clip(current_solution + move, bounds[:, 0], bounds[:, 1])

        if any(np.all(np.isclose(neighbor, tabu_sol)) for tabu_sol in tabu_list):
            continue

        cost = objective_function(neighbor)

        if cost < best_neighbor_cost:
            best_neighbor = neighbor
            best_neighbor_cost = cost
    # 如果没有找到合适的邻居，返回当前解
    if best_neighbor is None:
        best_neighbor = current_solution
    tabu_list.append(best_neighbor)
    if len(tabu_list) > 10:  # Keep tabu list size limited
        tabu_list.popleft()

    return best_neighbor, tabu_list


def check_convergence(best_cost, recent_costs, threshold=0.01):
    if len(recent_costs) < 100:
        return False
    # Calculate the minimum cost within the last 100 scenarios
    min_recent_cost = min(recent_costs)
    # Compute the percentage improvement relative to the current best cost
    improvement = (best_cost - min_recent_cost) / best_cost
    # Return True if the improvement is less than the threshold, indicating convergence
    return improvement < threshold


def expand_bounds(bounds, expansion_factor=1.2):
    return np.array([
        [bounds[0, 0], bounds[0, 1] * expansion_factor],
        [bounds[1, 0], bounds[1, 1] * expansion_factor]
    ])


# define a wrapper function to replace lambda function in optimize_inventory_system'
# pickle (serialize) a lambda function isn't supported in Python.
def objective_function_wrapper(x, params, cache):
    # print(f"Processing solution {x} on process {os.getpid()}")  # diagnostic tool

    CACHE_LOCK.acquire()
    cached_cost = cache.get(x)
    CACHE_LOCK.release()
    time.sleep(0.000001)

    if cached_cost is not None:
        return cached_cost
    cost = objective_function(x, params)

    CACHE_LOCK.acquire()
    cache.add(x, cost)
    CACHE_LOCK.release()
    time.sleep(0.000001)
    return cost


class SolutionCache:
    def __init__(self, tolerance=0.05):
        self.cache = {}
        self.tolerance = tolerance

    def get(self, solution):
        solution_tuple = tuple(solution)
        if solution_tuple in self.cache:
            return self.cache[solution_tuple]
        for cached_solution, cost in self.cache.items():
            if euclidean(solution, cached_solution) <= self.tolerance:
                return cost

        return None

    def add(self, solution, cost):
        self.cache[tuple(solution)] = cost


# Used in statistical collection to run one scenario of simulation
def run_single_replication(args):
    rep, r, Q, params = args
    env = simpy.Environment()
    system = InventorySystem(env, r, Q, **params)
    # env.run(until=20000)  # Run for 20,000 time units
    env.run(until=SIM_DAY)  # Run for 20,000 time units

    return {
        'Replication': rep + 1,
        'Total Cost': system.total_cost,
        'Average Inventory Level': system.average_inventory_level,
        'Cumulative Lost Sales': system.cumulative_lost_sales,
        'Cumulative Outdated Items': system.cumulative_outdated_items,
        'Weekly Reorder Frequency': system.weekly_reorder_frequency,
        'Weekly Waste Frequency': system.weekly_waste_frequency
    }, system.demands


# Function to save demands periodically
def save_demands_checkpoint(demands, rep):
    checkpoint_filename = f"demands_checkpoint_{rep}.csv"
    save_demands_to_file(demands, checkpoint_filename)
    print(f"Demands checkpoint saved to {checkpoint_filename}")


def run_statistics_collection(best_solution, params, pool):
    # Prepare for statistics collection
    best_r, best_Q = best_solution
    # Run multiple replications for stable statistics
    n_replications = 30

    results = list(tqdm(
        pool.imap(run_single_replication,
                  [(rep, best_r, best_Q, params) for rep in range(n_replications)]),
        total=n_replications,
        desc="Running replications"
    ))

    statistics, all_demands = zip(*results)
    return statistics, all_demands


# Create an optimize_inventory_system function that orchestrates the optimization process:
def optimize_inventory_system(params, initial_bounds, pool, max_scenarios=500):
    cache = SolutionCache()
    bounds = initial_bounds
    best_solution = None
    best_cost = float('inf')
    recent_costs = deque(maxlen=100)
    tabu_list = deque(maxlen=10)
    solutions = generate_initial_solutions(bounds)

    num_processes = os.cpu_count()

    checkpoints = []

    with Pool(num_processes) as pool:
        for scenario in range(max_scenarios):
            new_solutions = scatter_search(solutions, bounds, objective_function_wrapper, pool, cache)

            costs = list(pool.starmap(objective_function_wrapper, [(x, params, cache) for x in new_solutions]))

            for solution, cost in zip(new_solutions, costs):
                recent_costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = solution

            tabu_solution, tabu_list = tabu_search(best_solution, tabu_list, bounds,
                                                   lambda x: objective_function_wrapper(x, params, cache))
            tabu_cost = objective_function_wrapper(tabu_solution, params, cache)
            if tabu_cost < best_cost:
                best_cost = tabu_cost
                best_solution = tabu_solution

            if (scenario + 1) % 100 == 0:
                if check_convergence(best_cost, recent_costs):
                    print(f"Convergence achieved after {scenario + 1} scenarios.")
                    break

            near_bound = np.any(np.isclose(best_solution, bounds[:, 0], rtol=1e-2) |
                                np.isclose(best_solution, bounds[:, 1], rtol=1e-2))

            if near_bound:
                bounds = expand_bounds(bounds)
                solutions = generate_initial_solutions(bounds)
                print(f"Bounds expanded. New bounds: {bounds}")
            else:
                solutions = new_solutions + [best_solution]

            # tracking modules
            if scenario % 10 == 0:  # Log every 10 scenarios
                log_cpu_and_progress(scenario, max_scenarios)

            if (scenario + 1) % 10 == 0:
                print(f"Completed {scenario + 1} scenarios")

            if (scenario + 1) % 100 == 0:
                checkpoints.append((scenario + 1, best_solution.copy(), best_cost))
                print(f"Checkpoint at scenario {scenario + 1}: Best (r, Q) = {best_solution}, Best cost = {best_cost}")

    return best_solution, best_cost


# ## Parameter Setup
if __name__ == '__main__':
    multiprocessing.freeze_support()  # This helps with Windows compatibility

    # Set up parameters
    params = {
        'mu': 10, 'L': 1, 'm': 3, 'K': 100, 'h': 1, 'c': 5, 'w': 5, 'p': 20,
        'demand_params': {'means': [829.59, 1688.82], 'stds': [273.72, 384.07], 'weights': [0.76917451, 0.23082549]}
        # 'demand_params': {'means': [829.59 / SCALE, 1688.82 / SCALE], 'stds': [273.72 / SCALE, 384.07 / SCALE], 'weights': [0.76917451, 0.23082549]}
        # 'demand_params': {'means': [0, 0], 'stds': [1, 1], 'weights': [0.5, 0.5]}
    }

    # Set initial bounds based on the problem parameters
    # Calculate average demand
    avg_demand = params['demand_params']['means'][0] * params['demand_params']['weights'][0] + \
                 params['demand_params']['means'][1] * params['demand_params']['weights'][1]

    initial_bounds = np.array([
      [0, 3 * avg_demand * params['L']],  # bounds for r
      [0, 5 * avg_demand * params['m']]  # bounds for Q
    ])
    print("Parameters and initial bounds set.")

    num_processes = os.cpu_count()  # Use all available CPU cores

    # max_scenarios = 500
    max_scenarios = 100

# ## Actual Run & Statistics Collection

    with multiprocessing.Pool(num_processes) as pool:
         # Run the optimization
         best_solution, best_cost = optimize_inventory_system(params, initial_bounds, pool, max_scenarios)

         print(f"Optimization complete. Best solution (r, Q): {best_solution}")
         print(f"Best cost: {best_cost}")

         # Run statistics collection
         statistics, all_demands = run_statistics_collection(best_solution, params, pool)

         print(f"Completed {len(statistics)} replications for statistics collection.")

         # The following code doesn't need the pool, so it's outside the 'with' block
         # Save demands checkpoints
    n_replications = len(statistics)
    for i in range(0, len(all_demands), 5):
        save_demands_checkpoint(all_demands[i:i + 5], i + 5)

    # After the loop, combine all saved demand checkpoints
    final_demands = []
    for i in range(5, n_replications + 1, 5):
        checkpoint_filename = f"demands_checkpoint_{i}.csv"
        with open(checkpoint_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                final_demands.append([int(row[0]), int(row[1]), float(row[2])])

    # Save the combined demands to the final file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_demands_filename = f"inventory_simulation_demands_{timestamp}.csv"
    save_demands_to_file([final_demands], final_demands_filename)
    print(f"\nFinal demands have been written to {final_demands_filename}")

'''
# Prepare for statistics collection
best_r, best_Q = best_solution


# Run multiple replications for stable statistics
n_replications = 30
num_processes = os.cpu_count()  # Use all available CPU cores

if __name__ == '__main__':
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(run_single_replication, 
                      [(rep, best_r, best_Q, params) for rep in range(n_replications)]),
            total=n_replications,
            desc="Running replications"
        ))

    statistics, all_demands = zip(*results)

print(f"Completed {n_replications} replications for statistics collection.")
'''
# ## Save Result

# Calculate average statistics
avg_stats = {key: np.mean([stat[key] for stat in statistics]) for key in statistics[0] if key != 'Replication'}

# Write detailed results (statistics) to a CSV file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"inventory_simulation_statistics_{timestamp}.csv"

with open(filename, 'w', newline='') as csvfile:
    fieldnames = ['Replication'] + list(avg_stats.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for stat in statistics:
        writer.writerow(stat)

    # Write average statistics
    writer.writerow({'Replication': 'Average', **avg_stats})

print(f"\nDetailed results have been written to {filename}")

# ## Final Result Print and Validation Run

# Print results
print(f"Optimal (r, Q): {best_solution}")
print(f"Optimal cost: {best_cost}")

# Validate the optimal solution
validation_replications = 30  # Increase replications for more accurate validation
best_r, best_Q = best_solution
validated_cost = run_simulation((best_r, best_Q, params, SIM_DAY, validation_replications))

print(f"Validated cost with {validation_replications} replications: {validated_cost}")
print(f"Difference between optimized and validated cost: {abs(best_cost - validated_cost)}")

# Run a single long simulation for detailed statistics
env = simpy.Environment()
system = InventorySystem(env, best_r, best_Q, **params)
env.run(until=SIM_DAY)  # Run for a longer time to get more stable statistics

print("\nDetailed Statistics:")
print(f"Average Inventory Level: {system.average_inventory_level}")
print(f"Cumulative Lost Sales: {system.cumulative_lost_sales}")
print(f"Cumulative Outdated Items: {system.cumulative_outdated_items}")
print(f"Final Total Cost: {system.total_cost}")
