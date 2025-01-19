import simpy
import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt


class InventorySystem:
    def __init__(self, env, r, Q, L, m, K, h, c, w, p, demand_mean, demand_cv2):
        self.env = env
        self.r = r  # Reorder point
        self.Q = Q  # Order quantity
        self.L = L  # Lead time
        self.m = m  # Shelf life
        self.K = K  # Fixed ordering cost
        self.h = h  # Holding cost per unit per time
        self.c = c  # Purchase cost per unit
        self.w = w  # Outdating cost per unit
        self.p = p  # Shortage cost per unit
        self.demand_mean = demand_mean
        self.demand_cv2 = demand_cv2

        self.inventory_level = r
        self.inventory_position = r
        self.outdated_items = 0
        self.lost_sales = 0

        self.shelf_life = {}
        self.outstanding_orders = [] # a list of dictionary
        self.reorder_count = 0
        self.waste_count = 0
        self.total_cost = 0

        self.weekly_reorder_frequency = 0
        self.weekly_waste_frequency = 0
        self.cumulative_lost_sales = 0
        self.cumulative_outdated_items = 0
        self.total_inventory_level = 0
        self.average_inventory_level = 0

        # New attributes for daily tracking
        self.daily_perished_items = 0
        self.daily_holding_cost = 0
        self.daily_ordering_cost = 0
        self.daily_outdating_cost = 0
        self.daily_shortage_cost = 0

        self.demands = []

        # New variables for tracking
        self.holding_costs = []
        self.ordering_costs = []
        self.outdating_costs = []
        self.shortage_costs = []
        self.inventory_levels = []

        self.time_history = []  # Add this line to create the time_history list
        self.inventory_levels = []
        self.inventory_position_history = []

        self.total_demand = 0
        self.total_satisfied_demand = 0
        self.number_of_stockouts = 0

        self.env.process(self.run())

        # New attributes for weekly cost tracking
        self.weekly_costs = {
            'total': [],
            'holding': [],
            'ordering': [],
            'outdating': [],
            'shortage': []
        }
        self.current_week_costs = {
            'total': 0,
            'holding': 0,
            'ordering': 0,
            'outdating': 0,
            'shortage': 0
        }



    def run(self):
        while True:
            self.time_history.append(self.env.now)
            self.inventory_levels.append(self.inventory_level)
            self.inventory_position_history.append(self.inventory_position)

            print(
                f"Time: {self.env.now}, Inventory Level: {self.inventory_level}, Inventory Position: {self.inventory_position}")

            # Reset daily values
            self.daily_perished_items = 0
            self.daily_holding_cost = 0
            self.daily_ordering_cost = 0
            self.daily_outdating_cost = 0
            self.daily_shortage_cost = 0

            yield self.env.timeout(1)  # Time step
            self.check_order_arrivals()
            self.remove_perished_items()
            print(f"Perished Items: {self.daily_perished_items}")
            self.demand_occurrence()
            print(f"Demand: {self.demands[-1]}")
            self.review_inventory()
            self.update_costs()
            print(
                f"Costs - Holding: {self.daily_holding_cost:.2f}, Ordering: {self.daily_ordering_cost:.2f}, Outdating: {self.daily_outdating_cost:.2f}, Shortage: {self.daily_shortage_cost:.2f}")
            self.update_statistics()
            self.update_weekly_costs()

    def check_order_arrivals(self):
        arrived_orders = [order for order in self.outstanding_orders if order['arrival_time'] <= self.env.now]
        for order in arrived_orders:
            print(f"Order arrived at time {self.env.now} with quantity {order['quantity']}")
            self.inventory_level += order['quantity']
            self.inventory_position += order['quantity']
            self.outstanding_orders.remove(order)
            for i in range(int(order['quantity'])):
                self.shelf_life[self.env.now + i / order['quantity']] = self.env.now + self.m

    def remove_perished_items(self):
        perished = sum(1 for time, expiry in list(self.shelf_life.items()) if expiry <= self.env.now)
        if perished > 0:
            self.inventory_level -= perished
            self.inventory_position -= perished
            self.outdated_items += perished
            self.daily_perished_items = perished  # Update daily perished items
            self.waste_count += 1
            for time in list(self.shelf_life.keys()):
                if self.shelf_life[time] <= self.env.now:
                    del self.shelf_life[time]


    def demand_occurrence(self): # note to change the shelf life dictionary while demand consumes the items
        demand = self.generate_demand()
        self.total_demand += demand
        self.demands.append(demand)
        satisfied_demand = min(demand, self.inventory_level)
        self.total_satisfied_demand += satisfied_demand

        # Remove consumed items from shelf_life
        items_to_remove = []
        remaining_demand = satisfied_demand
        for time, expiry in sorted(self.shelf_life.items()):
            if remaining_demand <= 0:
                break
            if remaining_demand >= 1:
                items_to_remove.append(time)
                remaining_demand -= 1
            else:
                # Partial unit consumed
                self.shelf_life[time + remaining_demand] = expiry
                items_to_remove.append(time)
                break

        for time in items_to_remove:
            del self.shelf_life[time]

        if demand > self.inventory_level:
            self.lost_sales += demand - self.inventory_level
            if self.inventory_level == 0:
                self.number_of_stockouts += 1
            self.inventory_position = max(0, self.inventory_position - demand)
            self.inventory_level = 0
        else:
            self.inventory_level -= demand
            self.inventory_position -= demand

    def review_inventory(self):
        if self.inventory_position <= self.r:
            self.place_order()
            self.reorder_count += 1

    def place_order(self):
        order = {'quantity': self.Q, 'arrival_time': self.env.now + self.L}
        self.outstanding_orders.append(order)
        self.inventory_position += self.Q
        self.total_cost += self.K  # Add ordering cost
        self.ordering_costs.append(self.K)  # Record ordering cost
        self.daily_ordering_cost += self.K  # Update daily ordering cost
        print(f"Order placed at time {self.env.now} with quantity {self.Q}")

    def update_costs(self):
        self.daily_holding_cost = self.h * self.inventory_level
        self.daily_outdating_cost = self.w * self.daily_perished_items
        self.daily_shortage_cost = self.p * max(0, self.demands[-1] - self.inventory_level)

        self.total_cost += self.daily_holding_cost + self.daily_outdating_cost + self.daily_shortage_cost

        self.holding_costs.append(self.daily_holding_cost)
        self.outdating_costs.append(self.daily_outdating_cost)
        self.shortage_costs.append(self.daily_shortage_cost)

        # Update current week costs
        self.current_week_costs['total'] += (self.daily_holding_cost + self.daily_ordering_cost +
                                             self.daily_outdating_cost + self.daily_shortage_cost)
        self.current_week_costs['holding'] += self.daily_holding_cost
        self.current_week_costs['ordering'] += self.daily_ordering_cost
        self.current_week_costs['outdating'] += self.daily_outdating_cost
        self.current_week_costs['shortage'] += self.daily_shortage_cost

    def update_statistics(self):
        if self.env.now % 7 == 0:
            self.weekly_reorder_frequency = self.reorder_count / 7
            self.weekly_waste_frequency = self.waste_count / 7
            self.reorder_count = 0
            self.waste_count = 0

        self.cumulative_lost_sales = self.lost_sales
        self.cumulative_outdated_items = self.outdated_items
        self.average_inventory_level = sum(self.inventory_levels) / len(
            self.inventory_levels) if self.inventory_levels else 0

    def generate_demand(self):
        shape = 1 / self.demand_cv2
        scale = self.demand_mean / shape
        demand = gamma.rvs(shape, scale=scale)
        return max(0, demand)

    def update_weekly_costs(self):
        if self.env.now % 7 == 0 and self.env.now > 0:
            for cost_type in self.weekly_costs:
                self.weekly_costs[cost_type].append(self.current_week_costs[cost_type])
                self.current_week_costs[cost_type] = 0


def run_single_simulation(r, Q, sim_time):
    env = simpy.Environment()
    params = {
        'L': 1,
        'm': 3,
        'K': 100,
        'h': 1,
        'c': 5,
        'w': 5,
        'p': 20,
        'demand_mean': 10,
        'demand_cv2': 0.23
    }
    system = InventorySystem(env, r, Q, **params)
    env.run(until=sim_time)
    return system

# Run the simulation
r = 10 # Example reorder point, adjust as needed
Q = 50  # Example order quantity, adjust as needed
sim_time = 20000

result = run_single_simulation(r, Q, sim_time)

# Output results
print(f"Total Cost: {result.total_cost:.2f}")
print(f"Average Inventory Level: {result.average_inventory_level:.2f}")
print(f"Cumulative Lost Sales: {result.cumulative_lost_sales:.2f}")
print(f"Cumulative Outdated Items: {result.cumulative_outdated_items:.2f}")
print(f"Weekly Reorder Frequency: {result.weekly_reorder_frequency:.2f}")
print(f"Weekly Waste Frequency: {result.weekly_waste_frequency:.2f}")
print(f"Total Holding Cost: {sum(result.holding_costs):.2f}")
print(f"Total Ordering Cost: {sum(result.ordering_costs):.2f}")
print(f"Total Outdating Cost: {sum(result.outdating_costs):.2f}")
print(f"Total Shortage Cost: {sum(result.shortage_costs):.2f}")
print(f"Service Level: {result.total_satisfied_demand / result.total_demand * 100:.2f}%")
print(f"Number of Stockouts: {result.number_of_stockouts}")
# Calculate and print average weekly costs
print("\nAverage Weekly Costs:")
for cost_type in result.weekly_costs:
    avg_cost = sum(result.weekly_costs[cost_type]) / len(result.weekly_costs[cost_type])
    print(f"Average Weekly {cost_type.capitalize()} Cost: {avg_cost:.2f}")


# Update the plotting code
plt.figure(figsize=(12, 6))
plt.plot(result.time_history, result.inventory_levels, label='Inventory Level')
plt.plot(result.time_history, result.inventory_position_history, label='Inventory Position')
plt.title('Inventory Level and Position Over Time')
plt.xlabel('Time')
plt.ylabel('Quantity')
plt.legend()
plt.savefig('inventory_comparison.png')
plt.close()

# Plot cost components
cost_components = ['Holding', 'Ordering', 'Outdating', 'Shortage']
costs = [sum(result.holding_costs), sum(result.ordering_costs),
         sum(result.outdating_costs), sum(result.shortage_costs)]

plt.figure(figsize=(10, 6))
plt.bar(cost_components, costs)
plt.title('Total Costs by Component')
plt.ylabel('Cost')
plt.savefig('cost_components_plot.png')
plt.close()


# Troubleshooting
print(f"Min Inventory Level: {min(result.inventory_levels)}")
print(f"Max Inventory Level: {max(result.inventory_levels)}")

import statistics
print(f"Median Inventory Level: {statistics.median(result.inventory_levels)}")

plt.figure(figsize=(10, 6))
plt.hist(result.inventory_levels, bins=50)
plt.title('Distribution of Inventory Levels')
plt.xlabel('Inventory Level')
plt.ylabel('Frequency')
plt.savefig('inventory_distribution.png')
plt.close()

print(f"Number of inventory level records: {len(result.inventory_levels)}")

# Plot weekly average costs
weeks = range(1, len(result.weekly_costs['total']) + 1)

plt.figure(figsize=(12, 6))
plt.plot(weeks, result.weekly_costs['total'], label='Total Cost')
plt.plot(weeks, result.weekly_costs['holding'], label='Holding Cost')
plt.plot(weeks, result.weekly_costs['ordering'], label='Ordering Cost')
plt.plot(weeks, result.weekly_costs['outdating'], label='Outdating Cost')
plt.plot(weeks, result.weekly_costs['shortage'], label='Shortage Cost')
plt.title('Weekly Average Costs')
plt.xlabel('Week')
plt.ylabel('Cost')
plt.legend()
plt.savefig('weekly_average_costs.png')
plt.close()
