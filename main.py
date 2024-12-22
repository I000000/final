import random
import pandas as pd
from datetime import timedelta, datetime
from copy import deepcopy
from tabulate import tabulate

num_buses = 10
num_drivers = num_buses * 2  # итоговое количество водителей может быть меньше этого значения

work_hours_type_a = 8
work_hours_type_b = 12
work_days_type_a = 5
work_days_type_b = 2
break_dur_a = timedelta(minutes=10)
break_dur_b = [timedelta(minutes=15), timedelta(minutes=20)]
lunch_dur_a = timedelta(hours=1)
lunch_dur_b = timedelta(minutes=20)

route_dur = timedelta(minutes=60)
directions = ["A->B", "B->A"]
shift_start = datetime.strptime("06:00", "%H:%M")
shift_end = datetime.strptime("03:00", "%H:%M") + timedelta(days=1)

rush_hours = [(datetime.strptime("07:00", "%H:%M"), datetime.strptime("09:00", "%H:%M")),
              (datetime.strptime("17:00", "%H:%M"), datetime.strptime("19:00", "%H:%M"))]

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class Driver:
    def __init__(self, driver_id, driver_type):
        self.driver_id = driver_id
        self.driver_type = driver_type  # 'standard' или 'shift'
        self.available = True
        self.shift_started = False
        self.shift_start_time = None
        self.days_worked = 0
        self.days_rested = 2
        self.just_worked = False


class Bus:
    def __init__(self, bus_id):
        self.bus_id = bus_id
        self.available = True
        self.current_direction = None  # 'A->B' или 'B->A'


def is_rush_hour(current_time, route_dur, rush_hours):
    route_end = current_time + route_dur
    for start, end in rush_hours:
        if current_time >= start and route_end <= end + timedelta(minutes=30):
            return True
    return False


def handle_daily_reset(drivers):
    for driver in drivers:
        driver.work_time = 0
        if driver.just_worked:
            driver.just_worked = False
            driver.available = True
            driver.shift_start_time = None
            driver.days_worked += 1
        else:
            driver.days_rested += 1

        if (driver.driver_type == "Standard" and driver.days_worked >= 5) or (
                driver.driver_type == "Shift" and driver.days_worked >= 2):
            driver.available = True
            driver.days_worked = 0

        if driver.days_rested >= 2:
            driver.available = True
            driver.days_rested = 0


def generate_schedule():
    schedule = []

    for day in days:
        current_time = shift_start
        is_weekday = day not in ["Saturday", "Sunday"]

        while current_time < shift_end:

            for bus in buses:
                available_drivers = [driver for driver in drivers if driver.available]
                if not available_drivers:
                    continue

                driver = random.choice(available_drivers)
                driver.shift_start_time = current_time
                direction = random.choice(directions)

                if is_weekday:
                    is_peak = is_rush_hour(current_time, route_dur, rush_hours)
                else:
                    is_peak = False

                if current_time >= shift_end:
                    continue

                if (
                        driver.driver_type == "Standard" and current_time.hour >= driver.shift_start_time.hour + work_hours_type_a) or (
                        driver.driver_type == "Shift" and current_time.hour >= driver.shift_start_time.hour + work_hours_type_b):
                    driver.available = False
                    driver.days_worked += 1
                    driver.just_worked = True
                    continue

                schedule_entry = {
                    "Day": day,
                    "Bus ID": bus.bus_id,
                    "Driver ID": driver.driver_id,
                    "Driver Type": driver.driver_type,
                    "Direction": direction,
                    "Route Start": current_time.strftime("%H:%M"),
                    "Route End": (current_time + route_dur).strftime("%H:%M"),
                    "Break at": None,
                    "Break for": None,
                    "Rush Hour": is_peak
                }

                if not is_peak and driver.driver_type == "Standard" and current_time.hour == 13:
                    lunch_time = current_time.strftime("%H:%M")
                    schedule_entry["Break at"] = lunch_time
                    schedule_entry["Break for"] = 60
                    current_time += lunch_dur_a
                elif not is_peak and driver.driver_type == "Shift" and current_time.hour % 2 == 0:
                    break_time = current_time.strftime("%H:%M")
                    break_duration = random.choice(break_dur_b)
                    schedule_entry["Break at"] = break_time
                    schedule_entry["Break for"] = break_duration.seconds / 60
                    current_time += break_duration
                else:
                    current_time += route_dur

                schedule.append(schedule_entry)

        handle_daily_reset(drivers)

    return schedule


drivers = [Driver(driver_id=i, driver_type='Standard' if i % 2 == 0 else 'Shift') for i in range(1, num_drivers + 1)]
buses = [Bus(bus_id=i) for i in range(1, num_buses + 1)]

"""ГЕНЕРАЦИЯ РАСПИАНИЯ С ИСПОЛЬЗОВАНИЕМ ГЕНЕТИЧЕСКОГО АЛГОРИТМОМА"""

print("▄██████░▄█████░██████▄░▄█████░████████░██████░▄█████░░░▄████▄░██░░░░░▄██████░▄█████▄░█████▄░██████░████████░██░░░██░▄██▄▄██▄"
      "\n██░░░░░░██░░░░░██░░░██░██░░░░░░░░██░░░░░░██░░░██░░░░░░░██░░██░██░░░░░██░░░░░░██░░░██░██░░██░░░██░░░░░░██░░░░██░░░██░██░██░██"
      "\n██░░███░█████░░██░░░██░█████░░░░░██░░░░░░██░░░██░░░░░░░██░░██░██░░░░░██░░███░██░░░██░█████▀░░░██░░░░░░██░░░░███████░██░██░██"
      "\n██░░░██░██░░░░░██░░░██░██░░░░░░░░██░░░░░░██░░░██░░░░░░░██████░██░░░░░██░░░██░██░░░██░██░░██░░░██░░░░░░██░░░░██░░░██░██░██░██"
      "\n▀█████▀░▀█████░██░░░██░▀█████░░░░██░░░░██████░▀█████░░░██░░██░██████░▀█████▀░▀█████▀░██░░██░██████░░░░██░░░░██░░░██░██░██░██"
      "\n░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")
print("\n\nThe begging of the genetic algorithm\n")

POPULATION_SIZE = 100
MAX_GENERATIONS = 50
P_MUTATION = 0.1
P_ELITISM = 0.3

population = [generate_schedule() for _ in range(POPULATION_SIZE)]


def evaluate_fitness(schedule):
    total_drivers = len(set(entry['Driver ID'] for entry in schedule))
    routes_during_rush_time = sum(1 for entry in schedule if entry['Rush Hour'])
    breaks = sum([entry['Break for'] for entry in schedule if entry['Break for'] is not None])
    even_routes = abs(sum(1 for entry in schedule if entry['Direction'] == "A->B") - sum(
        1 for entry in schedule if entry['Direction'] == "B->A"))
    return len(schedule) + routes_during_rush_time * 10 - even_routes - total_drivers * 20 - breaks * 0.2


def mutate(schedule, mutation_rate=0.01):
    for entry in schedule:
        if random.random() < mutation_rate:
            mutation = random.randint(1, 2)
            if mutation == 1:
                cur_id = entry['Driver ID']
                while entry['Driver ID'] == cur_id:
                    entry['Driver ID'] = random.randint(1, len(drivers))
                    entry['Driver Type'] = 'Standard' if entry['Driver ID'] % 2 == 0 else 'Shift'
            if mutation == 2:
                if entry['Direction'] == "A->B":
                    entry['Direction'] = "B->A"
                else:
                    entry['Direction'] = "A->B"
    return schedule


def tournament_selection(population, tournament_size=5, selection_prob=0.8):
    tournament_group = random.sample(population, tournament_size)
    best = max(tournament_group, key=evaluate_fitness)
    if random.random() < selection_prob:
        return best
    else:
        return random.choice(tournament_group)


def multi_point_crossover(parent1, parent2, num_crossovers=2):
    child = []
    points = sorted(random.sample(range(1, len(parent1)), num_crossovers))
    start = 0
    for i in range(num_crossovers):
        if i % 2 == 0:
            child.extend(parent1[start:points[i]])
        else:
            child.extend(parent2[start:points[i]])
        start = points[i]
    if num_crossovers % 2 == 0:
        child.extend(parent1[start:])
    else:
        child.extend(parent2[start:])
    return child


for generation in range(MAX_GENERATIONS):
    print(f"\nValuating generation {generation + 1}/{MAX_GENERATIONS}...")
    fitness_values = [evaluate_fitness(individual) for individual in population]
    best_fitness = max(fitness_values)
    average_fitness = sum(fitness_values) / len(population)
    print(f"    Best fitness: {best_fitness}, Average fitness: {average_fitness}")

    new_population = []
    num_elites = int(P_ELITISM * POPULATION_SIZE)
    elites = sorted(population, key=evaluate_fitness, reverse=True)[:num_elites]
    new_population.extend([deepcopy(elite) for elite in elites])

    while len(new_population) < POPULATION_SIZE:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child = multi_point_crossover(parent1, parent2)
        child = mutate(child, P_MUTATION)
        new_population.append(child)
    population = new_population
    print(f"Generation {generation + 1} completed.")

best_schedule = max(population, key=evaluate_fitness)
schedule_df = pd.DataFrame(best_schedule)
schedule_df.to_csv("genetic_schedule.csv", index=False)

print("\nSchedule was successfully generated using genetic algorithm\nYou can find it in the genetic_schedule.csv "
      "file\n\n")

"""ЛИНЕЙНАЯ ГЕНЕРЕРАЦИЯ РАСПИСАНИЯ"""

print("██░░░░░██████░██████▄░▄█████░▄████▄░█████▄░░░▄████▄░██░░░░░▄██████░▄█████▄░█████▄░██████░████████░██░░░██░▄██▄▄██▄"
      "\n██░░░░░░░██░░░██░░░██░██░░░░░██░░██░██░░██░░░██░░██░██░░░░░██░░░░░░██░░░██░██░░██░░░██░░░░░░██░░░░██░░░██░██░██░██"
      "\n██░░░░░░░██░░░██░░░██░█████░░██░░██░█████▀░░░██░░██░██░░░░░██░░███░██░░░██░█████▀░░░██░░░░░░██░░░░███████░██░██░██"
      "\n██░░░░░░░██░░░██░░░██░██░░░░░██████░██░░██░░░██████░██░░░░░██░░░██░██░░░██░██░░██░░░██░░░░░░██░░░░██░░░██░██░██░██"
      "\n██████░██████░██░░░██░▀█████░██░░██░██░░██░░░██░░██░██████░▀█████▀░▀█████▀░██░░██░██████░░░░██░░░░██░░░██░██░██░██"
      "\n░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")

print("\n\nThe begging of the linear algorithm\n")

linear_schedule = []

for day in days:
    current_time = shift_start
    is_weekday = day not in ["Saturday", "Sunday"]

    while current_time < shift_end:

        for bus in buses:
            available_drivers = [driver for driver in drivers if driver.available]
            if not available_drivers:
                continue

            driver = random.choice(available_drivers)
            driver.shift_start_time = current_time
            direction = random.choice(directions)

            if is_weekday:
                is_peak = is_rush_hour(current_time, route_dur, rush_hours)
            else:
                is_peak = False

            if current_time >= shift_end:
                continue

            if (
                    driver.driver_type == "Standard" and current_time.hour >= driver.shift_start_time.hour + work_hours_type_a) or (
                    driver.driver_type == "Shift" and current_time.hour >= driver.shift_start_time.hour + work_hours_type_b):
                driver.available = False
                driver.days_worked += 1
                driver.just_worked = True
                continue

            schedule_entry = {
                "Day": day,
                "Bus ID": bus.bus_id,
                "Driver ID": driver.driver_id,
                "Driver Type": driver.driver_type,
                "Direction": direction,
                "Route Start": current_time.strftime("%H:%M"),
                "Route End": (current_time + route_dur).strftime("%H:%M"),
                "Rush Hour": is_peak,
                "Break at": None,
                "Break for": None
            }

            if driver.driver_type == "Standard" and current_time.hour == 13:
                lunch_time = current_time.strftime("%H:%M")
                schedule_entry["Break at"] = lunch_time
                schedule_entry["Break for"] = 60
                current_time += lunch_dur_a
            elif driver.driver_type == "Shift" and current_time.hour % 2 == 0:
                break_time = current_time.strftime("%H:%M")
                break_duration = random.choice(break_dur_b)
                schedule_entry["Break at"] = break_time
                schedule_entry["Break for"] = break_duration.seconds / 60
                current_time += break_duration
            else:
                current_time += route_dur

            linear_schedule.append(schedule_entry)

    handle_daily_reset(drivers)

print("\nSchedule was successfully generated using linear algorithm\nYou can find it in the linear_schedule.csv "
      "file\n\n")

"""СРАВНЕНИЕ РАБОТЫ ДВУХ АЛГОРИТМОВ"""

print("▄█████░▄█████▄░▄██▄▄██▄░█████▄░▄████▄░█████▄░██████░▄██████░▄█████▄░██████▄"
      "\n██░░░░░██░░░██░██░██░██░██░░██░██░░██░██░░██░░░██░░░██░░░░░░██░░░██░██░░░██"
      "\n██░░░░░██░░░██░██░██░██░█████▀░██░░██░█████▀░░░██░░░▀█████▄░██░░░██░██░░░██"
      "\n██░░░░░██░░░██░██░██░██░██░░░░░██████░██░░██░░░██░░░░░░░░██░██░░░██░██░░░██"
      "\n▀█████░▀█████▀░██░██░██░██░░░░░██░░██░██░░██░██████░██████▀░▀█████▀░██░░░██"
      "\n░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")

print("\n\n                       Comparing two algorithms:\n")

linear_schedule_df = pd.DataFrame(linear_schedule)

linear_unique_drivers = linear_schedule_df['Driver ID'].nunique()
linear_peak_routes = linear_schedule_df['Rush Hour'].sum()
linear_break_time = schedule_df['Break for'].sum()

genetic_unique_drivers = schedule_df['Driver ID'].nunique()
genetic_peak_routes = schedule_df['Rush Hour'].sum()
genetic_break_time = schedule_df['Break for'].sum()

comparison = {
    "Metric": ["Unique drivers (less is better)", "Buses during rush hour", "Time spent on breaks"],
    "Linear Algorithm": [linear_unique_drivers, linear_peak_routes, linear_break_time],
    "Genetic Algorithm": [genetic_unique_drivers, genetic_peak_routes, genetic_break_time]
}

comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv("comparison_results.csv", index=False)
print(comparison_df)

"""ВЫВОД ИТОГОВОГО ЛУЧШЕГО ВАРИАНТА РАСПИСАНИЯ"""

print("\n\n█████▄░▄█████░▄██████░████████░░░▄██████░▄█████░██░░░██░▄█████░██████▄░██░░░██░██░░░░░▄█████"
      "\n██░░██░██░░░░░██░░░░░░░░░██░░░░░░██░░░░░░██░░░░░██░░░██░██░░░░░██░░░██░██░░░██░██░░░░░██░░░░"
      "\n█████░░█████░░▀█████▄░░░░██░░░░░░▀█████▄░██░░░░░███████░█████░░██░░░██░██░░░██░██░░░░░█████░"
      "\n██░░██░██░░░░░░░░░░██░░░░██░░░░░░░░░░░██░██░░░░░██░░░██░██░░░░░██░░░██░██░░░██░██░░░░░██░░░░"
      "\n█████▀░▀█████░██████▀░░░░██░░░░░░██████▀░▀█████░██░░░██░▀█████░██████▀░▀█████▀░██████░▀█████"
      "\n░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░")

print("\n\nPrinting the best schedule\n")

print(tabulate(schedule_df, headers='keys', tablefmt='psql'))
