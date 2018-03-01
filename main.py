import numpy as np
import random
from enum import Enum, auto
import collections
import sys

# number of rounds greedy algorithm is started with randomly chosen candidates
# of the following lists
RANDOM_PARAMETER_ROUNDS = 30

best_part_candidates = [2, 4, 5, 20, 50]

vehicle_sort_function_candidates = [
    lambda vd: (len(vd[0].rides), vd[1]),
    lambda vd: (-len(vd[0].rides), vd[1]),
    lambda vd: (len(vd[0].rides), -vd[1]),
    lambda vd: (-len(vd[0].rides), vd[1]),
    lambda vd: (vd[1]),
    lambda vd: -(vd[1]),
    lambda vd: vd[0].available_time,
    lambda vd: -vd[0].available_time
]

ride_sort_function_candidates = [
    lambda ride: (ride.earliest_start, ride.latest_finish, -ride.distance),
    lambda ride: (ride.latest_start, -ride.distance, ride.latest_finish),
    lambda ride: (-ride.distance, ride.latest_start, ride.latest_finish),
    lambda ride: (ride.latest_start, -ride.distance, ride.latest_finish),
    lambda ride: (ride.latest_finish, ride.earliest_start, -ride.distance),
]


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    NONE = auto()


LOG_LEVEL = LogLevel.INFO

Ride = collections.namedtuple("Ride", ['index', 'a', 'b', 'x', 'y', 'earliest_start', 'latest_finish', 'distance',
                                       'latest_start'])
Ride.__doc__ = """\
Contains the information available for a single ride.
Attributes:
    index: the index of the ride in the input file
    a, b: starting position of the ride
    x, y: end position of the ride
    earliest_start: earliest start
    latest_finish: latest finish
    dist: distance between start and end
    latest_start: latest possible starting time still allowing to arrive in time.
"""


class Vehicle:
    """
    Contains the position and time the vehicle will next be available after, as well as a
    list of all assigned rides.
    """
    __slots__ = ('available_time', 'available_position', 'rides')

    def __init__(self):
        self.available_time = 0
        self.available_position = [0, 0]
        self.rides = []


class Problem:
    """
    Represents all header information known about the problem.
    """
    __slots__ = ('rows', 'cols', 'vehicles_in_fleet', 'rides_to_plan', 'bonus', 'steps')

    def __init__(self, rows, cols, vehicles_in_fleet, rides_to_plan, bonus, steps):
        self.rows = rows
        self.cols = cols
        self.vehicles_in_fleet = vehicles_in_fleet  # F
        self.rides_to_plan = rides_to_plan
        self.bonus = bonus
        self.steps = steps


def calculate_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def debug_out(s=""):
    if LOG_LEVEL.value <= LogLevel.DEBUG.value:
        print(s)


def info_out(s=""):
    if LOG_LEVEL.value <= LogLevel.INFO.value:
        print(s)


def assign_ride_to_vehicle(vehicle, dist, ride):
    """Assign ride to vehicle, updating position and time of vehicle."""
    vehicle.rides.append(ride.index)
    vehicle.available_position[0], vehicle.available_position[1] = ride.x, ride.y

    veh_start = max([ride.earliest_start, vehicle.available_time + dist])
    vehicle.available_time = veh_start + ride.distance


def read_problem(problem_name: str):
    info_out("reading: {}".format(problem_name))
    with open("data/" + problem_name + ".in", "r") as file:
        file_content = file.read()
    file_content = file_content.split('\n')
    file_header = np.array(file_content[0].split(' '), dtype=np.int)
    file_content.pop(0)
    file_content.pop(len(file_content) - 1)  # remove empty lines at the end
    # Rows, Cols, vehicles in Fleet, Number of rides, Bonus for starting time, T steps in simulation
    problem_definition = Problem(*file_header)
    # a,b: row,col start; x,y row,col end; s earliest start, f latest finish
    rides = []
    index = 0
    for line in file_content:
        str_decoded = np.array(line.split(' '), dtype=np.int)
        a, b, x, y, s, f = str_decoded[:7]
        dist = calculate_distance(a, b, x, y)
        latest_start = f - dist  # TODO check this
        ride = Ride(index, a, b, x, y, s, f, dist, latest_start)
        rides.append(ride)
        index += 1
    return problem_definition, rides


def plan_rides(rides, problem_definition, best_factor, vehicle_sort_function, ride_sort_function):
    """
    Greedily assigns rides to available cars, sorting and randomly choosing one of the
    best positions.
    """
    curr_assigned_rides = 0
    curr_distance_sum = 0
    vehicles = [Vehicle() for _ in range(problem_definition.vehicles_in_fleet)]

    rides_sorted = sorted(rides, key=ride_sort_function)

    for ride in rides_sorted:

        # distances for all vehicles to the starting position
        distances = [calculate_distance(*veh.available_position, ride.a, ride.b) for veh in vehicles]
        debug_out(distances)

        available_vehicles = [(vehicle, dist) for (vehicle, dist) in zip(vehicles, distances) if
                              (vehicle.available_time + dist <= ride.latest_start)]
        available_vehicles = sorted(available_vehicles, key=vehicle_sort_function)
        debug_out(available_vehicles)
        debug_out()

        bonus_vehicles = [(vehicle, dist) for (vehicle, dist) in available_vehicles if
                          vehicle.available_time + dist <= ride.earliest_start]

        if bonus_vehicles:
            index_upper_bound = int(len(bonus_vehicles) / best_factor)
            index = np.random.randint(0, index_upper_bound) if index_upper_bound >= 1 else 0
            vehicle, dist = bonus_vehicles[index]

            assign_ride_to_vehicle(vehicle, dist, ride)
            curr_assigned_rides += 1
            curr_distance_sum += ride.distance
            curr_distance_sum += problem_definition.bonus
        elif available_vehicles:
            index_upper_bound = int(len(available_vehicles) / best_factor)
            index = np.random.randint(0, index_upper_bound) if index_upper_bound >= 1 else 0
            vehicle, dist = available_vehicles[index]
            assign_ride_to_vehicle(vehicle, dist, ride)
            curr_assigned_rides += 1
            curr_distance_sum += ride.distance

    solution = [x.rides for x in vehicles]
    return solution, curr_assigned_rides, curr_distance_sum


def try_random_parameters_greedily(rides, problem_definition):
    max_points = 0
    current_solution = None
    current_number_assigned_rides = 0
    for i in range(RANDOM_PARAMETER_ROUNDS):
        params = (
            random.choice(best_part_candidates), random.choice(vehicle_sort_function_candidates),
            random.choice(ride_sort_function_candidates))
        new_solution, new_nr_assigned_rides, new_points = plan_rides(rides, problem_definition, *params)
        info_out("Greedily assigning with random parameters, round {}".format(i))
        info_out("Assigned rides: {}, total rides: ".format(new_nr_assigned_rides, len(rides)))
        info_out("Points made this round: {}".format(new_points))

        if new_points > max_points:
            # add successful parameters again so they will be taken with a
            # higher probability next time
            best_part_candidates.append(params[0])
            vehicle_sort_function_candidates.append(params[1])
            ride_sort_function_candidates.append(params[2])

            # update solution candidate
            max_points = new_points
            current_number_assigned_rides = new_nr_assigned_rides
            current_solution = new_solution
    return current_solution, max_points, current_number_assigned_rides


def write_result(problem_name: str, current_solution, max_points, current_number_assigned_rides, rides):
    info_out("Writing outfile for {}".format(problem_name))
    with open("out/" + problem_name + ".occams_razors.out", 'w') as out_file:
        for vehicle in current_solution:
            debug_out(vehicle)
            if len(vehicle) > 0:
                next_line = "{} {}\n".format(len(vehicle), " ".join([str(x) for x in vehicle]))
                out_file.write(next_line)
            else:
                out_file.write("0\n")
    info_out("Assigned Rides: {}, Total Rides: {}".format(current_number_assigned_rides, len(rides)))
    info_out("Total Distance and Bonus: {}".format(max_points))


def main():
    if len(sys.argv) > 1:
        problem_name = sys.argv[1]
    else:
        problem_name = "a_example"

    problem, rides = read_problem(problem_name)

    solution, points, assigned_rides = try_random_parameters_greedily(rides, problem)
    write_result(problem_name, solution, points, assigned_rides, rides)


if __name__ == '__main__':
    main()
