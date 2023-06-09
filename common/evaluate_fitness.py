from shapely.geometry import LineString

from common.robot_map import Map
from common.a_star import AStarPlanner
import config as cf

def eval_fitness(states):
    """
    The function evaluates the fitness of a given state by calculating the length of the path generated
    by the A* algorithm between the start and goal points on a map.

    :param states: The states parameter is a list of integers representing the order in which the
    obstacles should be placed on the map. The order of the obstacles affects the path generated by the
    A* algorithm
    :return: The fitness value of the current state of the genetic algorithm is being returned.
    """
    map_builder = Map(cf.model["map_size"])
    map_points = map_builder.get_points_from_states(states)
    points_list = map_builder.get_points_cords(map_points)

    o_x = [t[0] for t in points_list]
    o_y = [t[1] for t in points_list]

    a_star = AStarPlanner(
        o_x, o_y, cf.model["grid_size"], cf.model["robot_radius"]
    )  # noqa: E501

    r_x, r_y, _ = a_star.planning(
        cf.model["start"], cf.model["start"], cf.model["goal"], cf.model["goal"]
    )

    path = zip(r_x, r_y)

    if len(r_x) > 2:
        path = LineString([(t[0], t[1]) for t in path])
        fitness = path.length
    else:
        fitness = -10

    return round(fitness, 3), points_list, map_points