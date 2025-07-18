# Sample code from https://www.redblobgames.com/pathfinding/a-star/
# Copyright 2014 Red Blob Games <redblobgames@gmail.com>
#
# Feel free to use this code in your own projects, including commercial projects
# License: Apache v2.0 <http://www.apache.org/licenses/LICENSE-2.0.html>
#
# Code adapted by Lucas Bezerra <lcdbezerra@gmail.com>

from __future__ import annotations
# some of these types are deprecated: https://www.python.org/dev/peps/pep-0585/
# from typing import Protocol, Iterator, Tuple, TypeVar, Optional
from typing import Iterator, Tuple, TypeVar, Optional
T = TypeVar('T')
import heapq

Location = TypeVar('Location')
GridLocation = Tuple[int, int]


class Graph:
    def neighbors(self, id: Location) -> list[Location]: pass


class SimpleGraph:
    def __init__(self):
        self.edges: dict[Location, list[Location]] = {}
    
    def neighbors(self, id: Location) -> list[Location]:
        return self.edges[id]


class WeightedGraph(Graph):
    def cost(self, from_id: Location, to_id: Location) -> float: pass


class PriorityQueue:
    def __init__(self):
        self.elements: list[tuple[float, T]] = []
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> T:
        return heapq.heappop(self.elements)[1]


def draw_tile(graph, id, style):
    r = " . "
    if 'number' in style and id in style['number']: r = " %-2d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = " > "
        if x2 == x1 - 1: r = " < "
        if y2 == y1 + 1: r = " v "
        if y2 == y1 - 1: r = " ^ "
    if 'path' in style and id in style['path']:   r = " @ "
    if 'start' in style and id == style['start']: r = " A "
    if 'goal' in style and id == style['goal']:   r = " Z "
    if not graph.passable(id): r = "###"
    return r

def draw_grid(graph, **style):
    print("___" * graph.width)
    for y in range(-graph.height//2, graph.height//2+1):
        for x in range(-graph.width//2, graph.width//2+1):
            print("%s" % draw_tile(graph, (x, -y-1), style), end="")
        print()
    print("~~~" * graph.width)


class SquareGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.walls: list[GridLocation] = []
    
    def in_bounds(self, id: GridLocation) -> bool:
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id: GridLocation) -> bool:
        return id not in self.walls
    
    def neighbors(self, id: GridLocation) -> Iterator[GridLocation]:
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] # E W N S
        # see "Ugly paths" section for an explanation:
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return results
    
class SquareGridDiag(SquareGrid):
    def neighbors(self, id: GridLocation) -> Iterator[GridLocation]:
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1), # E W N S
                     (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)] # NE NW SE SW not in this order
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return results

class GridWithWeights(SquareGridDiag):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.weights: dict[GridLocation, float] = {}
    
    def cost(self, from_node: GridLocation, to_node: GridLocation) -> float:
        w = self.weights.get(to_node, 1)
        if is_diag_path(from_node, to_node):
            w *= 1.5
        return w
    
def is_diag_path(a, b):
    (x1, y1) = a
    (x2, y2) = b
    if abs(x1 - x2) + abs(y1 - y2) > 1:
        return True
    else:
        return False

class ObsGrid:
    def __init__(self, obstacles, objs_lvl1):
        assert obstacles.shape == objs_lvl1.shape, "Grid shape doesn't match"
        self.obstacles = obstacles
        self.objs_lvl1 = objs_lvl1
        self.width, self.height = obstacles.shape

    def to_obs_coords(self, id):
        (x, y) = id
        return (x+self.width//2, y+self.height//2)

    def in_bounds(self, id):
        (x, y) = id
        # return (-self.width <= x <= self.width) and (-self.height <= y <= self.height)
        return ((-self.width//2) <= x <= (self.width//2)) and ((-self.height//2) <= y <= (self.height//2))
    
    def passable(self, id):
        (x, y) = self.to_obs_coords(id)
        try:
            return not self.obstacles[x, y]
        except IndexError:
            return True
    
    
    def neighbors(self, id):
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1), # E W N S
                     (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)] # NE NW SE SW not in this order
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        # in_bounds = lambda pos: 0 <= pos[0] < self.width and 0 <= y < self.height
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        # results = filter(self.passable, neighbors)
        return results
    
    def check_lvl1(self, id):
        (x, y) = self.to_obs_coords(id)
        try:
            return self.objs_lvl1[x, y]
        except IndexError:
            return False
    
    def cost(self, from_node, to_node):
        if self.check_lvl1(to_node):
            return 0.
        elif is_diag_path(from_node, to_node):
            return 1.5
        else:
            return 1.


# thanks to @m1sp <Jaiden Mispy> for this simpler version of
# reconstruct_path that doesn't have duplicate entries

def reconstruct_path(came_from: dict[Location, Location],
                     start: Location, goal: Location) -> list[Location]:

    current: Location = goal
    path: list[Location] = []
    if goal not in came_from: # no path was found
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

def heuristic(a: GridLocation, b: GridLocation) -> float:
    (x1, y1) = a
    (x2, y2) = b
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    diag = min(dx,dy)
    str8 = max(dx,dy)-diag
    return 1.5*diag + str8

def a_star_search(graph: WeightedGraph, start: Location, goal: Location):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from: dict[Location, Optional[Location]] = {}
    cost_so_far: dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current: Location = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far
    

if __name__=="__main__":
    diagram4 = GridWithWeights(10, 10)
    diagram4.walls = [(1, 7), (1, 8), (2, 7), (2, 8), (3, 7), (3, 8)]
    diagram4.weights = {}
    start, goal = (1, 4), (2, 9)
    came_from, cost_so_far = a_star_search(diagram4, start, goal)
    draw_grid(diagram4, point_to=came_from, start=start, goal=goal)
    print()
    draw_grid(diagram4, path=reconstruct_path(came_from, start=start, goal=goal))