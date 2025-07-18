from multiprocessing.sharedctypes import Value
import numpy as np
from .astar import ObsGrid, a_star_search, reconstruct_path, draw_grid


act = {
    "0": 0,
    "U": 1,
    "UR": 2,
    "R": 3,
    "DR": 4,
    "D": 5,
    "DL": 6,
    "L": 7,
    "UL": 8,
}
num2act = list(act.keys())

def getActionFromDif(dif: tuple) -> int:
    """
    Determine the action the robot should take based on the difference vector ('dif').

    This function checks the direction in which the robot should move based on the
    difference passed as an argument, where 'dif' is a tuple (dx, dy). The robot can
    move only one block per animation frame.

    Parameters:
    dif (tuple): A tuple containing differences (dx, dy) representing the direction of movement.

    Returns:
    int: A integer representing the action the robot should perform. Possible actions include
         1 (up), 5 (down), 7 (left), 3 (right), 2 (up-right), 8 (up-left),
         4 (down-right), or 6 (down-left).

    Exceptions:
    ValueError: Raised if the absolute value any of the components from the 'dif' difference is bigger than one.

    Example:
    >>> getActionFromDif((1, 0))
    3
    >>> getActionFromDif((-1, -1))
    6
    >>> getActionFromDif((-2, 0))
    Invalid difference as argument, dif=(-2, 0)
    """

    assert abs(dif[0]) <= 1 and abs(dif[1]) <= 1, f"Invalid difference as argument, dif={dif}"

    if dif[0] == 0:
        if dif[1] > 0:
            return act["U"]
        else:
            return act["D"]
    elif dif[1] == 0:
        if dif[0] > 0:
            return act["R"]
        else:
            return act["L"]
    elif dif[0] > 0 and dif[1] > 0:
        return act["UR"]
    elif dif[0] < 0 and dif[1] > 0:
        return act["UL"]
    elif dif[0] > 0 and dif[1] < 0:
        return act["DR"]
    elif dif[0] < 0 and dif[1] < 0:
        return act["DL"]
    else:
        raise ValueError(f"Unexpected behavior, dif={dif}")
    
def is_adjacent(nxt: tuple, pos: tuple) -> bool:
    """
    Check whether the next position is adjacent to current position.

    Parameters:
    nxt (tuple): A tuple containing the next position coordinates (x1, y1).
    pos (tuple): A tuple containing the current position coordinates (x0, y0).

    Returns:
    bool: A boolean representing whether the next position is adjacent to current.

    Example:
    >>> is_adjacent((1, 0), (0, 0))
    True
    """

    dx = abs(nxt[0] - pos[0])
    dy = abs(nxt[1] - pos[1])

    return dx <= 1 and dy <= 1


class Controller:
    """
    Abstract class of a controller.
    """

    def __init__(self, robot):
        self.robot = robot
        self.cmd = None

    def __call__(self, obs, pos):
        raise NotImplementedError("Abstract controller")

    def assign_command(self, cmd, pos):
        self.cmd = cmd


class RandomController(Controller):
    """
    Class that implements a random controller.

    When called it returns a integer from 0 to 8 (both included)
    which refers to a random action.
    """

    def __init__(self, robot):
        super().__init__(robot)

    def __call__(self, obs, pos) -> int:
        """
        Dummy method that returns a random action.
        """

        return np.random.randint(9)


class ManualController(Controller):
    """
    Class that implements a manual controller.

    When called it maps keyboard inputs into actions.
    """

    def __call__(self, obs, pos):
        while True:
            txt = input("Use WASD for moving: ")
            mapping = {
                "Q": "UL",
                "W": "U",
                "E": "UR",
                "A": "L",
                "S": "0",
                "D": "R",
                "Z": "DL",
                "X": "D",
                "C": "DR",
            }
            try:
                return act[mapping[txt]]
            except:
                print(f"Unrecognized input: {txt}\nTry again")

class ShortestPathController(Controller):
    """
    Class that implements a controller that finds the shortest path.

    When called it first checks whether there is a command. If no,
    it returns a random action, if yes, it calculates the difference
    between the command coordinates and the input position, then
    it returns a action related to direction of the difference vector.

    Exemple:
    >>> ShortestPathController(None, (0, 0)) # Supose self.cmd is None
    None
    """

    def assign_command(self, cmd: tuple, pos: tuple) -> 'int | None':
        """
        Calls global_cmd() and assign its output to self.cmd.

        Parameters:
        cmd (tuple): A tuple containing command coordinates.
        pos (tuple): A tuple containing position coordinates.

        Returns:
        None: It should return None since this method is for variable assignment.
        int: If it fails to assign the command, returns 0.
        """

        if cmd is None or cmd == (None, None):
            return 0
        
        self.cmd = self.global_cmd(cmd, pos)
    
    def global_cmd(self, cmd: tuple, pos: tuple) -> 'tuple | None':
        """
        Calculate and return new command based on input command and input position.

        Parameters:
        cmd (tuple): A tuple containing command coordinates.
        pos (tuple): A tuple containing position coordinates.

        Returns:
        tuple: A tuple new command coordinates.

        Example:
        >>> global_cmd((1, 0), (2, 2))
        (3, 2)
        """

        if cmd is None or cmd == (None, None):
            return (None, None)
        
        return (cmd[0] + pos[0], cmd[1] + pos[1])

    def __call__(self, obs: np.ndarray, pos: tuple) -> int:
        """
        Dummy method that calculates a difference vector and
        returns an action related to the direction of that vector.
        """

        if self.cmd is None:
            return np.random.randint(1, 9)
        
        dif = (self.cmd[0] - pos[0], self.cmd[1] - pos[1])
        
        if dif == (0, 0):
            self.cmd = None
            return act["0"]
        
        if dif[0] == 0:
            if dif[1] > 0:
                return act["U"]
            else:
                return act["D"]
        elif dif[1] == 0:
            if dif[0] > 0:
                return act["R"]
            else:
                return act["L"]
        elif dif[0] > 0 and dif[1] > 0:
            return act["UR"]
        elif dif[0] < 0 and dif[1] > 0:
            return act["UL"]
        elif dif[0] > 0 and dif[1] < 0:
            return act["DR"]
        elif dif[0] < 0 and dif[1] < 0:
            return act["DL"]
        else:
            raise ValueError(f"Unexpected situation, cmd={self.cmd}, pos={pos}")
        

class PathfindingController(ShortestPathController):
    """
    Class that implements a controller based on the A* path finding algorithm.

    When called, it calculates the difference vector based on the command position
    (self.cmd) and input position (pos). Then, it checks if there is a path and
    if it is valid. If yes, it proceeds to the next step. If not, it follows the
    no_path_policy. If the path is invalid, it finds a new path using the A* algorithm.
    """

    def __init__(self, robot):
        super().__init__(robot)
        self.path = None

    def assign_command(self, cmd: tuple, pos: tuple) -> None:
        super().assign_command(cmd, pos)
        self.path = None

    def __call__(self, obs: np.ndarray, pos: tuple, next_step=True) -> int:
        """
        Dummy method that attempts to follow the next step in self.path or assigns
        a new path if necessary.
        """

        if (self.cmd is None) or (self.cmd == (None, None)):
            return self.no_command_policy(obs, pos)
        
        dif = (self.cmd[0] - pos[0], self.cmd[1] - pos[1])

        if dif == (0, 0):
            self.cmd = None
            self.path = None
            return act["0"]
        
        if not self.check_path_valid(self.path, obs, pos):
            self.path = self.find_path(obs, pos, dif)
        
        # If no path was found
        if self.path is None:
            return self.no_path_policy(obs, dif)

        return self.next_step_from_path(obs, pos, dif) if next_step else 0

    def no_command_policy(self, obs: np.ndarray, pos: tuple) -> int:
        """
        Returns a random action.
        """

        return np.random.randint(9)
    
    def find_path(self, obs: np.ndarray, pos: tuple, dif: tuple) -> list:
        """
        Finds a path using A* Algorithm.

        This method calculates a path from the current position (pos) to a target position (dif)
        on a grid based on the A* pathfinding algorithm. It takes into account the obstacles
        represented in the observation array and the level 1 object grid.

        Parameters:
        obs (np.ndarray): The observation array representing the environment.
        pos (tuple): The current position as a tuple (x, y).
        dif (tuple): The target position as a tuple (x, y).

        Returns:
        list: A list of waypoints representing the path from the current position to
        the target position in global coordinates.
        None: Returns None if no valid path is found.

        Notes:
        - The obstacle_grid is constructed from the 'obs' array.
        - The (0, 0) and 'dif' positions are treated as non-obstacle points.
        - The 'path' is converted to global coordinates using 'global_cmd'.
        - The starting position is excluded from the returned path.
        """

        # Create grid
        obstacle_grid = self.robot.parser.not_passable_grid(obs)
        lvl1_obj_grid = self.robot.parser(obs)["object"][0]
        assert len(lvl1_obj_grid.shape) == 2
        
        # (0, 0) and dif shouldn't be obstacles
        obstacle_grid[obstacle_grid.shape[0]//2, obstacle_grid.shape[1]//2] = 0
        try:
            obstacle_grid[dif[0] + obstacle_grid.shape[0]//2, dif[1] + obstacle_grid.shape[1]//2] = 0
        except:
            pass

        grid = ObsGrid(obstacle_grid, lvl1_obj_grid)
        came_from, cost_so_far = a_star_search(grid, start = (0, 0), goal = dif)
        path = reconstruct_path(came_from, start = (0, 0), goal = dif)

        if len(path) == 0: return None

        # Convert to global coordinates
        path = [self.global_cmd(x, pos) for x in path]

        return path[1:] # without starting position
        

    def next_step_from_path(self, obs: np.ndarray, pos: tuple, dif: tuple) -> int:
        """
        Determine the next step to reach a target position along a precalculated path.

        This method calculates the next step needed to reach a target position ('dif') based
        on a precalculated path. It considers the current position ('pos') and the observation
        of the environment ('obs').

        Parameters:
        obs (np.ndarray): The observation array representing the environment.
        pos (tuple): The current position as a tuple (x, y).
        dif (tuple): The target position as a tuple (x, y).

        Returns:
        int: An integer representing the action to be taken to move towards the target position.
             The action is determined by the difference between the next waypoint ('nxt') and
             the current position.

        Notes:
        - The 'path' attribute should be precalculated using the 'find_path' method.
        - The method checks if the goal has been reached and handles such cases.
        - The 'nxt_dif' represents the difference between the next waypoint and the current position.
        """

        nxt = self.path[0]

        # check if goal was reached
        if len(self.path) == 1:
            assert nxt == self.cmd, f"Unexpected behavior, len(path) = 1 but nxt: {nxt}, dif: {dif}"
            self.cmd  = None # ENSURES REEVALUATION WHEN REACHING AN OBJECT
            # self.path = None
        else:
            self.path = self.path[1:]

        nxt_dif = (nxt[0] - pos[0], nxt[1] - pos[1])

        return getActionFromDif(nxt_dif)
    
    def check_path_valid(self, path: list, obs: np.ndarray, pos: tuple) -> bool:
        """
        Check the validity of a given path for the robot.

        This method checks if a provided path is valid for the robot to follow. It verifies
        the following conditions:
        1. The path is defined (not None).
        2. The robot's current position ('pos') is on the path as expected.
        3. The path is unobstructed by obstacles in the environment ('obs').

        Parameters:
        path (list): A list of waypoints representing the path.
        obs (np.ndarray): The observation array representing the environment.
        pos (tuple): The current position of the robot as a tuple (x, y).

        Returns:
        bool: True if the path is valid, False otherwise.
        """

        # is the path defined?
        if path is None: return False
        
        # is the robot in the path?
        if pos not in path:
            return False
        
        # Convert to local position
        path = [(x[0] - pos[0] + obs.shape[0]//2, x[1] - pos[1] + obs.shape[1]//2) for x in path]

        # Do not consider destination
        path = path[:-1]
        not_passable = self.robot.parser.not_passable_grid(obs)

        # is the path obstructed?
        if any([not_passable(x) for x in path]):
            return False
        
        return True

    def no_path_policy(self, obs: np.ndarray, dif: tuple) -> int:
        """
        Define a policy for handling situations where no valid path is found.

        This method specifies a policy for the robot to follow when no valid path
        to the target position ('dif') is available. It considers the difference
        vector ('dif') between the current and target positions and chooses an action
        accordingly.

        Parameters:
        obs (np.ndarray): The observation array representing the environment.
        dif (tuple): The difference vector between the current and target positions.

        Returns:
        int: An integer representing the action chosen by the policy based on 'dif'.
        """
        
        self.cmd = None

        if dif[0] == 0:
            if dif[1] > 0:
                return act["U"]
            else:
                return act["D"]
        elif dif[1] == 0:
            if dif[0] > 0:
                return act["R"]
            else:
                return act["L"]
        elif dif[0] > 0 and dif[1] > 0:
            return act["UR"]
        elif dif[0] < 0 and dif[1] > 0:
            return act["UL"]
        elif dif[0] > 0 and dif[1] < 0:
            return act["DR"]
        elif dif[0] < 0 and dif[1] < 0:
            return act["DL"]