from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gymrt2a.robot import Robot

class Object:
    """Base class for all objects in the robot gym environment."""
    
    def __init__(self, pos: tuple):
        """
        Initialize an object.
        
        Args:
            pos: The position of the object (x, y)
        """
        self.pos = pos

    def pickup(self, robot: "Robot") -> bool:
        """
        Attempt to pick up this object with a robot.
        
        Args:
            robot: The robot attempting to pick up the object
            
        Returns:
            True if pickup was successful, False otherwise
        """
        return True

    def reset(self) -> None:
        """Reset the object to its initial state."""
        pass


class LevelObject(Object):
    """
    An object that requires multiple robots to pick up based on its level.
    Level determines how many robots must interact with it before it can be picked up.
    """
    
    def __init__(self, pos: tuple, lvl: int = 1):
        """
        Initialize a level object.
        
        Args:
            pos: The position of the object (x, y)
            lvl: The level of the object determining how many robots are needed
                 to pick it up (default: 1)
        
        Raises:
            AssertionError: If level is not an integer
        """
        super().__init__(pos)
        assert int(lvl) == lvl, f"Invalid object level: {lvl}"
        self.lvl = lvl
        self.robot_count = 0
        self.robot_list: List["Robot"] = []

    def pickup(self, robot: "Robot") -> bool:
        """
        Attempt to pick up this object with a robot.
        
        Args:
            robot: The robot attempting to pick up the object
            
        Returns:
            True if the object is successfully picked up (enough robots have attempted),
            False if more robots are needed
        """
        self.robot_list.append(robot)
        if self.robot_count == self.lvl - 1:
            return True
        else:
            self.robot_count += 1
            return False

    def remaining_robots(self) -> int:
        """
        Calculate how many more robots are needed to pick up this object.
        
        Returns:
            Number of additional robots needed
        """
        return self.lvl - self.robot_count
        
    def can_pickup(self, robot: "Robot") -> bool:
        """
        Check if the object can be picked up with one more robot.
        
        Args:
            robot: The robot attempting to pick up the object
            
        Returns:
            True if one more robot is needed for pickup, False otherwise
        """
        return self.robot_count == (self.lvl - 1)

    def reset(self) -> None:
        """Reset the object's state, clearing robot count and list."""
        self.robot_count = 0
        self.robot_list = []

    def reset_robots_cmd(self) -> None:
        """Clear the path commands for all robots associated with this object."""
        for robot in self.robot_list:
            robot.controller.path = None


class StackObject(LevelObject):
    """
    A stack of objects that must be picked up multiple times.
    Each pickup reduces the stack level until it reaches zero.
    """
    
    def pickup(self, robot: "Robot") -> bool:
        """
        Attempt to pick up one item from the stack.
        
        Args:
            robot: The robot attempting to pick up from the stack
            
        Returns:
            True if the stack is now empty, False otherwise
        """
        self.robot_list.append(robot)
        self.lvl -= 1
        return self.lvl <= 0