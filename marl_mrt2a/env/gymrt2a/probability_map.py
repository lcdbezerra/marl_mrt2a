"""Probability map module for object and robot placement in environments.

This module provides classes for probabilistic placement of objects 
within a grid-based environment, including uniform and sector-based options.
"""
import numpy as np
from typing import List, Tuple, Optional, Union, Any
# from math import prod


class ProbabilityMap:
    """Probability distribution for sampling positions in a grid.
    
    This class manages a probability distribution over a grid, allowing
    for sampling of positions based on that distribution. It can be used
    for placing objects or robots in an environment.
    """
    
    def __init__(self, sz: Tuple[int, int], rate: Optional[Union[float, List[float]]] = None, 
                 max_obj_lvl: Optional[int] = None):
        """Initialize the probability map.
        
        Args:
            sz: Size of the grid (rows, cols)
            rate: Spawn rate or list of rates for different object levels
            max_obj_lvl: Maximum object level when rate is not a list
            
        Raises:
            ValueError: If rate is not a list and max_obj_lvl is not provided
        """
        self.sz = sz
        self.L = np.prod(sz)
        self.ind_grid = np.arange(self.L).reshape(sz)

        # assert isinstance(rate, list), f"Expected a list of spawn rates, got: {rate}"
        if rate in (False, True, None):
            rate = [0]
        if not isinstance(rate, list):
            if max_obj_lvl is None:
                raise ValueError("Expected max_obj_lvl, since rate is not a list")
                # rate = [rate]
            else:
                # Divide the rate equally among levels
                rate = [rate/max_obj_lvl]*max_obj_lvl

        self.rate = rate
        self.total_rate = sum(rate)
        try:
            self.rel_rate = [x/self.total_rate for x in rate]
        except ZeroDivisionError:
            self.rel_rate = [1]
        self.spawn_prob = 1-np.exp(-self.total_rate)

        self.update_prob()

    def update_prob(self, p_grid: Optional[np.ndarray] = None) -> None:
        """Update the probability grid.
        
        Args:
            p_grid: New probability grid. If None, uses uniform distribution.
        """
        if p_grid is not None:
            self.p_grid = p_grid/np.sum(p_grid)
        else:
            # if no matrix is provided, assume uniform distribution
            self.p_grid = np.ones(self.sz, dtype=np.float32)/self.L # uniform distribution

    def _to_grid_ind(self, ind: np.ndarray) -> List[Tuple[int, int]]:
        """Convert linear indices to grid coordinates.
        
        Args:
            ind: Array of linear indices
            
        Returns:
            List of (x, y) grid coordinates
        """
        inds = np.unravel_index(ind, self.sz)
        lst = []
        for i in range(len(ind)):
            lst.append( (inds[0][i], inds[1][i]) )
        return lst

    def sample(self, N: int = 1, avail_grid: Optional[np.ndarray] = None, with_lvl: bool = False) -> List[Tuple[int, int]]:
        """Sample positions from the probability distribution.
        
        Args:
            N: Number of positions to sample
            avail_grid: Boolean grid indicating available positions
            with_lvl: Whether to also sample object levels
            
        Returns:
            List of (x, y) grid coordinates
        """
        if N==0: return []
        if not with_lvl:
            if avail_grid is not None:
                p_grid = self.p_grid*avail_grid
                p_grid /= np.sum(p_grid)
            else:
                p_grid = self.p_grid
            ind = np.random.choice(self.L, size=N, p=p_grid.reshape(-1), replace=False)
            return self._to_grid_ind(ind)
        else:
            pass # TODO

    def sample_obj_num(self, avail_grid: np.ndarray) -> List[int]:
        """Sample the number of objects at each level.
        
        Args:
            avail_grid: Boolean grid indicating available positions
            
        Returns:
            List of object levels (1-indexed)
        """
        # Get the number of available spots in the grid
        avail_grid = np.logical_and(avail_grid, self.p_grid)
        # Sample the total number of spawns
        total = np.random.binomial(n=np.sum(avail_grid), p=self.spawn_prob)
        # Then sample categorical trials
        N = np.random.choice(len(self.rel_rate), p=self.rel_rate, size=total)
        # The output is a list of object levels
        return [x+1 for x in N]

class SectorProbMap(ProbabilityMap):
    """Probability distribution that focuses on specific sectors of the grid.
    
    This class extends ProbabilityMap by dividing the grid into sectors
    and allowing focusing of the probability on one sector at a time.
    """
    
    def __init__(self, sz: Tuple[int, int], Lsec: int = 2, rate: Optional[Union[float, List[float]]] = None, 
                max_obj_lvl: Optional[int] = None):
        """Initialize the sector probability map.
        
        Args:
            sz: Size of the grid (rows, cols)
            Lsec: Number of sectors in each dimension
            rate: Spawn rate or list of rates for different object levels
            max_obj_lvl: Maximum object level when rate is not a list
        """
        super().__init__(sz, rate, max_obj_lvl)
        self.sec_sz = (Lsec, Lsec)
        self.Lsec = Lsec
        self.Nsec = Lsec**2

        self.sector_len = ( sz[0]//Lsec, sz[1]//Lsec )
        self.update_sector()

    def _get_bounds_from_sec(self, sec: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get grid bounds for a sector.
        
        Args:
            sec: Sector coordinates (x, y)
            
        Returns:
            Tuple of (x_bounds, y_bounds) where each bound is (min, max)
        """
        ind_x = (self.sector_len[0]*sec[0], self.sector_len[0]*(sec[0]+1))
        ind_y = (self.sector_len[1]*sec[1], self.sector_len[1]*(sec[1]+1))
        return ind_x, ind_y

    def update_sector(self, sec: Optional[Tuple[int, int]] = None) -> None:
        """Update the probability grid to focus on a specific sector.
        
        Args:
            sec: Sector coordinates (x, y). If None, selects a random sector.
            
        Raises:
            AssertionError: If the provided sector is invalid
        """
        if sec is None:
            # Choose a random sector if none is provided
            sec = np.random.choice(self.Nsec)
            sec = np.unravel_index(sec, self.sec_sz)
        assert len(sec)==2, f"Invalid sector provided: {sec}"

        p_grid = np.zeros(self.sz, dtype=np.float32)
        ind_x, ind_y = self._get_bounds_from_sec(sec)
        p_grid[ind_x[0]:ind_x[1], ind_y[0]:ind_y[1]] = 1

        super().update_prob(p_grid)


if __name__ == "__main__":
    sz = (10,10)
    p = ProbabilityMap(sz)

    p_grid = np.zeros(sz)
    p_grid[4:8,5:8] = 1
    p_grid[1:5,0:5] = 1
    p.update_prob(p_grid)

    avail = np.full_like(p_grid, True)
    avail[3:5, 3:6] = False

    vals = p.sample(1000, avail_grid=avail)
    x = [v[0] for v in vals]
    y = [v[1] for v in vals]
    import matplotlib.pyplot as plt
    plt.scatter(x,y)
    plt.xlim([-.5,sz[0]-.5])
    plt.ylim([-.5,sz[1]-.5])
    plt.show()