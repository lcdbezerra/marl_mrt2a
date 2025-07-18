"""Renderer module for visualization of the robot gym environment.

This module provides classes for rendering the robot gym environment using pygame,
including different visualization options for human viewing and agent observations.
"""
import pygame
import numpy as np
import math
from typing import Tuple, List, Dict, Optional, Union, Any, Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymrt2a
from .utils import is_ibex
from .env_recorder import VideoRecorder

# Colormap for object visualization
colormap = mpl.colormaps["OrRd"]
colormap_fn = lambda x: tuple([255.*y for y in colormap(x)[:3]])
obj_lvl2color = {1: colormap_fn(.5),
                 2: colormap_fn(.65),
                 3: colormap_fn(.8),
                 }

class BaseRenderer:
    """Base class for all renderers."""
    pass

class PyGameRenderer(BaseRenderer):
    """Pygame-based renderer for the robot gym environment.
    
    This renderer uses pygame to visualize the environment state,
    including robots, objects, obstacles, and other elements.
    """
    
    def __init__(self, mode: str, sz: Tuple[int, int], fps: int, 
                 obs_range: Optional[int] = None, window_size: Union[int, Tuple[int, int]] = 512, 
                 record: bool = False):
        """Initialize the pygame renderer.
        
        Args:
            mode: Rendering mode ("human" or "rgb_array")
            sz: Grid size (rows, cols)
            fps: Frames per second for rendering
            obs_range: Observation range for agents
            window_size: Size of the rendering window (pixels)
            record: Whether to record video
        """
        self.fps = fps
        self.sz = sz
        self.length = sz[0]
        self.render_mode = mode
        if not isinstance(window_size, tuple):
            self.window_size = (window_size, window_size)
        else:
            self.window_size = window_size
        self.recorder = VideoRecorder(record) if record else None
        self.ibex = is_ibex()  # sets video driver as dummy
        if self.render_mode == "human" and not self.ibex:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            # self.font = pygame.font.Font(pygame.font.get_default_font(), 32)
            self.font = pygame.font.Font(pygame.font.get_default_font(), 12)

        self.grid_origin =   {}
        self.grid_length =   {}
        self.grid_sz =       {}
        self.grid_pix_size = {}

        self.canvas = pygame.Surface(self.window_size)
        self.init_grid("main", sz, origin=(0,0), grid_sz=self.window_size)
        # self.pix_square_size = (
        #     self.window_size[0] / self.length
        # )  # The size of a single grid square in pixels

    def init_grid(self, name, sz, origin=(0,0), grid_sz=None):
        """Initialize a grid for rendering.
        
        Args:
            name: Name of the grid
            sz: Size of the grid (rows, cols)
            origin: Origin position of the grid in pixels
            grid_sz: Size of the grid in pixels
            
        Raises:
            AssertionError: If sz is not a tuple of length 2
        """
        if grid_sz is None: grid_sz = self.window_size
        assert isinstance(sz,tuple) and len(sz)==2, f"Expected sz to be a tuple of length 2, got {sz}"

        self.grid_origin[name]   = origin
        self.grid_length[name]   = sz
        self.grid_sz[name]       = grid_sz
        self.grid_pix_size[name] = (grid_sz[0]/sz[0], grid_sz[1]/sz[1])

    def blank(self):
        """Clear the canvas with a white background."""
        self.canvas.fill((255, 255, 255))

    def reset(self):
        """Reset the renderer state."""
        if self.recorder is not None:
            self.recorder.reset()
        self.blank()
        # self.tick()
    
    def save(self, filename):
        """Save the current canvas as an image.
        
        Args:
            filename: Path to save the image
        """
        pygame.image.save(self.canvas, filename)

    def screenshot(self, filename):
        """Take a screenshot and save it.
        
        Args:
            filename: Path to save the screenshot
        """
        print('Screenshot saved to', filename)
        self.save(filename=filename)

    def check_valid_pos(self, pos, grid="main"):
        """Check if a position is valid within a grid.
        
        Args:
            pos: Position to check (x, y)
            grid: Name of the grid to check in
            
        Returns:
            True if the position is valid, False otherwise
        """
        if pos is None: return False
        return 0 <= pos[0] < self.grid_length[grid][0] and 0 <= pos[1] < self.grid_length[grid][1]

    def to_canvas_pos(self, pos, origin="corner", grid="main"):
        """Convert grid position to canvas pixel position.
        
        Args:
            pos: Grid position (x, y)
            origin: Reference point ("corner" or "center")
            grid: Name of the grid
            
        Returns:
            Pixel position on the canvas (x, y)
            
        Raises:
            ValueError: If origin is not "corner" or "center"
            AssertionError: If grid name is not found
        """
        assert grid in self.grid_origin.keys(), f"Grid '{grid}' not found, available grids: {self.grid_origin.keys()}"
        if origin=="corner":
            # return ( pos[0]*self.pix_square_size, self.window_size-(pos[1]+1)*self.pix_square_size )
            return (self.grid_origin[grid][0] + pos[0]*self.grid_pix_size[grid][0], 
                    self.grid_origin[grid][1] + self.grid_sz[grid][1] - (pos[1]+1)*self.grid_pix_size[grid][1])
        elif origin=="center":
            # return ( (pos[0] + 0.5) * self.pix_square_size, self.window_size-(pos[1] + 0.5) * self.pix_square_size )
            return (self.grid_origin[grid][0] + (pos[0]+.5)*self.grid_pix_size[grid][0], 
                    self.grid_origin[grid][1] + self.grid_sz[grid][1] - (pos[1]+.5)*self.grid_pix_size[grid][1])
        else:
            raise ValueError(f"Unexpected origin type: {origin}, expected 'corner'/'center'")

    def paint_spot(self, pos, rgb, shape="rect", grid="main"):
        """Paint a spot on the canvas at a grid position.
        
        Args:
            pos: Grid position (x, y)
            rgb: Color in RGB format
            shape: Shape to draw ("rect" or "circle")
            grid: Name of the grid
            
        Raises:
            ValueError: If shape is not "rect" or "circle"
            AssertionError: If grid name is not found
        """
        assert grid in self.grid_origin.keys(), f"Grid '{grid}' not found, available grids: {self.grid_origin.keys()}"

        if shape=="rect":
            object_pos = self.to_canvas_pos(pos, origin="corner", grid=grid)
            pygame.draw.rect(
                self.canvas,
                rgb,
                pygame.Rect(
                    object_pos,
                    self.grid_pix_size[grid],
                ),
            )
        elif shape=="circle":
            object_pos = self.to_canvas_pos(pos, origin="center", grid=grid)
            pygame.draw.circle(
                self.canvas,
                rgb,
                object_pos,
                self.grid_pix_size[grid][0] / 3,
            )
        else:
            raise ValueError(f"Unexpected shape type: {shape}")

    def write_spot(self, pos, rgb, txt, grid="main"):
        """Write text at a grid position.
        
        Args:
            pos: Grid position (x, y)
            rgb: Text color in RGB format
            txt: Text to write
            grid: Name of the grid
        """
        text = self.font.render(txt, True, rgb)
        text_rect = text.get_rect()
        text_rect.center = self.to_canvas_pos(pos, origin="center", grid=grid)
        self.canvas.blit(text, text_rect)

    def connect_spots(self, pos1, pos2, rgb=(0,0,0), grid="main"):
        """Draw a line connecting two grid positions.
        
        Args:
            pos1: First grid position (x, y)
            pos2: Second grid position (x, y)
            rgb: Line color in RGB format
            grid: Name of the grid
        """
        pygame.draw.line(
            self.canvas,
            rgb,
            self.to_canvas_pos(pos1, origin="center", grid=grid),
            self.to_canvas_pos(pos2, origin="center", grid=grid),
            width=3,
        )

    def draw_gridlines(self, grid="main"):

        for y in range(self.grid_length[grid][1] + 1):
            pygame.draw.line(
                self.canvas,
                (150, 150, 150),
                (self.grid_origin[grid][0], 
                 self.grid_origin[grid][1] + self.grid_pix_size[grid][1] * y),
                (self.grid_origin[grid][0] + self.grid_sz[grid][0],
                 self.grid_origin[grid][1] + self.grid_pix_size[grid][1] * y),
                width=3,
            )

        for x in range(self.grid_length[grid][0] + 1):
            pygame.draw.line(
                self.canvas,
                (150, 150, 150),
                (self.grid_origin[grid][0] + self.grid_pix_size[grid][0]*x, 
                 self.grid_origin[grid][1]),
                (self.grid_origin[grid][0] + self.grid_pix_size[grid][0]*x, 
                 self.grid_origin[grid][1] + self.grid_sz[grid][1]),
                width=3,
            )

    def paint_grid(self, gd, rgb, grid="main"):
        if isinstance(rgb, tuple):
            rgb_fn = lambda x: rgb if x else (255,255,255)
            for x in range(gd.shape[0]):
                for y in range(gd.shape[1]):
                    if gd[x,y]: self.paint_spot((x,y), rgb, shape="rect", grid=grid)
        elif rgb in mpl.colormaps.values():
            rgb_fn = lambda x: tuple([255*y for y in rgb(x)[:3]])
            for x in range(gd.shape[0]):
                for y in range(gd.shape[1]):
                    self.paint_spot((x,y), rgb_fn(gd[x,y]*1), shape="rect", grid=grid)
        else:
            raise ValueError(f"Unexpected RGB type: {rgb}")

    def paint_robot(self, i, pos, grid="main"):
        self.paint_spot(pos, (0,0,255), shape="circle", grid=grid)
        self.write_spot(pos, (255,255,255), txt=str(i), grid=grid)

    def paint_intention(self, i, pos, grid="main"):
        if not self.check_valid_pos(pos, grid=grid): return
        self.paint_spot(pos, (255,130,130), shape="circle", grid=grid)
        self.write_spot(pos, (255,255,255), txt=str(i), grid=grid)

    def paint_object(self, pos, lvl, grid="main", with_paint=True):
        color = obj_lvl2color[lvl]
        if with_paint: self.paint_spot(pos, color, shape="rect", grid=grid)
        self.write_spot(pos, (255,255,255), txt=str(lvl), grid=grid)

    def paint_obstacle(self, pos, grid="main"):
        self.paint_spot(pos, (200,)*3, shape="rect", grid=grid)
        
    def render(self, robots=[], objects=[], obstacles=[], action_grid=True, grid="main", draw_gridlines=True, show_path=False):

        for r in objects:
            self.paint_object(r.pos, r.lvl, grid=grid)

        for r in obstacles:
            self.paint_obstacle(r.pos, grid=grid)

        if draw_gridlines:
            self.draw_gridlines(grid=grid)
    
        for i,r in enumerate(robots):
            if show_path:
                self.paint_path(r, grid=grid)
            self.paint_robot(i+1, r.pos, grid=grid)
            # self.paint_intention(i+1, r.controller.cmd, grid=grid)
        
        if show_path:
            for r in objects:
                self.paint_object(r.pos, r.lvl, grid=grid, with_paint=False)

    def paint_path(self, r, grid="main"):
        if r.controller.path is None: return

        # Draw a line from the robot along the path
        rgb = (42, 51, 53) # dark blue
        p = r.pos
        for p1 in r.controller.path:
            self.connect_spots(p, p1, rgb=rgb, grid=grid)
            p = p1

    def draw_dashed_line(self, start, end, rel="corner", width=3, dash_length=10, space_length=10, grid="main"):
        x1,y1 = self.to_canvas_pos(start, origin=rel, grid=grid)
        x2,y2 = self.to_canvas_pos(end, origin=rel, grid=grid)

        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        step_dx, step_dy = dx / length, dy / length
        dash_steps = dash_length * (1 / math.sqrt(step_dx**2 + step_dy**2))
        space_steps = space_length * (1 / math.sqrt(step_dx**2 + step_dy**2))

        start = 0
        while start < length:
            end = min(start + dash_steps, length)
            segment_start = (x1 + step_dx * start, y1 + step_dy * start)
            segment_end = (x1 + step_dx * end, y1 + step_dy * end)
            pygame.draw.line(self.canvas, (0,0,0), segment_start, segment_end, width)
            start += dash_steps + space_steps

    def draw_dashed_rectangle(self, top_left, bottom_right, grid="main"):
        x1, y1 = top_left
        x2, y2 = bottom_right
        x2 += 1
        y2 -= 1
        bottom_right = (x2, y2)
        top_right = (x2, y1)
        bottom_left = (x1, y2)

        self.draw_dashed_line(top_left, top_right, grid=grid)       # Top side
        self.draw_dashed_line(bottom_left, bottom_right, grid=grid) # Bottom side
        self.draw_dashed_line(top_left, bottom_left, grid=grid)     # Left side
        self.draw_dashed_line(top_right, bottom_right, grid=grid)   # Right side

    def draw_line(self, start, end, rel="corner", width=3, grid="main", rgb=(0,0,0)):
        x1,y1 = self.to_canvas_pos(start, origin=rel, grid=grid)
        x2,y2 = self.to_canvas_pos(end, origin=rel, grid=grid)
        pygame.draw.line(self.canvas, rgb, (x1,y1), (x2,y2), width)

    def draw_rectangle(self, top_left, bottom_right, grid="main", rgb=(0,0,0)):
        x1, y1 = top_left
        x2, y2 = bottom_right
        x2 += 1
        y2 -= 1
        bottom_right = (x2, y2)
        top_right = (x2, y1)
        bottom_left = (x1, y2)

        self.draw_line(top_left, top_right, grid=grid, rgb=rgb)       # Top side
        self.draw_line(bottom_left, bottom_right, grid=grid, rgb=rgb) # Bottom side
        self.draw_line(top_left, bottom_left, grid=grid, rgb=rgb)     # Left side
        self.draw_line(top_right, bottom_right, grid=grid, rgb=rgb)   # Right side
    
    @property
    def obs_enabled(self):
        return False
    
    def render_obs(self, obs):
        pass

    def render_viz_grid(self, gd):
        pass

    def tick(self):
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # If the episode is recorded, save the frame
            if self.recorder is not None:
                self.recorder.tick(pygame.surfarray.array3d(self.canvas))

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.fps)
            return True
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

    def close(self):
        pygame.display.quit()
        pygame.quit()


class ObservationRenderer:
    def __init__(self, sz, window_size=256):
        self.sz = sz
        self.im = np.zeros(sz)
        plt.ion()
        self.fig = plt.figure()
        extent = (-(sz[0]//2), sz[0]//2, -(sz[1]//2), sz[1]//2)
        self.plt = plt.imshow(self.im, animated=True, vmax=1, origin="lower", extent=extent)
        plt.xticks(np.arange(np.ceil(-(sz[0]//2)), np.floor(sz[0]//2)))
        plt.yticks(np.arange(np.ceil(-(sz[1]//2)), np.floor(sz[1]//2)))
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=100, blit=True)
        # self.ax = plt.gca()
        # self.ax.grid(which='minor', color='w', linestyle='-', linewidth=5, zorder=99)
        self.fig.colorbar(self.plt)
        plt.show()
    
    def update(self, *args):
        self.plt.set_array(self.im)
        return self.plt,

    def color_map(self, val, min=0, max=1):
        val = (val-min)/(max-min)
        val = 255*(1-val)
        return 3*(int(val),)

    def imshow(self, im):
        assert im.shape==self.sz, f"Sizes don't match: sz: {self.sz}, im.shape: {im.shape}"
        self.im = im.T
        return True


class EnvObsRenderer(PyGameRenderer):
    def __init__(self, mode, sz, fps, obs_range, window_size=(1500,1000), orientation="horizontal", minigrid_heatmap=True, minigrid_state=False, pad=50, record=False):
        super().__init__(mode, sz, fps, window_size=window_size, record=record)
        self.minigrid_heatmap = minigrid_heatmap
        self.minigrid_state = minigrid_state
        mini_sz = sz if minigrid_state else (obs_range, obs_range)
        # Initialize grids
        if orientation=="horizontal":
            main_sz = window_size[1]-2*pad
            self.init_grid("main", sz, origin=(pad,pad), grid_sz=((main_sz,main_sz)))
            side_grid_sz = window_size[0]-3*pad-main_sz
            mini_grid_sz = (side_grid_sz-pad/2)/2

            self.init_grid("side1", mini_sz, origin=(main_sz+2*pad,pad), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side2", mini_sz, origin=(main_sz+2*pad+mini_grid_sz+pad/2,pad), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side3", mini_sz, origin=(main_sz+2*pad,pad+mini_grid_sz+pad/2), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side4", mini_sz, origin=(main_sz+2*pad+mini_grid_sz+pad/2,pad+mini_grid_sz+pad/2), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side5", mini_sz, origin=(main_sz+2*pad,pad+2*mini_grid_sz+pad), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side6", mini_sz, origin=(main_sz+2*pad+mini_grid_sz+pad/2,pad+2*mini_grid_sz+pad), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side7", mini_sz, origin=(main_sz+2*pad,pad+3*mini_grid_sz+pad*1.5), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side8", mini_sz, origin=(main_sz+2*pad+mini_grid_sz+pad/2,pad+3*mini_grid_sz+pad*1.5), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
        elif orientation=="vertical":
            main_sz = window_size[0]-2*pad
            self.init_grid("main", sz, origin=(pad,pad), grid_sz=((main_sz,main_sz)))
            side_grid_sz = window_size[1]-3*pad-main_sz
            mini_grid_sz = (side_grid_sz-pad/2)/2

            self.init_grid("side1", mini_sz, origin=(pad, main_sz+2*pad), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side2", mini_sz, origin=(pad, main_sz+2*pad+mini_grid_sz+pad/2), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side3", mini_sz, origin=(pad+mini_grid_sz+pad/2, main_sz+2*pad), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side4", mini_sz, origin=(pad+mini_grid_sz+pad/2, main_sz+2*pad+mini_grid_sz+pad/2), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side5", mini_sz, origin=(pad+2*mini_grid_sz+pad, main_sz+2*pad), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side6", mini_sz, origin=(pad+2*mini_grid_sz+pad, main_sz+2*pad+mini_grid_sz+pad/2), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side7", mini_sz, origin=(pad+3*mini_grid_sz+pad*1.5, main_sz+2*pad), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
            self.init_grid("side8", mini_sz, origin=(pad+3*mini_grid_sz+pad*1.5, main_sz+2*pad+mini_grid_sz+pad/2), 
                        grid_sz=((mini_grid_sz,mini_grid_sz)))
        else:
            raise ValueError(f"Unexpected orientation: {orientation}")
        
        self.colormap = mpl.colormaps["plasma"]

    @property
    def obs_enabled(self):
        return True

    def render_obs(self, obs, agent_id=0):
        if not self.minigrid_heatmap:
            self._render_obs(obs, agent_id)
        else:
            self._render_obs_heatmap(obs, agent_id)
        
    def _render_obs(self, obs, agent_id=0):
        obs = obs[agent_id]

        # Robots
        ind = np.argwhere(obs["robot"])
        for i,index in enumerate(ind):
            self.paint_robot("", index[1:], grid="side1")

        # Intention
        if obs["intention"] is not None:
            ind = np.argwhere(obs["intention"])
            for index in ind:
                self.paint_intention(int(obs["intention"][tuple(index)]), index[1:], grid=f"side2")
        
        # Objects
        ind = np.argwhere(obs["object"])
        for index in ind:
            lvl = index[0]+1
            self.paint_object(index[1:], lvl, grid=f"side{lvl+2}")

        # Obstacles
        ind = np.argwhere(obs["obstacle"])
        for index in ind:
            grid_ind = obs["object"].shape[0]+3
            self.paint_obstacle(index[1:], grid=f"side{grid_ind}")

    def _render_obs_heatmap(self, obs, agent_id=0):
        obs = obs[agent_id]
        # colormap = (128,128,128)

        # Robots
        self.paint_grid(obs["robot"][0], self.colormap, grid="side1")

        # Intention
        if obs["intention"] is not None:
            self.paint_grid(obs["intention"][0], self.colormap, grid="side2")

        # Objects
        for i in range(3):
            self.paint_grid(obs["object"][i], self.colormap, grid=f"side{i+3}")

        # Obstacles
        grid_ind = obs["object"].shape[0]+3
        self.paint_grid(obs["obstacle"][0], self.colormap, grid=f"side{grid_ind}")

    def render_viz_grid(self, gd):
        self.paint_grid(gd[0], self.colormap, grid="side7")
        self.paint_grid(gd[1], self.colormap, grid="side8")

    def render(self, *args, **kwargs):
        super().render(*args, **kwargs)

if __name__=="__main__":
    r = ObservationRenderer((17,17))
    print(r)