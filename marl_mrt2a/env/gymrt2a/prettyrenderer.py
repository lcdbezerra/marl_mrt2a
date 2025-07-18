"""PrettyRenderer module providing enhanced visualization for robot environments.

This module extends the basic renderer with image-based representations for robots,
objects, and other elements in the environment.
"""
import pygame
from typing import Tuple, Optional, List, Any
from io import BytesIO
import pkgutil
import base64

from gymrt2a.renderer import PyGameRenderer, EnvObsRenderer, obj_lvl2color
from .image_data import IMAGE_DATA

pygame.init()

def load_img(path):
    # data = pkgutil.get_data("gymrt2a", path)
    # if data is None:
    #     raise FileNotFoundError(f"Could not find {path} in package")
    # return pygame.image.load(BytesIO(data)).convert_alpha()
    # Extract filename from path
    filename = path.split('/')[-1]
    if filename not in IMAGE_DATA:
        raise FileNotFoundError(f"Could not find {filename} in embedded image data")
    
    # Decode base64 data and load as pygame surface
    binary_data = base64.b64decode(IMAGE_DATA[filename])
    return pygame.image.load(BytesIO(binary_data)).convert_alpha()

def load_images(self):
    self.truck_img = self.load_img("images/robot.png")
    self.lvl_img = [self.load_img("images/lvl1.png"),
                    self.load_img("images/lvl2.png"),
                    self.load_img("images/lvl3.png")]
    self.lvl_scale = [.8,.9,1.]

def img_to_spot(self, img, pos, grid="main", scale=1., offset=(0,0)):
    canvas_pos = self.to_canvas_pos(pos, grid=grid, origin="center")
    canvas_pos = (canvas_pos[0]+offset[0], canvas_pos[1]+offset[1])
    img = pygame.transform.smoothscale(img, (scale*self.grid_pix_size[grid][0], 
                                                scale*self.grid_pix_size[grid][1]))
    rect = img.get_rect(center=canvas_pos)
    self.canvas.blit(img, rect)

def paint_robot(self, i, pos, grid="main"):
    self.img_to_spot(self.truck_img, pos, grid=grid, scale=.65, offset=(0,0))

def paint_intention(self, i, pos, grid="main"):
    if not self.check_valid_pos(pos, grid=grid): return
    self.paint_spot(pos, (255,130,130), shape="circle", grid=grid)
    self.write_spot(pos, (255,255,255), txt=str(i), grid=grid)

def paint_object(self, pos, lvl, grid="main", with_paint=True):
    color = obj_lvl2color[lvl]
    if with_paint: self.paint_spot(pos, color, shape="rect", grid=grid)
    self.img_to_spot(self.lvl_img[lvl-1], pos, grid=grid, scale=self.lvl_scale[lvl-1])

def paint_obstacle(self, pos, grid="main"):
    self.paint_spot(pos, (200,)*3, shape="rect", grid=grid)

class PrettyRenderer(PyGameRenderer):
    def load_img(self, path):
        return load_img(path)
    
    def img_to_spot(self, img, pos, grid="main", scale=1., offset=(0,0)):
        img_to_spot(self, img, pos, grid=grid, scale=scale, offset=offset)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_images(self)

    def paint_robot(self, i, pos, grid="main"):
        paint_robot(self, i, pos, grid=grid)

    def paint_intention(self, i, pos, grid="main"):
        paint_intention(self, i, pos, grid=grid)

    def paint_object(self, pos, lvl, grid="main", with_paint=True):
        paint_object(self, pos, lvl, grid=grid, with_paint=with_paint)

    def paint_obstacle(self, pos, grid="main"):
        paint_obstacle(self, pos, grid=grid)

    def render(self, *args, **kwargs):
        super().render(*args, **kwargs)

class PrettyEnvObsRenderer(EnvObsRenderer):
    def load_img(self, path):
        return load_img(path)
    
    def img_to_spot(self, img, pos, grid="main", scale=1., offset=(0,0)):
        img_to_spot(self, img, pos, grid=grid, scale=scale, offset=offset)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        load_images(self)

    def paint_robot(self, i, pos, grid="main"):
        paint_robot(self, i, pos, grid=grid)

    def paint_intention(self, i, pos, grid="main"):
        paint_intention(self, i, pos, grid=grid)

    def paint_object(self, pos, lvl, grid="main", with_paint=True):
        paint_object(self, pos, lvl, grid=grid, with_paint=with_paint)

    def paint_obstacle(self, pos, grid="main"):
        paint_obstacle(self, pos, grid=grid)

    def render(self, *args, **kwargs):
        super().render(*args, **kwargs)