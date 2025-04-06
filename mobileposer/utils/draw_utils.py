import math
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

from mobileposer.config import *


DEVICE_POSITIONS = [(-5, 0, -10.0), (-2.5, 0, -10.0), (0, 0, -10.0), (2.5, 0, -10.0), (5, 0, -10.0)]


def init_pygame():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


# Initialize a global cache
text_cache = {}

def draw_text(position, textString, size=14):
    if textString not in text_cache:
        font = pygame.font.SysFont("Courier", size, True)
        textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
        textData = pygame.image.tostring(textSurface, "RGBA", True)
        text_cache[textString] = {
            "data": textData,
            "width": textSurface.get_width(),
            "height": textSurface.get_height()
        }
    
    cache_entry = text_cache[textString]
    glRasterPos3d(*position)
    glDrawPixels(cache_entry["width"], cache_entry["height"], GL_RGBA, GL_UNSIGNED_BYTE, cache_entry["data"])


def draw_cuboid(w=2, h=2, d=0.4, colors=None):
    w /= 2
    h /= 2
    d /= 2

    # Define all vertices and colors upfront
    vertices = [
        [ w,  d, -h],
        [-w,  d, -h],
        [-w,  d,  h],
        [ w,  d,  h],
        [ w, -d,  h],
        [-w, -d,  h],
        [-w, -d, -h],
        [ w, -d, -h]
    ]

    faces = [
        (0, 1, 2, 3),  # Top
        (4, 5, 6, 7),  # Bottom
        (0, 3, 4, 7),  # Front
        (1, 2, 5, 6),  # Back
        (0, 1, 5, 4),  # Left
        (3, 2, 6, 7)   # Right
    ]

    glBegin(GL_QUADS)
    for face, color in zip(faces, colors):
        glColor3f(*color)
        for vertex in face:
            glVertex3f(*vertices[vertex])
    glEnd()


def draw(device_id, ori, acc):
    [nx, ny, nz, w] = list(ori)
    glLoadIdentity()
    device_pos = DEVICE_POSITIONS[device_id] 

    glTranslatef(*device_pos)
    device_label = list(sensor.device_ids.keys())[device_id]
    draw_text((-0.7, 1.8, 0), device_label, 14)
    angle = 2 * math.acos(w) * 180.00 / math.pi
    sin_half_angle = math.sqrt(1.0 - w * w)
    if sin_half_angle < 0.001:
        axis = (1.0, 0.0, 0.0)
    else:
        axis = (nx / sin_half_angle, ny / sin_half_angle, nz / sin_half_angle)
    glRotatef(angle, *axis)
    draw_cuboid(1, 1, 1)


def resize_win(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-7, 7, -7, 7, 0, 15)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()