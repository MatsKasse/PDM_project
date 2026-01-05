from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
from mpscenes.obstacles.urdf_obstacle import UrdfObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.obstacles.cylinder_obstacle import CylinderObstacle

import os


wall_length = 20
wall_thickness=0.1
wall_obstacles_dicts = [
    {
        'type': 'box', 
         'geometry': {
             'position': [wall_length/2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': wall_thickness
        },
        'high': {
            'position' : [wall_length/2.0, 0.0, 0.4],
            'width': wall_length,
            'height': 0.8,
            'length': wall_thickness,
        },
        'low': {
            'position' : [wall_length/2.0, 0.0, 0.4],
            'width': wall_length,
            'height': 0.8,
            'length': wall_thickness,
        },
    },
    {
        'type': 'box', 
         'geometry': {
             'position': [0.0, wall_length/2.0, 0.4], 'width': wall_thickness, 'height': 0.8, 'length': wall_length
        },
        'high': {
            'position' : [0.0, wall_length/2.0, 0.4],
            'width': wall_thickness,
            'height': 0.8,
            'length': wall_length,
        },
        'low': {
            'position' : [0.0, wall_length/2.0, 0.4],
            'width': wall_thickness,
            'height': 0.8,
            'length': wall_length,
        },
    },
    {
        'type': 'box', 
         'geometry': {
             'position': [0.0, -wall_length/2.0, 0.4], 'width': wall_thickness, 'height': 0.8, 'length': wall_length
        },
        'high': {
            'position' : [0.0, -wall_length/2.0, 0.4],
            'width': wall_thickness,
            'height': 0.8,
            'length': wall_length,
        },
        'low': {
            'position' : [0.0, -wall_length/2.0, 0.4],
            'width': wall_thickness,
            'height': 0.8,
            'length': wall_length,
        },
    },
    {
        'type': 'box', 
         'geometry': {
             'position': [-wall_length/2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': wall_thickness
        },
        'high': {
            'position' : [-wall_length/2.0, 0.0, 0.4],
            'width': wall_length,
            'height': 0.8,
            'length': wall_thickness,
        },
        'low': {
            'position' : [-wall_length/2.0, 0.0, 0.4],
            'width': wall_length,
            'height': 0.8,
            'length': wall_thickness,
        },
    },
    
    
    {
        'type': 'box', 
         'geometry': {
             'position': [7.5, 4.0, 0.4], 'width': wall_thickness, 'height': 5, 'length': 5.0
        },
        'high': {
            'position' : [6.5, 4.0, 0.4],
            'width': wall_thickness,
            'height': 0.8,
            'length': 7.0,
        },
        'low': {
            'position' : [6.5, 4.0, 0.4],
            'width': wall_thickness,
            'height': 0.8,
            'length': 7.0,
        },
    },
    {
        'type': 'box', 
         'geometry': {
             'position': [2.0, 8.0, 0.4], 'width': 4.0, 'height': 5, 'length': wall_thickness
        },
        'high': {
            'position' : [2.0, 8.0, 0.4],
            'width': 4.0,
            'height': 0.8,
            'length': wall_thickness,
        },
        'low': {
            'position' : [2.0, 8.0, 0.4],
            'width': 4.0,
            'height': 0.8,
            'length': wall_thickness,
        },
    },
]

wall_obstacles = [BoxObstacle(name=f"wall_{i}", content_dict=obst_dict) for i, obst_dict in enumerate(wall_obstacles_dicts)]


cylinder_obstacles_dicts = [
    #table1
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [8.0, -8.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table2
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [4.0, -8.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },

    #table3
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [0.0, -8.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table4
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [-4.0, -8.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table5
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [-8.0, -8.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table6
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [6.0, -5.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table7
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [2.0, -5.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table8
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [-2.0, -5.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table9
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [-6.0, -5.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },

    #table10
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [0.0, 8.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table11
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [-4.0, 8.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table12
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [-8.0, 8.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table13
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [-2.0, 5.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    },


    #table14
    {
        "type": "cylinder",
        "movable": False,
        "geometry": {
            "position": [-6.0, 5.0, 0.0],
            "radius": 1.0,
            "height": 1.0,
        },
        "rgba": [0.1, 1.0, 0.3, 1.0],
    }

]

cylinder_obstacles = [CylinderObstacle(name=f"cylinder_{i}", content_dict=obst_dict) for i, obst_dict in enumerate(cylinder_obstacles_dicts)]


box_obstacles_dicts = [
#table15
{
    'type': 'box',
    'movable': False,
    'geometry': {
        'position' : [7.0, 0.0, 0.0],
        'width': 4.0,
        'height': 1.0,
        'length': 3.0,
    },
    "rgba": [0.1, 1.0, 0.5, 1.0],
},

#table16
{
    'type': 'box',
    'movable': False,
    'geometry': {
        'position' : [2.5, 0.0, 0.0],
        'width': 4.0,
        'height': 1.0,
        'length': 2.0,
    },
    "rgba": [0.1, 1.0, 0.5, 1.0],
},

#table17
{
    'type': 'box',
    'movable': False,
    'geometry': {
        'position' : [-2.5, 0.0, 0.0],
        'width': 4.0,
        'height': 1.0,
        'length': 2.0,
    },
    "rgba": [0.1, 1.0, 0.5, 1.0],
},

#table18
{
    'type': 'box',
    'movable': False,
    'geometry': {
        'position' : [-7.0, 0.0, 0.0],
        'width': 4.0,
        'height': 1.0,
        'length': 3.0,
    },
    "rgba": [0.1, 1.0, 0.5, 1.0],
}
]

box_obstacles = [BoxObstacle(name=f"box_{i}", content_dict=obst_dict) for i, obst_dict in enumerate(box_obstacles_dicts)]


dynamic_sphere_obstacles_dicts = [
    {
        "type": "sphere",
        "geometry": {"trajectory": ["4.0 - 0.2 * t", "2.0", "0.2"], "radius": 0.35},
        "rgba": [1.0, 0.2, 0.2, 1.0],
    },
    {
        "type": "sphere",
        "geometry": {"trajectory": ["-2.0", "-2.0 + 0.05 * t", "0.2"], "radius": 0.35},
        "rgba": [1.0, 0.6, 0.1, 1.0],
    },
    {
        "type": "sphere",
        "geometry": {"trajectory": ["0.0 + 0.2 * t", "2.0", "0.2"], "radius": 0.35},
        "rgba": [1.0, 0.2, 0.2, 1.0],
    },
    {
        "type": "sphere",
        "geometry": {"trajectory": ["0.0 + 0.1 * t", "3.0", "0.2"], "radius": 0.35},
        "rgba": [1.0, 0.2, 0.2, 1.0],
    },
]

dynamic_sphere_obstacles = [
    DynamicSphereObstacle(name=f"dyn_sphere_{i}", content_dict=obst_dict)
    for i, obst_dict in enumerate(dynamic_sphere_obstacles_dicts)
]
