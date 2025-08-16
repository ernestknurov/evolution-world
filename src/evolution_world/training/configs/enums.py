from enum import Enum

class Entity(Enum):
    EMPTY = 0
    WALL = 1
    FOOD = 2
    WATER = 3

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    REST = 4
    EAT = 5
    DRINK = 6