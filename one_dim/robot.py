# Imports from external libraries
import numpy as np

# Imports from this project
import one_dim.constants as constants
import one_dim.config as config


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self):
        # A list of visualisations which will be displayed on the right-side of the window
        self.visualisations = []
        # The number of steps in the episode so far
        self.num_steps = 0

    # Reset the robot at the start of an episode
    def reset(self):
        self.num_steps = 0
        self.visualisations = []

    # Get the next action
    def select_action(self, state):
        # An action which moves the robot diagonally up and right
        action = np.array([0.5 * constants.MAX_ATION_MAGNITUDE])
        action = np.random.uniform(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE, size=(1,))
        # Currently, the episode never ends
        episode_done = False
        # Increment the number of steps executed so far
        self.num_steps += 1
        # Return the action, and a flag indicatingC if the episode has finished, to the main program loop
        return action, episode_done


# The VisualisationLine class enables us to store a line segment which will be drawn to the screen
class Visualisation:
    # Initialise a new visualisation (a new line)
    def __init__(self, x1, y1, x2, y2, colour=(255, 255, 255), width=0.01):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width
