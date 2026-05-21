# The random seed for numpy.
# Setting this to 0 means that it will be different each time.
RANDOM_SEED = 0

# The window width and height in pixels, for both the "environment" window and the "planning" window.
# If you wish, you can modify this according to the size of your screen.
WINDOW_SIZE = 800

# The environment type, which determines the positions of the robot's initial state, the goal state, and the obstacle.
# Setting it to 'fixed' means that the environment is always a pre-defined, relatively simple environment.
# Setting it to 'random' means that the environment will be randomly generated on each episode.
ENVIRONMENT_TYPE = 'random'  # Options are: 'fixed', 'random'

# The frame rate for pygame, which determines how quickly the program runs.
# Specifically, this is the number of time steps per second that the robot will execute an action in the environment.
# You may wish to slow this down to observe the robot's movement, or speed it up to run large-scale experiments.
FRAME_RATE = 30

# You may want to add your own configuration variables here, depending on the algorithm you implement.
NUM_DEMOS = 5
EPISODE_LENGTH = 100
MINIBATCH_SIZE = 20
NUM_MINIBATCH = 100
LEARNING_RATE = 0.005
NUM_TRAINING_EPOCHS = 500
