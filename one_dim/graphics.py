# Imports from external libraries
import pygame

# Imports from this project
import one_dim.constants as constants
import one_dim.config as config


# The Graphics class performs all the pygame drawing
class Graphics:

    # Initialisation of new graphics
    def __init__(self):
        pygame.init()
        track_height = config.WINDOW_SIZE // 4
        # Screen dimensions
        self.screen = pygame.display.set_mode((2 * config.WINDOW_SIZE + constants.WINDOW_MARGIN, track_height + constants.WINDOW_HEADER))
        # The left-side and right-side canvases which make up the full window
        self.left_canvas = pygame.Surface((config.WINDOW_SIZE, track_height))
        self.right_canvas = pygame.Surface((config.WINDOW_SIZE, track_height))
        # Set a window title
        pygame.display.set_caption("Robot Learning")
        # Clock to control the frame rate
        self.clock = pygame.time.Clock()

    # Function to draw the environment, and any visualisations, on the window
    def draw(self, environment, visualisations):
        # Clear the screen
        self.screen.fill((0, 0, 0))
        # Draw the text along the top
        font = pygame.font.SysFont('Arial', 30)
        text_surface = font.render('Environment', True, (255, 255, 255))
        self.screen.blit(text_surface, (0.4 * config.WINDOW_SIZE, 0.2 * constants.WINDOW_HEADER))
        font = pygame.font.SysFont('Arial', 30)
        text_surface = font.render('Visualisations', True, (255, 255, 255))
        self.screen.blit(text_surface, (1.4 * config.WINDOW_SIZE, 0.2 * constants.WINDOW_HEADER))
        # Draw the left side (just the environment)
        self.left_canvas.fill((0, 0, 0))
        self.draw_environment(environment, self.left_canvas)
        self.draw_init_and_goal_states(environment, self.left_canvas)
        self.draw_robot(environment, self.left_canvas)
        self.screen.blit(self.left_canvas, (0, constants.WINDOW_HEADER))
        # Draw the right side (the environment plus any visualisations)
        self.right_canvas.fill((0, 0, 0))
        self.draw_environment(environment, self.right_canvas)
        self.draw_visualisations(visualisations, self.right_canvas)
        self.draw_init_and_goal_states(environment, self.right_canvas)
        self.draw_robot(environment, self.right_canvas)
        self.screen.blit(self.right_canvas, (config.WINDOW_SIZE + constants.WINDOW_MARGIN, constants.WINDOW_HEADER))
        # Update the display
        pygame.display.flip()
        # Tick the clock, i.e. wait for one step of the environment
        self.clock.tick(config.FRAME_RATE)

    # Function to draw the environment
    def draw_environment(self, environment, canvas):
        track_height = config.WINDOW_SIZE // 4
        pygame.draw.rect(canvas, (200, 200, 200), pygame.Rect(0, 0, config.WINDOW_SIZE, track_height), 5)

    # Function to draw the initial and goal states
    def draw_init_and_goal_states(self, environment, canvas):
        # Draw the init state
        position = self.world_pos_to_window_pos(environment.init_state)
        radius = self.world_len_to_window_len(constants.ROBOT_RADIUS)
        pygame.draw.circle(canvas, constants.ROBOT_INIT_COLOUR, position, radius)
        # Draw the goal state
        position = self.world_pos_to_window_pos(environment.goal_state)
        radius = self.world_len_to_window_len(constants.GOAL_RADIUS)
        pygame.draw.circle(canvas, constants.GOAL_COLOUR, position, radius)

    # Function to draw the robot
    def draw_robot(self, environment, canvas):
        # Draw the robot
        position = self.world_pos_to_window_pos(environment.state)
        radius = self.world_len_to_window_len(constants.ROBOT_RADIUS)
        pygame.draw.circle(canvas, constants.ROBOT_COLOUR, position, radius)

    # Function to draw any visualisations
    def draw_visualisations(self, visualisations, canvas):
        for visualisation in visualisations:
            # For each visualisation, get the attributes necessary to create a pygame line
            start_pos = self.world_pos_to_window_pos([visualisation.x1, visualisation.y1])
            end_pos = self.world_pos_to_window_pos([visualisation.x2, visualisation.y2])
            width = self.world_len_to_window_len(visualisation.width)
            pygame.draw.line(canvas, visualisation.colour, start_pos, end_pos, width)

    # Function to covert a position in world/environment space, to a position in pixels on the window
    def world_pos_to_window_pos(self, world_pos):
        track_height = config.WINDOW_SIZE // 4
        lo, hi = constants.ENV_BOUNDS
        x = int(config.WINDOW_SIZE * (world_pos[0] - lo) / (hi - lo))
        y = track_height // 2
        return x, y

    # Function to convert a length in world/environment space, to a length in pixels on the window
    def world_len_to_window_len(self, world_length):
        window_length = int(config.WINDOW_SIZE * world_length)
        return window_length
