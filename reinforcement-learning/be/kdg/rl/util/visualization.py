import matplotlib.pyplot as plt
import random
import numpy as np


def draw_quiver(agent, save=False, path="quiver.png", picture=False):
    # coordinates of the arrows
    # subtract .5 to center the arrows or else they will be drawn on the edge of each tile
    # add .5 to the first coordinate to center the arrows on the tile
    # 4x4 grid --> range from .5 to 3.5
    x = [.5, 1.5, 2.5, 3.5, .5, 2.5, .5, 1.5, 2.5, 1.5, 2.5]
    y = [3.5, 3.5, 3.5, 3.5, 2.5, 2.5, 1.5, 1.5, 1.5, .5, .5]

    # direction of the arrows
    x_dir = []
    y_dir = []

    # All squares that end in a hole (-1 reward) or the goal (1 reward)
    holes = [5, 7, 11, 12, 15]

    for state in range(agent.env.state_size):
        if state in holes:
            pass
        else:
            # Get the best action for this state
            # Get all the actions for this state
            actions = agent.learning_strategy.π[:, state]
            # Get the index of the best action
            max_values = np.argwhere(actions == np.amax(actions))
            # Get the best action with a random index if there are multiple best actions
            best_move = random.choice(max_values.flatten().tolist())

            # Get the direction of the arrow
            # 0 = left, 1 = down, 2 = right, 3 = up
            if 0 == best_move:
                x_dir.append(-1)
                y_dir.append(0)
            elif 1 == best_move:
                x_dir.append(0)
                y_dir.append(-1)
            elif 2 == best_move:
                x_dir.append(1)
                y_dir.append(0)
            else:
                x_dir.append(0)
                y_dir.append(1)

    # Only show grid lines at whole numbers
    # plt.xticks(np.arange(0, 4, 1))
    # plt.yticks(np.arange(0, 4, 1))

    # Set the grid lines
    # plt.grid()

    plt.quiver(x, y, x_dir, y_dir, scale=4, scale_units='xy')

    # If picture is added in parameter
    if picture:
        img = plt.imread(picture)
        # Resize it to fit the grid (4x4)
        plt.imshow(img, extent=[0, 4, 0, 4])

    # Remove the ticks
    plt.tick_params(
        axis='both',  # changes apply to the x-axis and y-axis
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the left edge are off
        labelbottom=False,  # labels along the bottom edge are off
        labelleft=False)  # labels along the left edge are off

    if save:
        plt.savefig(path)

    plt.show()
