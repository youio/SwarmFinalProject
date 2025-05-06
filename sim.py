import pygame
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# --- Setup ---
rows, cols = 25, 25
cell_size = 30

# Define color for each fire state
STATE_COLORS = {
    "healthy": (52, 110, 43),
    "onfire": (214, 45, 45),
    "burnt": (31, 15, 11)
}

alpha = 0.07  # Probability base for catching fire
beta = 1      # Probability of staying on fire before burning out
wind_vector = (1, 1, 0.5)  # Default wind direction and strength

class UAVAgent():
    '''
    UAV agent that moves in a direction, observes its surroundings,
    and updates its belief about the fire state of cells.
    '''
    def __init__(self, init_pos, rotation, orientation, pc=0.95):
        self.pos = init_pos
        self.rotation = rotation  # 1 for clockwise, -1 for counterclockwise
        self.orientations = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])  # N, E, S, W
        self.ori = orientation
        self.belief = None
        self.path = []
        self.pc = pc  # Probability of correct observation
        self.cam_h, self.cam_w = (3, 3)  # Camera size

    def move(self, env_grid):
        '''Moves the UAV according to fire state and turning logic.'''
        o = self.ori
        r = self.rotation
        forward_pos = self.orientations[o] + self.pos
        side_pos = self.orientations[(o - r) % 4] + self.pos

        if in_bounds(forward_pos, env_grid.shape):
            if env_grid[tuple(forward_pos)] != 'healthy':
                self.ori = (self.ori + r) % 4
            elif env_grid[tuple(side_pos)] == 'healthy':
                self.ori = (self.ori - r) % 4
                self.pos += self.orientations[(o - r) % 4]
            else:
                self.pos += self.orientations[o]

    def observe(self, env_grid):
        '''UAV observes its 3x3 neighborhood and updates its belief.'''
        r_center, c_center = self.pos
        for dr in range(-self.cam_h // 2, self.cam_h // 2 + 1):
            for dc in range(-self.cam_w // 2, self.cam_w // 2 + 1):
                r, c = r_center + dr, c_center + dc
                if 0 <= r < rows and 0 <= c < cols:
                    true_state = env_grid[r][c]
                    obs = true_state if random.random() < self.pc else random.choice([s for s in ["healthy", "onfire", "burnt"] if s != true_state])
                    self.belief[r, c] = obs

def in_bounds(pos, grid_shape):
    '''Check if a position is within the grid bounds.'''
    x, y = pos
    return 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1]

def update_grid(grid, wind_vector):
    '''Update the environment grid based on fire spread and wind.'''
    new_grid = grid.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E

    wind_direction = np.array(wind_vector[:2], dtype=float)
    wind_direction /= np.linalg.norm(wind_direction) + 1e-8
    wind_speed = 0 if wind_vector[:2] == (0, 0) else wind_vector[2]

    for r in range(rows):
        for c in range(cols):
            state = grid[r][c]
            if state == "healthy":
                total_prob = 0.0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "onfire":
                        u = np.array([dr, dc], dtype=float)
                        u /= np.linalg.norm(u) + 1e-8
                        proj = np.dot(wind_direction, u)
                        prob_wind_effect = alpha * (0.7 + proj * wind_speed)
                        total_prob += prob_wind_effect
                if random.random() < total_prob:
                    new_grid[r][c] = "onfire"
            elif state == "onfire":
                if random.random() > beta:
                    new_grid[r][c] = "burnt"
    return new_grid

def share_beliefs(agents):
    '''Share fire beliefs between agents that are close to each other.'''
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            dist = np.linalg.norm(agents[i].pos - agents[j].pos)
            if dist < 2.0:
                fire1 = np.where(agents[i].belief == 'onfire')
                fire2 = np.where(agents[j].belief == 'onfire')
                agents[i].belief[fire2] = 'onfire'
                agents[j].belief[fire1] = 'onfire'

def sim_step(tick, agents, grid, wind_vector):
    '''Performs one step of the simulation.'''
    if tick % 10 == 0:
        grid = update_grid(grid, wind_vector)

    if tick % 5 == 0:
        for agent in agents:
            agent.move(grid)
            agent.observe(grid)
        share_beliefs(agents)
    return grid

def runsim(timesteps=500, num_uavs=2, wind_vector=(1, 1, 0.5), render=False):
    '''
    Runs the wildfire simulation.
    
    Args:
        timesteps: number of ticks to simulate
        num_uavs: number of UAV agents
        wind_vector: (dx, dy, speed) tuple for wind
        render: whether to visualize the simulation in real-time
    Returns:
        Average front coverage observed by representative UAV
    '''
    # Create environment grid
    grid = np.array([["healthy" for _ in range(cols)] for _ in range(rows)])
    grid[10:15, 10:15] = 'onfire'
    grid[12:13, 12:13] = 'burnt'

    if render:
        pygame.init()
        screen = pygame.display.set_mode((cols * cell_size, rows * cell_size))
        clock = pygame.time.Clock()

    # Initialize UAVs around center
    center = (13, 13)
    radius = 2.5
    cx, cy = center
    agents = []

    for i in range(num_uavs):
        angle = 2 * math.pi * i / num_uavs
        if angle < math.pi / 4 or angle >= 7 * math.pi / 4:
            orientation = 0
        elif angle < 3 * math.pi / 4:
            orientation = 1
        elif angle < 5 * math.pi / 4:
            orientation = 2
        else:
            orientation = 3

        rotation = 1 if i % 2 == 0 else -1
        if rotation == -1:
            orientation = (orientation + 2) % 4

        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        grid_x, grid_y = round(x), round(y)
        agents.append(UAVAgent(init_pos=np.array([grid_x, grid_y]), rotation=rotation, orientation=orientation))

    for agent in agents:
        agent.belief = grid.copy()

    # Representative for measuring coverage
    representative = agents[0]
    coverages = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    tick = 0
    running = True
    while tick <= timesteps and running:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

        # Simulation step
        grid = sim_step(tick, agents, grid, wind_vector)

        # Track front coverage
        front = []
        fires = np.array(np.where(grid == 'onfire')).T
        for fire in fires:
            if np.sum([grid[max(min(fire[0] + d[0], cols - 1), 0), max(min(fire[1] + d[1], cols - 1), 0)] == 'healthy' for d in directions]) > 0:
                front.append(fire)

        if front:
            front_coverage = np.sum([representative.belief[f[0], f[1]] == 'onfire' for f in front]) / len(front)
            coverages.append(front_coverage)

        # Render environment and UAVs
        if render:
            screen.fill((200, 200, 200))
            for row in range(rows):
                for col in range(cols):
                    state = grid[row][col]
                    color = STATE_COLORS.get(state, (128, 128, 128))
                    rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, color, rect)
                    pygame.draw.rect(screen, (0, 0, 0), rect, 1)

            for agent in agents:
                ar, ac = agent.pos
                pygame.draw.circle(screen, (0, 0, 255),
                                   (int(ac * cell_size + cell_size // 2), int(ar * cell_size + cell_size // 2)),
                                   cell_size // 3)

            pygame.display.flip()
            clock.tick(30)

        tick += 1

    if render:
        pygame.quit()

    return np.mean(coverages)


def wind_speed_experiment():
    '''Experiment to test coverage spread with fixed direction, fixed swarm size, variable wind speed'''
    wind_speeds = [0.0, 0.5, 1.0, 1.5]
    trials = 5
    fixed_direction = (1, 1)
    num_uavs = 4

    means = []
    stds = []

    for speed in wind_speeds:
        print(f"\nRunning for wind speed {speed}")
        trial_results = []
        for t in range(trials):
            print(f"  Trial {t+1}")
            cov = runsim(num_uavs=num_uavs, wind_vector=(fixed_direction[0], fixed_direction[1], speed), render=False)
            trial_results.append(cov)
        means.append(np.mean(trial_results))
        stds.append(np.std(trial_results))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(wind_speeds, means, yerr=stds, fmt='-o', capsize=5)
    plt.title("Experiment 1: Coverage vs Wind Speed\nDirection = (1,1), 4 UAVs")
    plt.xlabel("Wind Speed")
    plt.ylabel("Average Front Coverage")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def swarm_size_experiment():
    '''Experiment to test coverage with fixed wind speed and direction, increasing swarm size'''
    uav_counts = []
    for i in range(50):
        uav_counts.append(i+1)
    print(uav_counts)
    trials = 5
    wind_vec = (1, 1, 1.0)  # fixed wind

    means = []
    stds = []

    for n_uavs in uav_counts:
        print(f"\nRunning with {n_uavs} UAVs")
        trial_results = []
        for t in range(trials):
            print(f"  Trial {t+1}")
            cov = runsim(num_uavs=n_uavs, wind_vector=wind_vec, render=False)
            trial_results.append(cov)
        means.append(np.mean(trial_results))
        stds.append(np.std(trial_results))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(uav_counts, means, yerr=stds, fmt='-o', capsize=5)
    plt.title("Experiment 2: Coverage vs Number of UAVs\nWind = (1,1), Speed = 1.0")
    plt.xlabel("Number of UAVs")
    plt.ylabel("Average Front Coverage")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def wind_direction_experiment():
    '''Experiment to test coverage with fixed swarm and wind speed, varying wind direction'''
    wind_directions = [
        ((1, 0), "East"),
        ((0, 1), "South"),
        ((-1, 0), "West"),
        ((0, -1), "North"),
        ((1, 1), "Southeast"),
        ((-1, -1), "Northwest")
    ]
    wind_speed = 1.0
    num_uavs = 4
    trials = 5

    means = []
    stds = []
    labels = []

    for (direction, label) in wind_directions:
        print(f"\nRunning with wind direction {label}")
        trial_results = []
        for t in range(trials):
            print(f"  Trial {t+1}")
            wind_vec = (direction[0], direction[1], wind_speed)
            cov = runsim(num_uavs=num_uavs, wind_vector=wind_vec, render=False)
            trial_results.append(cov)
        means.append(np.mean(trial_results))
        stds.append(np.std(trial_results))
        labels.append(label)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.errorbar(labels, means, yerr=stds, fmt='-o', capsize=5)
    plt.title("Experiment 3: Coverage vs Wind Direction\nSpeed = 1.0, 4 UAVs")
    plt.xlabel("Wind Direction")
    plt.ylabel("Average Front Coverage")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def swarmsize_windspeed_experiment():
    '''Experiment to test coverage with increasing swarm size and wind speed, fixed direction'''
    uav_counts = []
    for i in range(20):
        uav_counts.append(i + 1)
    trials = 5
    wind_speeds = [0.0, 0.5, 1.0, 1.5, 2.0]
    fixed_direction = (1, 0)

    # Store mean results for each wind speed
    wind_results = {ws: [] for ws in wind_speeds}

    for ws in wind_speeds:
        for n_uavs in uav_counts:
            print(f"\nRunning with {n_uavs} UAVs at wind speed {ws}")
            trial_results = []
            for t in range(trials):
                print(f"  Trial {t+1}")
                cov = runsim(num_uavs=n_uavs, wind_vector=(fixed_direction[0], fixed_direction[1], ws), render=False)
                trial_results.append(cov)
            avg_cov = np.mean(trial_results)
            wind_results[ws].append(avg_cov)

    # Plot
    plt.figure(figsize=(10, 6))
    for ws in wind_speeds:
        plt.plot(uav_counts, wind_results[ws], label=f"Wind Speed {ws} m/s")  # No capsize here!

    plt.title("Coverage vs UAV Count at Different Wind Speeds")
    plt.xlabel("Number of UAVs")
    plt.ylabel("Average Front Coverage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # wind_speed_experiment()
    # swarm_size_experiment()
    # wind_direction_experiment()
    swarmsize_windspeed_experiment()
    # runsim(render=True) # to see visualization, set render=True
