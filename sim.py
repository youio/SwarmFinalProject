import pygame
import numpy as np
import random

# --- Setup ---
pygame.init()

# Grid settings
rows, cols = 25, 25
cell_size = 30

screen = pygame.display.set_mode((cols*cell_size, rows*cell_size))
clock = pygame.time.Clock()

# Define some state colors
STATE_COLORS = {
    "healthy": (52, 110, 43),
    "onfire": (237, 117, 57),
    "burnt": (31, 15, 11)
}

# Transition parameters
alpha = 0.4  # susceptibility to catching fire
beta = 0.6   # chance to stay on fire

# Create a grid with initial states
grid = np.array([["healthy" for _ in range(cols)] for _ in range(rows)])
grid[8:17, 8:17] = 'onfire'
grid[10:15, 10:15] = 'burnt'

# --- Wildfire update function ---
def update_grid(wind_vector):
    new_grid = grid.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    wind_vector = np.array(wind_vector, dtype=float)
    wind_vector /= np.linalg.norm(wind_vector) + 1e-8  # normalize

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
                        proj = np.dot(wind_vector, u)
                        pw_f = alpha * (1 + proj)
                        total_prob += pw_f

                if random.random() < total_prob:
                    new_grid[r][c] = "onfire"

            elif state == "onfire":
                if random.random() > beta:
                    new_grid[r][c] = "burnt"

    return new_grid

# --- Agent model ---
class UAVAgent:
    def __init__(self, pos, cam_size=(5, 5), pc=0.8):
        self.pos = np.array(pos)
        self.cam_h, self.cam_w = cam_size
        self.pc = pc
        self.belief = np.full((rows, cols), 1/3)  # uniform initial belief
        self.path = [tuple(self.pos)]

    def move(self):
        directions = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]
        dr, dc = random.choice(directions)
        new_pos = self.pos + np.array([dr, dc])
        self.pos = np.clip(new_pos, [0, 0], [rows-1, cols-1])
        self.path.append(tuple(self.pos))

    def observe(self, env_grid):
        center_r, center_c = self.pos
        for dr in range(-self.cam_h // 2, self.cam_h // 2 + 1):
            for dc in range(-self.cam_w // 2, self.cam_w // 2 + 1):
                r = center_r + dr
                c = center_c + dc
                if 0 <= r < rows and 0 <= c < cols:
                    true_state = env_grid[r][c]
                    if random.random() < self.pc:
                        obs = true_state
                    else:
                        obs = random.choice(
                            [s for s in ["healthy", "onfire", "burnt"] if s != true_state])
                    print(f"  cell ({r},{c}): true={true_state}, obs={obs}")
                    
                    # Approximate Bayesian update: crude method
                    if obs == "healthy":
                        self.belief[r][c] = 0.9
                    elif obs == "onfire":
                        self.belief[r][c] = 0.5
                    else:
                        self.belief[r][c] = 0.1

# --- Initialize agents ---
agents = [
    UAVAgent(pos=(5, 5)),
    UAVAgent(pos=(15, 5)),
    UAVAgent(pos=(5, 20)),
    UAVAgent(pos=(20, 20))
]

# --- Main loop ---
tick = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update wildfire every 10 frames
    if tick % 10 == 0:
        grid = update_grid(wind_vector=(1, 0))  # wind blowing down

    # Update agents
    for agent in agents:
        if tick % 20 == 0:  # slower update rate for agents
            agent.move()
        agent.observe(grid)

    # Draw background grid
    screen.fill((200, 200, 200))
    for row in range(rows):
        for col in range(cols):
            state = grid[row][col]
            color = STATE_COLORS.get(state, (128, 128, 128))
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    # Draw agents and their paths
    for agent in agents:
        # Draw path
        for i in range(1, len(agent.path)):
            r1, c1 = agent.path[i - 1]
            r2, c2 = agent.path[i]
            pygame.draw.line(
                screen, (255, 255, 255),
                (c1 * cell_size + cell_size // 2, r1 * cell_size + cell_size // 2),
                (c2 * cell_size + cell_size // 2, r2 * cell_size + cell_size // 2), 2
            )
        # Draw agent
        ar, ac = agent.pos
        pygame.draw.circle(
            screen, (0, 0, 255),
            (ac * cell_size + cell_size // 2, ar * cell_size + cell_size // 2),
            cell_size // 3
        )

    pygame.display.flip()
    clock.tick(30)
    tick += 1

pygame.quit()
