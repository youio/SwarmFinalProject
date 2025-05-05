import pygame
import numpy as np
import random

# --- Setup ---
pygame.init()

rows, cols = 25, 25
cell_size = 30
screen = pygame.display.set_mode((cols*cell_size, rows*cell_size))
clock = pygame.time.Clock()

STATE_COLORS = {
    "healthy": (52, 110, 43),
    "onfire": (237, 117, 57),
    "burnt": (31, 15, 11)
}

alpha = 0.6
beta = 0.8

grid = np.array([["healthy" for _ in range(cols)] for _ in range(rows)])
grid[10:15, 10:15] = 'onfire'
grid[12:13, 12:13] = 'burnt'

def update_grid(wind_vector):
    new_grid = grid.copy()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Normalize direction
    wind_direction = np.array(wind_vector[:2], dtype=float)
    wind_direction /= np.linalg.norm(wind_direction) + 1e-8

    # Extract wind speed
    wind_speed = wind_vector[2]

    if(wind_vector[:2] == (0,0)): # if no wind, set wind speed to 0
        wind_speed = 0

    for r in range(rows):
        for c in range(cols):
            state = grid[r][c]
            if state == "healthy":
                total_prob = 0.0
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "onfire":
                        u = np.array([dr, dc], dtype=float) 
                        u /= np.linalg.norm(u) + 1e-8 # direction from neighbor to current cell
                        proj = np.dot(wind_direction, u) # how wind aligns in cell's direction

                        print(f"projection: " + str(proj))

                        prob_wind_effect = alpha * max(0.2, (1 + proj) * wind_speed) # if proj > 0, wind helps spread. if proj < 0, wind doesn't. pw_f never goes below 0.2
                        total_prob += prob_wind_effect # spread probability

                        print(f"total_prob: " + str(total_prob))

                if random.random() < total_prob: # set fire if spread probability is high enough
                    new_grid[r][c] = "onfire"
            elif state == "onfire":
                if random.random() > beta: # burnt if greater than threshold beta
                    new_grid[r][c] = "burnt"
    return new_grid

class UAVAgent:
    def __init__(self, center, radius, angle, direction=1, cam_size=(5, 5), pc=0.8):
        self.center = np.array(center)
        self.radius = radius
        self.angle = angle
        self.direction = direction
        self.cam_h, self.cam_w = cam_size
        self.pc = pc
        self.belief = np.full((rows, cols), 1/3)
        self.path = []
        self.update_position()

    def update_position(self):
        offset = self.radius * np.array([np.sin(self.angle), np.cos(self.angle)])
        pos = self.center + offset
        self.pos = np.clip(pos, [0, 0], [rows - 1, cols - 1])
        self.path.append(tuple(self.pos.astype(int)))

    def move(self):
        self.angle += self.direction * 0.05
        self.update_position()

    def observe(self, env_grid):
        center_r, center_c = self.pos.astype(int)
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
                    if obs == "healthy":
                        self.belief[r][c] = 0.9
                    elif obs == "onfire":
                        self.belief[r][c] = 0.5
                    else:
                        self.belief[r][c] = 0.1

def share_beliefs(agents):
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            dist = np.linalg.norm(agents[i].pos - agents[j].pos)
            if dist < 2.0:
                shared = (agents[i].belief + agents[j].belief) / 2
                agents[i].belief = shared.copy()
                agents[j].belief = shared.copy()

# --- Initialize agents in circular paths ---
agents = [UAVAgent(center=(12, 12), radius=8, angle=np.pi/2 * i, direction=1) for i in range(2)] + [UAVAgent(center=(12, 12), radius=8, angle=np.pi/2 * i, direction=-1) for i in range(2, 4)]

# --- Main loop ---
tick = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if tick % 10 == 0:
        grid = update_grid(wind_vector=(0, 0, 10)) # CHANGE WIND DIRECTION AND SPEED HERE ((direction), speed)
                                                    # 

    if tick % 5 == 0:
        for agent in agents:
            agent.move()
            agent.observe(grid)
        share_beliefs(agents)

    screen.fill((200, 200, 200))
    for row in range(rows):
        for col in range(cols):
            state = grid[row][col]
            color = STATE_COLORS.get(state, (128, 128, 128))
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    for agent in agents:
        for i in range(1, len(agent.path)):
            r1, c1 = agent.path[i - 1]
            r2, c2 = agent.path[i]
            pygame.draw.line(
                screen, (255, 255, 255),
                (c1 * cell_size + cell_size // 2, r1 * cell_size + cell_size // 2),
                (c2 * cell_size + cell_size // 2, r2 * cell_size + cell_size // 2), 2
            )
        ar, ac = agent.pos
        pygame.draw.circle(
            screen, (0, 0, 255),
            (int(ac * cell_size + cell_size // 2), int(ar * cell_size + cell_size // 2)),
            cell_size // 3
        )

    pygame.display.flip()
    clock.tick(30)
    tick += 1

pygame.quit()
