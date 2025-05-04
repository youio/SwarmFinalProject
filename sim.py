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
grid[8:17,8:17] = 'onfire'
grid[10:15,10:15] = 'burnt'

# --- Wildfire update function ---
def count_fire_neighbors(r, c):
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # 4-neighbors
    count = 0
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] == "onfire":
                count += 1
    return count

def update_grid():
    new_grid = grid.copy()
    for r in range(rows):
        for c in range(cols):
            state = grid[r][c]
            fire_neighbors = count_fire_neighbors(r, c)
            if state == "healthy" and fire_neighbors > 0:
                prob = 1 - (1 - alpha) ** fire_neighbors
                if random.random() < prob:
                    new_grid[r][c] = "onfire"
            elif state == "onfire":
                if random.random() > beta:
                    new_grid[r][c] = "burnt"
    return new_grid

# --- Main loop ---
tick = 0
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update every 10 frames
    if tick % 10 == 0:
        grid = update_grid()

    screen.fill((200, 200, 200))

    # Draw grid
    for row in range(rows):
        for col in range(cols):
            state = grid[row][col]
            color = STATE_COLORS.get(state, (128, 128, 128))
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # grid lines

    pygame.display.flip()
    clock.tick(30)
    tick += 1

pygame.quit()
