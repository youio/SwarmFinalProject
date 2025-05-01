import pygame
import numpy as np

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

# Create a grid with example states
grid = np.array([
    ["healthy" for _ in range(cols)] for _ in range(rows)
])
grid[8:17,8:17] = 'onfire'
grid[10:15,10:15] = 'burnt'
# --- Main loop ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

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
    clock.tick(60)

pygame.quit()