import pygame

# --- Setup ---
pygame.init()
screen = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()

# Grid settings
rows, cols = 10, 10
cell_size = 60

# Define some state colors
STATE_COLORS = {
    "healthy": (52, 110, 43),
    "onfire": (0, 255, 0),
    "burnt": (255, 0, 0)
}

# Create a grid with example states
grid = [
    ["empty" for _ in range(cols)] for _ in range(rows)
]

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