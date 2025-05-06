import pygame
import numpy as np
import random
import math

# --- Setup ---

rows, cols = 25, 25
cell_size = 30


STATE_COLORS = {
    "healthy": (52, 110, 43),
    "onfire": (214, 45, 45),
    "burnt": (31, 15, 11)
}

alpha = 0.07 # for setting on fire
beta = 1 # for remaining on fire/turning burnt
wind_vector = (1, 1, 0.5)

# Shared patrol parameters
patrol_center = np.array([12, 12], dtype=float)
patrol_radius = 8


grid = np.array([["healthy" for _ in range(cols)] for _ in range(rows)])
grid[10:15, 10:15] = 'onfire'
grid[12:13, 12:13] = 'burnt'

def update_grid(wind_vector):
    '''Updates wildfire spread'''
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

                        # print(f"projection: " + str(proj))

                        prob_wind_effect = alpha * (0.7 + proj * wind_speed)  # if proj > 0, wind helps spread. if proj < 0, wind doesn't. pw_f never goes below 0.2
                        total_prob += prob_wind_effect # spread probability

                        # print(f"total_prob: " + str(total_prob))

                if random.random() < total_prob: # set fire if spread probability is high enough
                    new_grid[r][c] = "onfire"
            elif state == "onfire":
                if random.random() > beta: # burnt if greater than threshold beta
                    new_grid[r][c] = "burnt"
    return new_grid

# class UAVAgent:
#     '''Initialize UAV Agents'''
#     def __init__(self, center, radius, angle, direction=1, cam_size=(5, 5), pc=0.8):
#         self.center = np.array(center)
#         self.radius = radius
#         self.angle = angle
#         self.direction = direction
#         self.cam_h, self.cam_w = cam_size
#         self.pc = pc
#         self.belief = np.full((rows, cols), 1/3)
#         self.path = []
#         self.update_position()

#     def update_position(self):
#         offset = self.radius * np.array([np.sin(self.angle), np.cos(self.angle)])
#         self.pos = np.clip(self.center + offset, [0, 0], [rows - 1, cols - 1])
#         self.path.append(tuple(self.pos.astype(int)))

#     def move(self):
#         '''Move UAV in circular direction'''
#         self.angle += self.direction * 0.05
#         self.update_position()

#     def observe(self, env_grid):
#         '''Determine UAV observation'''
#         center_r, center_c = self.pos.astype(int)
#         for dr in range(-self.cam_h // 2, self.cam_h // 2 + 1):
#             for dc in range(-self.cam_w // 2, self.cam_w // 2 + 1):
#                 r = center_r + dr
#                 c = center_c + dc
#                 if 0 <= r < rows and 0 <= c < cols:
#                     true_state = env_grid[r][c]
#                     if random.random() < self.pc:
#                         obs = true_state
#                     else:
#                         obs = random.choice(
#                             [s for s in ["healthy", "onfire", "burnt"] if s != true_state])
#                     if obs == "healthy":
#                         self.belief[r][c] = 0.9
#                     elif obs == "onfire":
#                         self.belief[r][c] = 0.5
#                     else:
#                         self.belief[r][c] = 0.1

class UAVAgent():
    def __init__(self, init_pos, rotation, orientation, pc=0.95):
        self.pos = init_pos
        self.rotation = rotation
        self.orientations = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])
        self.ori = orientation
        self.belief = grid.copy()
        self.path = []
        self.pc = pc
        self.cam_h, self.cam_w = (3,3)

    def move(self):
        o = self.ori
        r = self.rotation

        # check forward
        if grid[*(self.orientations[o]+self.pos)] != 'healthy':
            # self.pos += self.orientations[(o+r)%4]
            self.ori = (self.ori+r)%4
        elif grid[*(self.orientations[(o-r)%4]+self.pos)] == 'healthy':
            self.ori = (self.ori-r)%4
            self.pos += self.orientations[(o-r)%4]
        else:
            self.pos += self.orientations[o]


    def observe(self, env_grid):
        r_center,c_center = self.pos
        for dr in range(-self.cam_h // 2, self.cam_h // 2 + 1):
            for dc in range(-self.cam_w // 2, self.cam_w // 2 + 1):
                r = r_center + dr
                c = c_center + dc
                if 0 <= r < rows and 0 <= c < cols:
                    true_state = env_grid[r][c]
                    if random.random() < self.pc:
                        obs = true_state
                    else:
                        obs = random.choice(
                            [s for s in ["healthy", "onfire", "burnt"] if s != true_state])
                        
                    self.belief[r,c] = obs 


def share_beliefs(agents):
    '''Share UAV beliefs when in close proximity'''
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            print('pose', agents[i].pos, agents[j].pos)
            dist = np.linalg.norm(agents[i].pos - agents[j].pos)
            if dist < 2.0:
                # shared = (agents[i].belief + agents[j].belief) / 2
                fire1 = np.where(agents[i].belief == 'onfire')
                fire2 = np.where(agents[j].belief == 'onfire')

                agents[i].belief[fire2] = 'onfire'
                agents[j].belief[fire1] = 'onfire'

# --- Create N UAVs evenly spaced around the fire center ---
# def runsim(timesteps=500, num_uavs=2):
pygame.init()
screen = pygame.display.set_mode((cols*cell_size, rows*cell_size))
clock = pygame.time.Clock()

timesteps = 500
center = (13, 13)
radius = 3.5
cx,cy = center
num_uavs = 5
agents = []

# Spawn agents
for i in range(num_uavs):

    angle = 2 * math.pi * i / num_uavs
    if angle < math.pi/4 or angle >= 7*math.pi/4:
        orientation = 0
    elif angle < 3*math.pi/4:
        orientation = 1
    elif angle < 5*math.pi/4:
        orientation = 2
    else:
        orientation = 3
    
    if i %2 == 0:
        rotation = 1
    else:
        #flip direction of rotation
        rotation =-1
        orientation = (orientation+2)%4
    
    x = cx + radius * math.cos(angle)
    y = cy + radius * math.sin(angle)
    grid_x, grid_y = round(x), round(y)
    agents.append(UAVAgent(init_pos = np.array([grid_x, grid_y]), rotation=rotation,orientation=orientation ))
    
representative = agents[0]

accuracies = []
coverages = []
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --- Main loop ---
tick = 0
running = True
while tick <= timesteps and running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if tick % 10 == 0:
        grid = update_grid(wind_vector) # CHANGE WIND DIRECTION AND SPEED HERE ((direction), speed)
                                                    # 
    
    if tick % 5 == 0:
        for agent in agents:
            agent.move()
            agent.observe(grid)
        share_beliefs(agents)
        
    # for agent in agents:
        
    
    
    front = []
    fires = np.array(np.where(grid == 'onfire')).T
    # print('fires', fires.shape)


    for fire in fires:
        if np.sum([grid[max(min(fire[0]+d[0],cols-1 ), 0), max(min(fire[1]+d[1],cols-1 ), 0)] == 'healthy' for d in directions]) > 0:
            front.append(fire)
    front_coverage = np.sum([representative.belief[f[0],f[1]] == 'onfire' for f in front])/len(front)
    print(len([representative.belief[f] == 'onfire' for f in front]), len(front))
    # for f in front:
    #     print(f)
    coverages.append(front_coverage)
    

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

print('average coverage for this sim:', np.mean(coverages))

pygame.quit()

# if __name__ == '__main__':
#     runsim()