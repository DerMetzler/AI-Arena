import random
import math
import json
import os
from types import LambdaType
# Function to generate random colors
def generate_random_color():
    return [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

random.seed()

LOWEST_AVAILABLE_ID = 0

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
BALL_RADIUS = 20
MAX_SPEED = 5
COLLIDE_SPEED =2.5
ACCEL = 0.05
CENTER_X = SCREEN_WIDTH/2
CENTER_Y = SCREEN_HEIGHT/2

KILL_FRAME_TIME = 30 
BREED_FRAME_TIME = 50 

FLIP_SIDES = (random.randint(0,1)-0.5)*2

# Colors
WHITE = [255, 255, 255]
RED = [255, 0, 0]
ORANGE = [255,165,0]
GREEN= [0, 255, 0]
BLUE = [0,0,255]
VIOLET = [255,0,255]
TURQOISE = [0,255,255]
BLACK= [0, 0, 0]
ARENA_COLOR=[100,100,100]

#NO1BOT = ["no1bot", RED]
#NO2BOT = ["no2bot", ORANGE]
#LAMEBOT = ["lamebot", WHITE]
#WILDBOT = ["wildbot", GREEN]
#CHALLENGER = ["challenger", BLUE]
#NEXTGENBOT = ["nextgenbot",VIOLET]
#OSKARBOT = ["oskarbot",TURQOISE]
#AGGROBOT = ["aggrobot",BLACK]

#CONTESTANTS = [ OSKARBOT, NEXTGENBOT, CHALLENGER, NO2BOT, AGGROBOT, AGGROBOT, AGGROBOT, AGGROBOT]



FRAME_TIME = 10
RESTART_TIME = 1000

with open("nn_options.txt") as f: 
    temp = json.load(f)
LAYER_SIZE = tuple(temp['layer_size'])
LEARNING_RATE = temp['learning_rate']

print("\n\nNewly created bots will have the architecture: " + str(LAYER_SIZE))
print("The general learning rate is set to: " + str(LEARNING_RATE))
print("You can change these options in 'nn_options.txt'.")

#MS_REMEMBERED =1000
#FRAMES_REMEMBERED = MS_REMEMBERED / FRAME_TIME

MODE = 'n'

while MODE == 'n' or MODE == 'N':
    print("\n\nModes: \n(1): FFA with real-time selection \n(2): 1v1 copilot training \n(3): FFA with score based selection \n(4): 4v4 teamfight \n(N): Infos on NN options.")
    print("Mode selection: ")
    MODE =input()
    if (MODE == 'n' or MODE == 'N'):
        print("The input layer has to have size 8, the output layer has to have size 2.")
        print("Tried out architectures are [8, 16, 2] and [8, 16, 8, 2].")
        print("The second option does mean that it takes longer to arrive at a decent bot, \nbut also that the skill ceiling for bots is noticeably raised.")
        print("A learning rate of roughly 0.05 is recommended to first get a bot to semi-work at all. \nLater, a learning rate of 0.01 seems to work well for fine-tuning.")

MODE = int(MODE)-1

#FFA: 0, 1v1: 1, FFA_SCORE: 2, 4v4: 3


START_BALLS = 5
ARENA_START_RADIUS = 400
SHRINKING_R = 0.01

TRUE_DEATH = True

if(MODE == 0):
    print("\n\nFFA with real-time selection. Controls: \n* Left click on a ball to delete it \n* Right click on a ball to let it replicate \n* Press 'Q' to speed up the game. \n* Press 'S' to save all balls currently in game.")

if (MODE == 1):
    print("\n\n1v1 copilot training. Controls: \n* Copilot mode is active while the space bar is pressed \nIn copilot mode, you do not control the ball, but the bot learns from you. \n* In copilot mode, use the arrow keys to indicate recommended movement. \n* Press 'F' to flip sides, changing which bot you train. \n* Press 'Q' to speed up the game. \n* Press 'S' to save all balls currently in game.")
    START_BALLS = 2
    ARENA_START_RADIUS = 200
    SHRINKING_R = 0.1
    TRUE_DEATH = False


#SCORE_DICT = {
#    'kill': 10000,
#    'frame_alive': 1,
#    'frame_center':0,
#    'score_collide':0,
#    'score_win':0}
if (MODE == 2):
    print("\n\nFFA with score based selection. Controls: \n* Edit \"scores.txt\" _beforehand_ to adjust how score is gained. \n* Press 'Q' to slow down the game. \n* Press 'S' to save all balls currently in game. \n* Press 'T' to toggle the scoreboard (worsens performance).")
    
    with open("scores.txt", 'r') as f:
        #json.dump(SCORE_DICT, f) 
        SCORE_DICT = json.load(f)
    
    
    START_BALLS = 21
    SCORE_KILL = SCORE_DICT['kill']
    SCORE_FRAME_ALIVE = SCORE_DICT['frame_alive']
    SCORE_FRAME_CENTER = SCORE_DICT['frame_center']
    SCORE_RAM = SCORE_DICT['score_ram']
    SCORE_COLLIDE = SCORE_DICT['score_collide']
    SCORE_WIN = SCORE_DICT['score_win']
    TRUE_DEATH = False
    SHRINKING_R = 0.1




 
    


if (MODE == 3):
    print("\n\n4v4 teamfight. Controls: \n* Press SPACE to start the game. \n* Press 'Q' to speed up the game.")
    TEAMS = 2
    TEAMSCORE = [0,0]
    START_BALLS = 8
    ARENA_START_RADIUS = 300
    SHRINKING_R = 0.1
    TRUE_DEATH = False
    

with open("bots.txt") as f: 
    ALL_CONTESTANTS = json.load(f)

CONTESTANTS = []
chosen_indices = []
while len(CONTESTANTS) < START_BALLS:
    print("\n\nNo. Contestants: " + str(START_BALLS)+". Choose from your list: ")
    for i in range(0,len(ALL_CONTESTANTS)):
        if i not in chosen_indices:
            print(str(i) + ": " + ALL_CONTESTANTS[i][0])
    
    print("You may press \n'r' to add a randomly created bot, \n'a' to choose every bot listed in that order, or \n'x' to end this step.")
    user = input()
    if user == 'r':
        CONTESTANTS.append([None, None])
        chosen_indices.append('r')
    elif user == 'x':
        break
    elif user == 'a':
        CONTESTANTS = ALL_CONTESTANTS
        break
    else:
        j = int(user)
        chosen_indices.append(j)
        CONTESTANTS.append(ALL_CONTESTANTS[j])
    
    print("Contestants chosen: " +str(chosen_indices))



ARENA_RADIUS = ARENA_START_RADIUS

MAX_BALLS_PER_AREA = 1/20
def max_balls():
    if (MODE == 0):
        return (ARENA_RADIUS/BALL_RADIUS)**2 * MAX_BALLS_PER_AREA
    elif (MODE == 1):
        return 2

def generate_spawn_coords(pos = None):
    if (MODE == 0):
        r = random.random()*ARENA_RADIUS /4 
        raw_x = random.random()-0.5
        raw_y = random.random()-0.5
        factor = r/math.sqrt(raw_x**2 + raw_y**2)

        return (raw_x * factor + CENTER_X, raw_y * factor +CENTER_Y)
    elif (MODE == 2):
        r = random.random()*ARENA_RADIUS/1.2 
        raw_x = random.random()-0.5
        raw_y = random.random()-0.5
        factor = r/math.sqrt(raw_x**2 + raw_y**2)

        return (raw_x * factor + CENTER_X, raw_y * factor +CENTER_Y)
    elif (MODE == 1):
        rx = (random.random()-0.5)*ARENA_RADIUS /10 
        ry = (random.random()-0.5)*ARENA_RADIUS / 10
        x = CENTER_X + (0.5-pos)*FLIP_SIDES*ARENA_RADIUS +rx
        y = CENTER_Y +ry
        return x,y
    elif (MODE == 3):
        rx = (random.random()-0.5)*ARENA_RADIUS /10 
        ry = (random.random()-0.5)*ARENA_RADIUS / 10
        if (pos < 4):
            x = CENTER_X - 0.5*ARENA_RADIUS*FLIP_SIDES+rx
            y = CENTER_Y + 0.2*(pos-1.5)*ARENA_RADIUS+ry
        else:
            x = CENTER_X + 0.5*ARENA_RADIUS *FLIP_SIDES+rx
            y = CENTER_Y + 0.2*(pos-5.5)*ARENA_RADIUS+ry
        return x,y
        


#BOT_COLORS = [generate_random_color() for _ in range(NUM_BALLS)]