import pygame
import sys
import math
import math
import constants as cst
import NNBall
import random
import numpy as np



def spawn_balls():
    balls = []
    for i in range(cst.START_BALLS):
        name = None
        color = None
        if (i < len(cst.CONTESTANTS)):
            name, color = cst.CONTESTANTS[i]
        team = None
        if (cst.MODE == 3):
            if (i < 4):
                team = 0
            else:
                team = 1
        balls.append(NNBall.NeuralBall(color = color, name = name, team =team))
    return balls

# Initialize Pygame
pygame.init()

balls = spawn_balls()
# Create a list of balls



# Create the game window
screen = pygame.display.set_mode((cst.SCREEN_WIDTH, cst.SCREEN_HEIGHT))
pygame.display.set_caption("AI Ball Game")

# Main game loop
#if (cst.MODE == 1):
#    input_list = []
#    target_list = []


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    screen.fill(cst.BLACK)
    
    cst.ARENA_RADIUS -= cst.SHRINKING_R
    pygame.draw.circle(screen, cst.ARENA_COLOR, (cst.CENTER_X, cst.CENTER_Y), cst.ARENA_RADIUS)

    if (cst.MODE == 0):
        if (pygame.mouse.get_pressed()[0]):
            m_x, m_y = pygame.mouse.get_pos()
            for ball in balls:
                distance = math.sqrt((m_x - ball.x) ** 2 + (m_y - ball.y) ** 2)
                if (distance < cst.BALL_RADIUS):
                    ball.alive = False

        if (pygame.mouse.get_pressed()[2]):
            m_x, m_y = pygame.mouse.get_pos()
            for ball in balls:
                distance = math.sqrt((m_x - ball.x) ** 2 + (m_y - ball.y) ** 2)
                if (distance < cst.BALL_RADIUS):
                    ball.instant_breed(balls)
                    
        # Move and draw each ball
        breed_allowed = len(balls) < cst.max_balls()
        if breed_allowed:
            for ball in balls:
                ball.breed(balls)
    
    numAlive = 0
    for ball in balls:
        if ball.alive:
            numAlive += 1
            ball.move(balls)
            if (ball.x -cst.CENTER_X)**2 + (ball.y - cst.CENTER_Y)**2 > cst.ARENA_RADIUS**2:
                if (cst.MODE == 2 and ball.last_rammed_with != None):
                    ball.last_rammed_with.score += cst.SCORE_KILL
                ball.die()

            ball.draw(screen)
        elif cst.TRUE_DEATH:
            balls.remove(ball)

    # Check for collisions between balls
    for i in range(len(balls)):
        if balls[i].alive:
            for j in range(i + 1, len(balls)):
                if balls[j].alive:
                    distance = math.sqrt((balls[i].x - balls[j].x) ** 2 + (balls[i].y - balls[j].y) ** 2)
                    if distance < 2 * cst.BALL_RADIUS:
                        # Collision detected, reverse velocities
                        balls[i].vx, balls[j].vx = balls[j].vx, balls[i].vx
                        balls[i].vy, balls[j].vy = balls[j].vy, balls[i].vy

                        xtransl= (balls[i].x - balls[j].x)*cst.BALL_RADIUS/(10*distance)
                        ytransl= (balls[i].y - balls[j].y)*cst.BALL_RADIUS/(10*distance)

                        balls[i].x += xtransl
                        balls[j].x -= xtransl
                        balls[i].y += ytransl
                        balls[j].y -= ytransl
                        
                        if (cst.MODE == 2):
                            balls[i].score += cst.SCORE_COLLIDE
                            balls[j].score += cst.SCORE_COLLIDE

                        if((balls[i].vx - balls[j].vx)**2 + (balls[i].vy - balls[j].vy)**2 > cst.COLLIDE_SPEED**2):
                            balls[i].last_rammed_with=balls[j]
                            balls[j].last_rammed_with=balls[i]

                            if (cst.MODE == 2):
                                balls[i].score += cst.SCORE_RAM
                                balls[j].score += cst.SCORE_RAM

            

    if (cst.MODE ==1):
        main = balls[0]
        if (main.alive and pygame.key.get_pressed()[pygame.K_SPACE]):

            #input_list.append(main.prepare_inputs(balls))
            #target_list.append(np.round(main.predict))
            #if (len(input_list) >= cst.FRAMES_REMEMBERED):
                #input_list.pop()
                #target_list.pop()
            targets = np.array([0,0])
            if (pygame.key.get_pressed()[pygame.K_UP]):
                #main.learn(input_list,target_list , unlearn = False)
                targets[1] = -1  
            if (pygame.key.get_pressed()[pygame.K_DOWN]):
                #main.learn(input_list,target_list, unlearn = True)
                targets[1] = 1
            if (pygame.key.get_pressed()[pygame.K_LEFT]):
                targets[0] = -1  
            if (pygame.key.get_pressed()[pygame.K_RIGHT]):
                targets[0] = 1
            inputs = main.prepare_inputs(balls)
            main.nn.train(inputs,targets)

        if numAlive <=1:
            cst.ARENA_RADIUS = cst.ARENA_START_RADIUS
            pygame.time.delay(cst.RESTART_TIME)
            for ball in balls:
                ball.spawn()

    if (cst.MODE ==2):
        for ball in balls:
            if ball.alive:
                ball.score += cst.SCORE_FRAME_ALIVE
                if (ball.x-cst.CENTER_X)**2 + (ball.y-cst.CENTER_Y)**2 < cst.ARENA_RADIUS**2 /4:
                    ball.score += cst.SCORE_FRAME_CENTER
        
        if numAlive <= 1:
            if numAlive == 1:
                for ball in balls:
                    if ball.alive:
                        ball.score+=cst.SCORE_WIN
            
            cst.ARENA_RADIUS = cst.ARENA_START_RADIUS
            pygame.time.delay(cst.RESTART_TIME)

            balls.sort(key = lambda b : -b.score)
            balls = balls[:5]
            for ball in balls:
                print(str(ball.name) + " ("+ str(ball.id) + "): "+str(ball.score))

            for i in range(0,6):
                for j in range(0,6-i):
                    balls[i].instant_breed(balls)

            for ball in balls:
                ball.spawn()
                ball.score = 0

    if (cst.MODE == 3):
        team1alive = False
        team2alive = False
        for ball in balls:
            if ball.alive:
                if (ball.id < 4):
                    team1alive = True
                else:
                    team2alive = True

        if (not team1alive or not team2alive):
            if team1alive:
                cst.TEAMSCORE[0] +=1
            if team2alive:
                cst.TEAMSCORE[1] +=1
            print("Current score: " + str(cst.TEAMSCORE))

            cst.ARENA_RADIUS = cst.ARENA_START_RADIUS
            pygame.time.delay(cst.RESTART_TIME)
            for ball in balls:
                ball.spawn()


                            

    # Update the display
    pygame.display.flip()

    if(numAlive == 1 or pygame.key.get_pressed()[pygame.K_s]):
        balls[0].save()



    # Add a small delay to control the speed of the simulation
    pygame.time.delay(cst.FRAME_TIME)
