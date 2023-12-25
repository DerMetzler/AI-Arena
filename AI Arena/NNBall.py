from tkinter import HIDDEN
import constants as cst
import random
import pygame
import os

import numpy as np


#LAYER_SIZE = [16,24,16,4]
INPUT_SIZE = 8  # 4 inputs for each of the vector components (x, y) for center and closest ball, and 4 inputs for speed and relative speed
HIDDEN_SIZE = 16 #first layer values in [-1,1], then in [0,1], last layer again in [-1,1]
OUTPUT_SIZE = 2  # Output size for acceleration in the x and y directions



def distance_sqrd(x1,y1,x2,y2):
    return (x1-y1)**2 + (x2-y2)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(self, x):
    return x * (1 - x)

def rect(x):
    return np.clip(x,0,None)

def rect_derivative(x):
    return np.heaviside(x,0)

def wiggle_weights(weights,lower,upper):
    wiggled = weights+ (np.random.random_sample(weights.shape)-0.5)*((upper-lower)*0.1)
    return np.clip(wiggled,lower,upper)
   

def wiggle_color(color):
    newcolor = color.copy()
    for i in range(0,len(newcolor)):
        c = newcolor[i] + random.randint(-5, 5)
        if (c < 0):
            c = 0
        if (c > 255):
            c = 255
        newcolor[i] = c
    return newcolor


# Neural network class
class NeuralNetwork:
    def __init__(self, parent = None):
        # Initialize weights randomly
        if (parent == None):
            self.weights_input_hidden = (np.random.rand(INPUT_SIZE, HIDDEN_SIZE)-0.5)*2
            self.weights_hidden_output =(np.random.rand(HIDDEN_SIZE, OUTPUT_SIZE)-0.5)*2
            self.biases_hidden = np.random.rand(HIDDEN_SIZE)
            #self.biases_output = np.random.rand(OUTPUT_SIZE)
        else:
            self.weights_input_hidden = wiggle_weights(parent.weights_input_hidden,-1,1)
            self.weights_hidden_output = wiggle_weights(parent.weights_hidden_output,-1,1)
            self.biases_hidden = wiggle_weights(parent.biases_hidden,0,1)
            #self.biases_output = wiggle_weights(parent.biases_output,0,1)

    def predict(self, inputs):
        # Forward pass through the network with biases
        hidden = np.dot(inputs, self.weights_input_hidden) + self.biases_hidden
        hidden_activation = rect(hidden)

        output = np.dot(hidden_activation, self.weights_hidden_output) #+ self.biases_output
        return output

    def save(self,filename):
        filename = os.path.join("save",filename)
        print(filename)
        with open(filename, 'wb') as f:
            np.save(f, self.weights_input_hidden)
            np.save(f, self.weights_hidden_output)
            np.save(f, self.biases_hidden)
            #np.save(f, self.biases_output)

    def load(self,filename):
        filename = os.path.join("load",filename)
        print(filename)
        with open(filename, 'rb') as f:
            self.weights_input_hidden = np.load(f)
            self.weights_hidden_output = np.load(f)
            self.biases_hidden = np.load(f)
            #self.biases_output = np.load(f)


    def train(self, inputs, targets, learning_rate=0.01):
        # Forward pass
        
        hidden = np.dot(inputs, self.weights_input_hidden) + self.biases_hidden
        hidden_activation = rect(hidden)

        output = np.dot(hidden_activation, self.weights_hidden_output) #+ self.biases_output
        output = np.clip(output,-0.9,0.9)
        # Calculate the error
        output_error = targets - output

        # Backpropagation
        output_delta = output_error #* rect_derivative(output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * rect_derivative(hidden_activation)

        # Update weights and biases
        self.weights_hidden_output += hidden_activation.reshape(-1, 1) * output_delta * learning_rate
        #self.biases_output += output_delta * learning_rate

        self.weights_input_hidden += inputs.reshape(-1, 1) * hidden_delta * learning_rate
        self.biases_hidden += hidden_delta * learning_rate





# Ball class
class Ball:
    def __init__(self, color = None, name = None, team = None):
        self.id = cst.LOWEST_AVAILABLE_ID
        cst.LOWEST_AVAILABLE_ID+=1
        self.name = name
        if (color == None):
            self.color = cst.generate_random_color()
        else:
            self.color = wiggle_color(color)
        self.score = 0
        self.respawn_time = 0
        self.spawn()
        self.last_rammed_with = None
        self.team = None


    def spawn(self):
        self.x, self.y = cst.generate_spawn_coords(self.id)
        self.vx = 0
        self.vy = 0
        self.alive = True
        self.last_rammed_with = None

    def accel(self, balls):
        return

    def breed(self, balls):
        return

    def save(self):
        return

    def move(self, balls):
        self.accel(balls)

        self.x += self.vx
        self.y += self.vy

        # Check and handle collisions with walls
        #if self.x - cst.BALL_RADIUS < 0 or self.x + cst.BALL_RADIUS > cst.SCREEN_WIDTH:
        #    self.vx = -self.vx
        #if self.y - cst.BALL_RADIUS < 0 or self.y + cst.BALL_RADIUS > cst.SCREEN_HEIGHT:
        #    self.vy = -self.vy

    def die(self):
        self.alive = False
        self.respawn_time = cst.RESPAWN_TIME
        if self.last_rammed_with != None:
            if self.last_rammed_with.last_rammed_with == self:
                self.last_rammed_with.last_rammed_with = None
            self.last_rammed_with = None

    def draw(self, screen):
        if self.alive:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), cst.BALL_RADIUS)

    def learn(self, doing_well,balls):
        return

# Ball class with neural network control
class NeuralBall(Ball):
    def __init__(self, color = None, name = None, parent_nn = None, team = None):
        Ball.__init__(self,color, name, team)
        self.nn = NeuralNetwork(parent_nn)
        self.breed_time = cst.BREED_TIME
        self.predict = None
        
        if(self.name != None):
            filename = self.name + ".npy"
            self.nn.load(filename)

    def prepare_inputs(self, balls):
        center_x = (self.x-cst.CENTER_X)/cst.ARENA_RADIUS
        center_y = (self.y-cst.CENTER_Y)/cst.ARENA_RADIUS
        
        smallest_dist_sqrd = (2 * cst.ARENA_RADIUS)**2
        closest_ball = self
        
        for ball in balls:
            if (self.team == None or self.team != ball.team):
                if (ball.id != self.id):
                    dist_sqrd = distance_sqrd(ball.x, ball.y, self.x, self.y) 
                    if (dist_sqrd <= smallest_dist_sqrd):
                        smallest_dist_sqrd = dist_sqrd
                        closest_ball = ball

        closest_ball_x = (self.x-closest_ball.x)/cst.ARENA_RADIUS
        closest_ball_y = (self.y-closest_ball.y)/cst.ARENA_RADIUS

        closest_ball_speed_x = (self.vx-closest_ball.vx)/cst.ARENA_RADIUS
        closest_ball_speed_y = (self.vy-closest_ball.vy)/cst.ARENA_RADIUS

        inputs = np.array([
            center_x, center_y, self.vx, self.vy,
            closest_ball_x, closest_ball_y, closest_ball_speed_x, closest_ball_speed_y
        ])
        return inputs

    def accel(self, balls):
        # Prepare inputs for the neural network
        inputs = self.prepare_inputs(balls)

        # Get acceleration from the neural network
        self.predict = np.clip(self.nn.predict(inputs),-0.9,0.9)
        acceleration = np.round(self.predict)

        # Update velocity based on acceleration
        self.vx += acceleration[0]*cst.ACCEL
        self.vy += acceleration[1]*cst.ACCEL

        if (self.vx >= cst.MAX_SPEED):
            self.vx = cst.MAX_SPEED
        if (self.vx <= -cst.MAX_SPEED):
            self.vx = -cst.MAX_SPEED
        if (self.vy >= cst.MAX_SPEED):
            self.vy = cst.MAX_SPEED
        if (self.vy <= -cst.MAX_SPEED):
            self.vy = -cst.MAX_SPEED

    def breed(self, balls):
        if not self.alive:
            return

        self.breed_time -= cst.FRAME_TIME
        if (self.breed_time <= 0):
            self.breed_time = cst.BREED_TIME
            self.instant_breed(balls)
            

    def instant_breed(self,balls):
        child = NeuralBall(color = self.color, name = None, parent_nn = self.nn)
        balls.append(child)

    #doing_well assumes -1,0,1
    def learn(self,input_list,target_list, unlearn = False):


        #if (doing_well == 1): #doing well
        for i in range(0,len(input_list)):
            inputs = input_list[i]
            targets = target_list[i]
            if unlearn:
                targets *= -1
                r = (random.randint(0,1)-0.5)*2
                np.where(targets == 0, r,targets)
            self.nn.train(inputs, targets)



    def save(self):
        self.nn.save(str(self.id) + ".npy")


# Example usage
# Assume you have two balls 'ball' and 'closest_ball', and a center point (cx, cy)
# Calculate vectors and speeds
#center_vector = [cx - ball.x, cy - ball.y]
#center_speed = math.sqrt(ball.vx ** 2 + ball.vy ** 2)

#closest_ball_vector = [closest_ball.x - ball.x, closest_ball.y - ball.y]
#closest_ball_speed = math.sqrt((closest_ball.x - ball.x)**2 + (closest_ball.y - ball.y)**2)

# Update the neural ball using the neural network
#neural_ball = NeuralBall(ball.x, ball.y)
#neural_ball.move(center_vector, center_speed, closest_ball_vector, closest_ball_speed)