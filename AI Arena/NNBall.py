
import constants as cst
import random
import pygame
import os

import numpy as np



#INPUT_SIZE = 8  # 4 inputs for each of the vector components (x, y) for center and closest ball, and 4 inputs for speed and relative speed
#HIDDEN_SIZE = 16 #first layer values in [-1,1], then in [0,1], last layer again in [-1,1]
#OUTPUT_SIZE = 2  # Output size for acceleration in the x and y directions



def distance_sqrd(x1,y1,x2,y2):
    return (x1-y1)**2 + (x2-y2)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

def rect(x):
    return np.clip(x,0,None)

def rect_derivative(x):
    return np.heaviside(x,0)

def wiggle_weights(weights,lower,upper):
    wiggled = weights+ (np.random.random_sample(weights.shape)-0.5)*(upper-lower)*cst.LEARNING_RATE
    return np.clip(wiggled,lower,upper)
   

def wiggle_color(color):
    newcolor = color.copy()
    for i in range(0,len(newcolor)):
        c = newcolor[i] + random.randint(-5, 5) * cst.LEARNING_RATE * 100
        if (c < 0):
            c = 0
        if (c > 255):
            c = 255
        newcolor[i] = c
    return newcolor


# Neural network class
class NeuralNetwork:
    def __init__(self, parent = None, layer_size =cst.LAYER_SIZE):
        # Initialize weights randomly
        if (parent == None):
            self.layer_size = layer_size
            self.layers = len(self.layer_size)
            self.create_layers()
        else:
            self.layer_size = parent.layer_size
            self.layers = len(self.layer_size)
            self.weights = []
            self.biases = []
            for (i,weights) in enumerate(parent.weights):
                if (i == 0 or i == self.layers-2):
                    self.weights.append(wiggle_weights(weights,-1,1))
                else:
                    self.weights.append(wiggle_weights(weights,0,1))
            for (i, biases) in enumerate(parent.biases):
                if (i == self.layers-2):
                    self.biases.append(wiggle_weights(biases,-1,1))
                else:
                    self.biases.append(wiggle_weights(biases,0,1))

        
    def create_layers(self):
        self.weights = []
        self.biases = []
        for i in range(0,self.layers-1):
            sizein = self.layer_size[i]
            sizeout = self.layer_size[i+1]
            self.weights.append(np.random.rand(sizein, sizeout))
            if (i == 0 or i==self.layers-2):
                self.weights[i] = (self.weights[i] - 0.5)*2
            self.biases.append(np.random.rand(sizeout))  
            if (i==self.layers-2):
                self.biases[i] = (self.biases[i] - 0.5)*2

    def predict(self, inputs):
        # Forward pass through the network with biases
        activation = inputs
        for i in range(0,self.layers-2):
            activation = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = rect(activation)
        output = np.dot(activation, self.weights[self.layers-2]) #+ self.biases[LAYERS-2]
        return np.clip(output,-1,1)

    def save(self,filename):
        filename = os.path.join("save",filename)
        print(filename)
        with open(filename, 'wb') as f:
            np.save(f, self.layer_size)
            for weights in self.weights:
                np.save(f, weights)
            for biases in self.biases:
                np.save(f, biases)

    def load(self,filename):
        filename = os.path.join("load",filename)
        print(filename)
        try:
            with open(filename, 'rb') as f:
                self.layer_size= tuple(np.load(f))
                self.layers = len(self.layer_size)
                self.create_layers()
                for i in range(0, self.layers -1):
                    self.weights[i] = np.load(f)
                for i in range(0, self.layers-1):
                    self.biases[i] = np.load(f)
        except:
            self.layer_size=(8,16,2)
            self.layers = len(self.layer_size)
            self.create_layers()
            with open(filename, 'rb') as f:
                for i in range(0, self.layers -1):
                    self.weights[i] = np.load(f)
                for i in range(0, self.layers-2):
                    self.biases[i] = np.load(f)
                self.biases[self.layers-2] = np.array([0,0])



    def train(self, inputs, targets, learning_rate=cst.LEARNING_RATE):
        # Forward pass
        
        activation = []
        activation.append(inputs)
        for i in range(0,self.layers-2):
            activation.append(rect(np.dot(activation[i], self.weights[i]) + self.biases[i]))
        output =  np.clip(np.dot(activation[self.layers-2], self.weights[self.layers-2]),-1,1) #+ self.biases[LAYERS-2]


        error = []
        delta = []
        error.append(targets - output)
        #print(error)

        # Backpropagation
        delta.append(error[0])

        for i in range(0,self.layers-1):
            error.append(delta[i].dot(self.weights[self.layers-2-i].T))
            delta.append(error[i+1] * rect_derivative(activation[self.layers-2-i]))

        # Update weights and biases
        for i in range(0, self.layers-1):
            self.weights[i] += activation[i].reshape(-1, 1) * delta[self.layers-2-i] * learning_rate
        for i in range(0, self.layers-2):    
            self.biases[i] += delta[self.layers-2-i] * learning_rate




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
        self.spawn()
        self.last_rammed_with = None
        self.last_rammed_countdown = 0
        self.team = None


    def spawn(self):
        self.x, self.y = cst.generate_spawn_coords(self.id)
        self.vx = 0
        self.vy = 0
        self.alive = True
        self.last_rammed_with = None
        self.last_rammed_countdown = 0

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

        if (self.last_rammed_countdown >0):
            self.last_rammed_countdown -= 1
        if (self.last_rammed_countdown <= 0):
            self.last_rammed_with = None

        # Check and handle collisions with walls
        #if self.x - cst.BALL_RADIUS < 0 or self.x + cst.BALL_RADIUS > cst.SCREEN_WIDTH:
        #    self.vx = -self.vx
        #if self.y - cst.BALL_RADIUS < 0 or self.y + cst.BALL_RADIUS > cst.SCREEN_HEIGHT:
        #    self.vy = -self.vy

    def die(self):
        self.alive = False
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
        self.breed_time = cst.BREED_FRAME_TIME
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
            center_x, center_y, self.vx, self.vy, closest_ball_x,closest_ball_y, closest_ball_speed_x, closest_ball_speed_y
        ])
        return inputs

    def accel(self, balls):
        # Prepare inputs for the neural network
        inputs = self.prepare_inputs(balls)

        # Get acceleration from the neural network
        self.predict = self.nn.predict(inputs)
        accel = np.round(self.predict)

        # Update velocity based on acceleration
        self.vx += accel[0]* cst.ACCEL
        self.vy += accel[1]* cst.ACCEL

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

        self.breed_time -= 1
        if (self.breed_time <= 0):
            self.breed_time = cst.BREED_FRAME_TIME
            self.instant_breed(balls)
            

    def instant_breed(self,balls):
        child = NeuralBall(color = self.color, name = None, parent_nn = self.nn)
        balls.append(child)
        



    def save(self):
        if (self.name == None):
            self.nn.save(str(self.id) + ".npy")
        else:
            self.nn.save(self.name + ".npy")


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