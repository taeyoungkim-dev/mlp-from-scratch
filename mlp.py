#28*28 == 784
import random
import math
import numpy as np
#----activation function--------#
def ReLU(num):
    return max(0, num)
def Sigmoid(num):
    return 1 / (1 + math.exp(-num))
def Identity(num):
    return num

def softmax(output_scores):
    # 1. output_scores (z 벡터)에서 최댓값을 찾습니다.
    #    (np.array를 쓰면 편하지만, 리스트로도 max(list)로 찾을 수 있습니다.)
    max_score = np.max(output_scores)
   
    # 2. 모든 점수에서 최댓값을 빼줍니다. (오버플로우 방지)
    stable_scores = output_scores - max_score
   
    # 3. e^z_i 값들을 계산합니다.
    exp_scores = np.exp(stable_scores)
   
    # 4. e^z_i 값들의 총합을 계산합니다.
    sum_exp_scores = np.sum(exp_scores)
   
    # 5. 최종 확률을 계산합니다.
    probabilities = exp_scores / sum_exp_scores
   
    return probabilities
 
#-------------------------Define model-------------------------------#
class node:
    def __init__(self, input_size,act_fun, bias = 0.01):
        #num of input node
        self.input_size =  input_size
        #define input node
        self.weights = [random.uniform(-1,1) for _ in range(self.input_size)]
        self.act_fun = act_fun
        self.bias = bias
    def run(self,my_input):
        total = 0.0
        for n in range(self.input_size):
            total += my_input[n] * self.weights[n]
        total += self.bias
        return self.act_fun(total)

class layer:
    def __init__(self, layer_size, input_size, act_fun):
        self.layer_size = layer_size
        self.input_size = input_size
        self.act_fun = act_fun
        self.nodes = []
        self.input = []
        self.output = []
        for _ in range(layer_size):
            self.nodes.append(node(self.input_size,act_fun))
    def forward(self,input):
        self.output = []
        self.input = input
        for index in range(self.layer_size):
            self.output.append(self.nodes[index].run(self.input))
        return self.output

class Network:
    def __init__(self,input_size):
        self.input_size = input_size
        self.layer_depths = 0
        self.temp_output_size = input_size
        self.layers = []
    def add_layer(self,layer_size,act_fun=ReLU):
        self.layers.append(layer(layer_size,self.temp_output_size,act_fun))
        self.layer_depths += 1
    def predict(self,data):
        for x in range(self.layer_depths):
            data = self.layers[x].forward(data)
        return data

my_network = Network(784)
my_network.add_layer(784,Identity)
my_network.add_layer(128,ReLU)
my_network.add_layer(64,ReLU)
my_network.add_layer(10,softmax)

