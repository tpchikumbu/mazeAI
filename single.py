from FourRooms import FourRooms
import random
import numpy
import sys
import time

def main():
    # Create FourRooms Object
    stoch : bool = False
    if len(sys.argv) == 2:
        if (sys.argv[1] == "-stochastic"):
            stoch = True
    fourRoomsObj = FourRooms('simple', stoch)
    
    #State matrix of the form [action, y, x]
    q_matrix = numpy.zeros((4,11,11))

    #Train agent and test trained agent at end
    train(fourRoomsObj, q_matrix)
    test(fourRoomsObj, q_matrix)
    
    # Show trained path
    fourRoomsObj.showPath(-1)

#Determine the best action using greedy selection 
def best_move(matrix, pos):
    x,y = pos[0] - 1, pos[1] -1 #Adjust indices to fit in array bounds
    best_act, best_value = 0, -100
    for act in range(4):
        act_value = matrix[act, y, x]
        #Store action with maximum reward. Value gets returned by function
        if (act_value > best_value) and (act_value != 0):
            best_act = act
            best_value = matrix[act, y, x]
    return best_act
    
#Move through grid world at random. Update understanding of world in matrix
#Train agent to find optimal path to a single package   
def train(agent: FourRooms, q_matrix):
    print(">>>Training...")
    epochs = 150
    learning_rate = 0.3
    discount_rate = 0.8
    epsilon = 0.6

    for run in range(epochs):
        agent.newEpoch()
        #Perform random actions until in terminal state. Update Q-matrix with each action.
        while (not agent.isTerminal()): 
            old_pos = agent.getPosition()
            
            #Epsilon-greedy action selection
            action = random.randint(0,3)
            if(random.random() < epsilon):
                action = best_move(q_matrix, old_pos)
            cell, new_pos, packages, terminal = agent.takeAction(action)
            
            #Adjust indices to fit in array bounds
            x_old,y_old = old_pos[0]-1,old_pos[1]-1
            x_new,y_new = new_pos[0]-1,new_pos[1]-1
            r = reward(old_pos, new_pos,cell)
            next_action = best_move(q_matrix, new_pos)
            #Adjust matrix values according to learning rule
            q_matrix[action, y_old, x_old] = q_matrix[action, y_old, x_old] +  round((learning_rate * (r + (discount_rate * (q_matrix[next_action, y_new,x_new])) - q_matrix[action, y_old, x_old])), 1)
#        agent.showPath(-1)
    print(">>>Training complete!")

#Take action according to Q-matrix to see if optimal policy has been found
def test(agent: FourRooms, q_matrix):
    print(">>>Testing...")
    agent.newEpoch()
    print('Agent starts at: {0}'.format(agent.getPosition()))
    
    #Store timiming information to check for anomalies
    start_time = time.time() #Store time to compare runtimes
    max_secs = 200
    
    while not agent.isTerminal():
        pos = agent.getPosition()
        action = best_move(q_matrix, pos)
        agent.takeAction(action)
        print('Agent is now at: {0}'.format(agent.getPosition()))
        
        #Timeout if threshold exceeded
        if (time.time() - start_time > max_secs):
            print("Agent timed out. Possible anomaly")
            break
    print('Package found at: {0}'.format(agent.getPosition()))

#Determine reward for each action
def reward(old_pos, new_pos, cell):
    if old_pos == new_pos: #Punish running into boundaries
        return -50
    if cell > 0: #Reward finding package
        return 1000
    elif cell == 0: #Reduce reward for each step taken
        return -5
    else:
        return 0

if __name__ == "__main__":
    main()
