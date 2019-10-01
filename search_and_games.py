############################################################
# CIS 521: Homework 2
############################################################

student_name = "Jiachen Wang"


############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import random
import heapq
from collections import defaultdict
from math import sqrt


############################################################
# Section 1: Lights Out
############################################################

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.bd = board
        self.M = len(self.bd)
        self.N = len(self.bd[0])


    def get_board(self):
        return self.bd

    def perform_move(self, row, col):
        self.bd[row][col] = not self.bd[row][col]
        if row - 1 >= 0:
        	self.bd[row - 1][col] = not self.bd[row - 1][col]
        if row + 1 <= self.M - 1:
        	self.bd[row + 1][col] = not self.bd[row + 1][col]
        if col - 1 >= 0:
        	self.bd[row][col - 1] = not self.bd[row][col - 1]
        if col + 1 <= self.N - 1:
        	self.bd[row][col + 1] = not self.bd[row][col + 1]


    def scramble(self):
        for row in range(self.M):
        	for col in range(self.N):
        		if random.random() < 0.5:
        			self.perform_move(row, col)


    def is_solved(self):
        for row in range(self.M):
        	for col in range(self.N):
        		if self.bd[row][col] == True:
        			return False
        return True

    def copy(self):
        temp = [[self.bd[i][j] for j in range(self.N)] for i in range(self.M)]

        return LightsOutPuzzle(temp)

    def reset(self, row, col):
        self.perform_move(row, col)

    def successors(self):
        for row in range(self.M):
            for col in range(self.N):
                tuple = (row, col)
                copy_ofself = self.copy()
                copy_ofself.perform_move(row, col)
                yield (tuple, copy_ofself)


    def find_solution(self):
        #successor is our frontier
        #queue is for storing the path
        queue = []
        visited = set()
        #this part is for the case where initial state is already solved. We just return []

        if self.is_solved():
            return queue

        #this part is for initializing our queue
        #it will have the first level children.
        for move, next_state in self.successors():

            visited.add(tuple(map(tuple, next_state.get_board())))
            if next_state.is_solved():
                # if after first move we solve the puzzle, we just return this move
                return [move]

            queue.append([(move, next_state)])

        while queue:
            # get the current path
            current_path = queue.pop(0)
            # get the last state in this current path
            recent_state = current_path[-1][1]

            if recent_state.is_solved():
                return  [item[0] for item in current_path]
            # try to move to next level
            for move2, next_state2 in recent_state.successors():
                # check if the new_state2 has been visited before, if so, just skip
                if tuple(map(tuple, next_state2.get_board())) not in visited:
                    # if not, add this next_state2 to our new_path
                    new_path = list(current_path)
                    new_path.append((move2, next_state2))
                    #mark this new_state2 as visited
                    visited.add(tuple(map(tuple, next_state2.get_board())))
                    queue.append(new_path)

        #if not returned above, it meaans not solvable, return None
        return None


def create_puzzle(rows, cols):
    temp = [[False for _ in range(cols)] for _ in range(rows)]
    return LightsOutPuzzle(temp)

############################################################
# Section 2: Grid Navigation
############################################################


def A_star(start, goal, scene):
    def h_estimate(start, goal):

        H = sqrt(pow((start[0]-goal[0]),2) + pow((start[1] - goal[1]),2))
        return H
    def construct_path(came_from, current):
        #print("entering construct path")
        #print("current is")
        #print(current)
        total_path = [current]
        #print(came_from)
        while current in came_from:
            #print("debug")
            current = came_from[current]
            total_path.insert(0, current)
        return total_path

    #construct neighbors first
    neighbors = defaultdict(set)
    for i in range(len(scene)):
        for j in range(len(scene[0])):
            if i -1 >= 0:
                neighbors[(i,j)].add(((i-1,j), 1))

            if j - 1 >= 0:
                neighbors[(i,j)].add(((i, j-1), 1))

            if i - 1>= 0 and j -1>= 0:
                neighbors[(i,j)].add(((i-1, j-1), sqrt(2)))

            if j + 1 <= len(scene[0]) - 1:
                neighbors[(i,j)].add(((i, j+1), 1))

            if i - 1>= 0 and j + 1 <= len(scene[0]) - 1:
                neighbors[(i,j)].add(((i-1, j+1), sqrt(2)))

            if i + 1 <= len(scene) - 1:
                neighbors[(i,j)].add(((i+1, j), 1))

            if  i + 1 <= len(scene) - 1 and j - 1 >= 0:
                neighbors[(i,j)].add(((i+1, j-1), sqrt(2)))

            if  i + 1 <= len(scene) - 1 and j + 1 <= len(scene[0]) - 1:
                neighbors[(i,j)].add(((i+1, j+1), sqrt(2)))

    open_set = set()
    # for ex, key is A, value is B, this means if you want to go to A, you need to go to B first
    # so C-> B-> A, when we construct path, came_from{A} will help us find B, then came_from[B] will
    # help us find C, so on and so on.
    came_from = {}
    g_score = defaultdict(lambda:float("inf"))
    #this is for storing the g_score for each location
    g_score[start] = 0
    #this is for storing the f_score for each location
    f_score = defaultdict(lambda:float("inf"))
    f_score[start] = h_estimate(start, goal)
    open_set.add((f_score[start], start))

    #print("initial open_set is")
    #print(open_set)
    while open_set:
        # current has the smallest f_score
        #print("entering our big while loop while open_set")
        current = min(open_set)
        #print("current analyzing point is")
        #print(current[1])
        if current[1] == goal:
            #print("we have found the goal")
            return construct_path(came_from, current[1])

        open_set.remove(current)
        # for example, current[1] is (0,0)
        for neighbor in neighbors[current[1]]:
            #print("current neighbor is")
            #print(neighbor[0])
            # neighbor is something like ((1, 0), 1.4142135623730951))
            # so neighbor[0] is (1,0), the actual neighbor
            # neighbor[1] is the distance from current point to this neighbor
            if scene[neighbor[0][0]][neighbor[0][1]] is False:
                #print("this neighbor is empty, try to go there")
                # this neighbor is empty, we can go there
                tentative_gscore = g_score[current[1]] + neighbor[1]

                if tentative_gscore < g_score[neighbor[0]]:
                    # this path to neighbor is better than previous one
                    #print("tentative g_score is better, so go here")
                    came_from[neighbor[0]] = current[1]
                    #print(came_from)
                    #update the g value
                    g_score[neighbor[0]] = tentative_gscore
                    f_score[neighbor[0]] = g_score[neighbor[0]] + h_estimate(neighbor[0], goal)
                    if neighbor[0] not in open_set:
                        #print("add this neighbor into open_set for further analyzing")
                        # problem is here, you should add f_score and this point together into our open_set
                        open_set.add((f_score[neighbor[0]], neighbor[0]))
                        #print(open_set)
    return None

def find_path(start, goal, scene):
    return A_star(start, goal, scene)

############################################################
# Section 3: Dominoes Games
############################################################
def create_dominoes_game(rows, cols):
    temp = [[False for _ in range(cols)] for _ in range(rows)]
    return DominoesGame(temp)

class DominoesGame(object):
    # Required
    def __init__(self, board):
        self.best_move = None
        self.board = board
        self.visited = set()
        self.M = len(board)
        self.N = len(board[0])

    def get_board(self):
        return self.board

    def reset(self):
        for i in range(self.M):
            for j in range(self.N):
                if self.board[i][j] == True:
                    self.board[i][j] = False

    def is_legal_move(self, row, col, vertical):
        if row < 0 or row > (self.M - 1) or col < 0 or col>(self.N - 1):
            return False
        if vertical == True:
            if row + 1 >= self.M or self.board[row][col] == True or self.board[row + 1][col] ==True:
                return False
        if vertical == False:
            if col + 1 >= self.N or self.board[row][col] == True or self.board[row][col + 1] == True:
                return False
        return True

    def legal_moves(self, vertical):
        for i in range(self.M):
            for j in range(self.N):
                if self.is_legal_move(i, j, vertical):
                    yield (i, j)

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical):
            if vertical == True:
                self.board[row][col] = True
                self.board[row + 1][col] = True
            else:
                self.board[row][col] = True
                self.board[row][col + 1] = True

    def game_over(self, vertical):
        for i in range(self.M):
            for j in range(self.N):
                if self.is_legal_move(i, j, vertical):
                    return False
        return True

    def copy(self):
        temp = [[self.board[i][j] for j in range(self.N)] for i in range(self.M)]

        return DominoesGame(temp)

    def successors(self, vertical):
        for row in range(self.M):
            for col in range(self.N):
                if self.is_legal_move(row, col, vertical):
                    move = (row, col)
                    copy_ofself = self.copy()
                    copy_ofself.perform_move(row, col, vertical)
                    yield (move, copy_ofself)

    def get_random_move(self, vertical):
        temp = [(i,j) for i in range(self.M) for j in range(self.N)]
        if self.game_over(vertical):
            return None
        while True:
            choice = random.choice(temp)
            if self.is_legal_move(choice[0], choice[1], vertical):
                return choice

    # Required
    def get_best_move(self, vertical, limit):
        self.leaf_node = 0

        negative_inf = - float('inf')
        positive_inf = float("inf")

        def h_value(node):
            my_legalmove = len(list(node.legal_moves(vertical)))
            opponent_legalmove = len(list(node.legal_moves(not vertical)))
            return my_legalmove - opponent_legalmove

        def alphabeta(node, depth, alpha, beta, vertical, maxNode):
            # successor usecase:
            # for m, new_g in g.successors(True):
            #   print(m, new_g.get_board()) so, new_g is an object,m is a tuple,means move
            if depth == 0:
                # means we arrive at leaf node.
                self.leaf_node += 1
                #print("current leaf node number is")
                #print(self.leaf_node)
                return h_value(node)
            if maxNode:
                # this is max_node
                value = negative_inf
                if node.game_over(vertical) and depth > 0:
                    self.leaf_node += 1
                    return h_value(node)
                for m, new_g in node.successors(vertical):
                    value = max(value, alphabeta(new_g, depth - 1, alpha, beta, not vertical, False))
                    previous_alpha = alpha
                    alpha = max(previous_alpha, value)
                    if alpha >= beta:
                        #print("break in max node")
                        break
                    if alpha > previous_alpha:
                        self.best_move = m            
                return value
            else:
                # this is min_node
                value = positive_inf
                if node.game_over(vertical) and depth > 0:
                    self.leaf_node += 1
                    return h_value(node)

                for m, new_g in node.successors(vertical):
                    value = min(value, alphabeta(new_g, depth - 1, alpha, beta, not vertical, True))
                    beta = min(beta, value)
                    if alpha >= beta:
                        #print("break in min node")
                        break
                return value

        # call alphabeta function to start calculation
        value = alphabeta(self.copy(), limit, negative_inf, positive_inf, vertical, True)
        num_ofleaf = self.leaf_node

        return (self.best_move, value, self.leaf_node)
