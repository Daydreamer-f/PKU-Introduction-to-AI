# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util,time
from math import sqrt, log

from game import Agent



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        maxValue = -float('inf')
        bestAction = None
        legalMoves = gameState.getLegalActions(0)
        for action in legalMoves:
            value = self.minVal(gameState.generateSuccessor(0,action), 0, 1)
            if value > maxValue:
                maxValue = value
                bestAction = action
        return bestAction
    def maxVal(self, gameState, depth = 0, agentIndex = 0):
        legalMoves = gameState.getLegalActions(0)
        maxValue = -float('inf')
        if depth == self.depth: #达到搜索深度
            return self.evaluationFunction(gameState)
        if not legalMoves: #判断是否空列表
            return self.evaluationFunction(gameState)
        for action in legalMoves:
            value = self.minVal(gameState.generateSuccessor(0,action), depth, 1)
            if value > maxValue:
                maxValue = value
        return maxValue
    def minVal(self, gameState, depth = 0, agentIndex = 1):
        legalMoves = gameState.getLegalActions(agentIndex)
        minValue = float('inf')
        if depth == self.depth: #达到搜索深度
            return self.evaluationFunction(gameState)
        if not legalMoves:
            return self.evaluationFunction(gameState)
        for action in legalMoves:
            if agentIndex == gameState.getNumAgents() - 1:
                value = self.maxVal(gameState.generateSuccessor(agentIndex,action), depth+1, 0)
            else:
                value = self.minVal(gameState.generateSuccessor(agentIndex,action), depth, agentIndex+1)
            if value < minValue:
                minValue = value
        return minValue
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        maxValue = -float('inf')
        bestAction = None
        alpha = -float('inf')
        beta = float('inf')
        legalMoves = gameState.getLegalActions(0)
        for action in legalMoves:
            value = self.minVal(gameState.generateSuccessor(0,action), 0, 1, alpha, beta)
            if value > maxValue:
                maxValue = value
                bestAction = action
            #if maxValue > beta:
                #return action
            if maxValue > alpha:
                alpha = maxValue
        return bestAction
    def maxVal(self, gameState, depth = 0, agentIndex = 0, alpha = -float('inf'), beta = float('inf')):
        legalMoves = gameState.getLegalActions(0)
        maxValue = -float('inf')
        if depth == self.depth: #达到搜索深度
            return self.evaluationFunction(gameState)
        if not legalMoves: #判断是否空列表
            return self.evaluationFunction(gameState)
        for action in legalMoves:
            value = self.minVal(gameState.generateSuccessor(0,action), depth, 1, alpha, beta)
            if value > maxValue:
                maxValue = value
            #此处有两种剪枝的判断方式
            #if maxValue > beta:
                #return maxValue
            if maxValue > alpha:
                alpha = maxValue
            if alpha > beta:
                return maxValue
        return maxValue
    def minVal(self, gameState, depth = 0, agentIndex = 1, alpha = -float('inf'), beta = float('inf')):
        legalMoves = gameState.getLegalActions(agentIndex)
        minValue = float('inf')
        if depth == self.depth: #达到搜索深度
            return self.evaluationFunction(gameState)
        if not legalMoves:
            return self.evaluationFunction(gameState)
        for action in legalMoves:
            if agentIndex == gameState.getNumAgents() - 1:
                value = self.maxVal(gameState.generateSuccessor(agentIndex,action), depth+1, 0, alpha, beta)
            else:
                value = self.minVal(gameState.generateSuccessor(agentIndex,action), depth, agentIndex+1, alpha, beta)
            if value < minValue:
                minValue = value
            #if minValue < alpha:
                #return minValue
            if minValue < beta:
                beta = minValue
            if alpha > beta:
                return minValue
        return minValue
        #util.raiseNotDefined()

class MCTSAgent(MultiAgentSearchAgent):
    """
      Your MCTS agent with Monte Carlo Tree Search (question 3)
    """

    def getAction(self, gameState):

        class Node:
            """
            We have provided node structure that you might need in MCTS tree.
            """
            def __init__(self, data):
                self.north = None
                self.east = None
                self.west = None
                self.south = None
                self.stop = None
                self.parent = None
                self.statevalue = data[0] # gameState
                self.numerator = data[1] # 节点获胜次数
                self.denominator = data[2] # 节点访问次数

        data = [gameState, 0, 1]
        cgstree = Node(data) # currentGamestateTree

        def Selection(cgs, cgstree):
            "*** YOUR CODE HERE ***"
            action = None
            while cgstree.north is not None or cgstree.east is not None or cgstree.west is not None or cgstree.south is not None:
                children = [] # children is a list of tuples, i.e. children[][] = (state, action),(state, action),...
                nextStep = (cgstree.north, "North")
                children.append(nextStep)
                nextStep = (cgstree.east, "East")
                children.append(nextStep)
                nextStep = (cgstree.west, "West")
                children.append(nextStep)
                nextStep = (cgstree.south, "South")
                children.append(nextStep)
                nextStep = (cgstree.stop, "Stop")
                children.append(nextStep)
                # children是所有可能行动，通过UCB选出最佳statevalue和action
                bestStateVal, action = UCB(children)
                cgs = bestStateVal
                if action == "North":
                    cgstree.north.statevalue = bestStateVal
                    cgstree = cgstree.north
                if action == "East": 
                    cgstree.east.statevalue = bestStateVal
                    cgstree = cgstree.east
                if action == "West":
                    cgstree.west.statevalue = bestStateVal
                    cgstree = cgstree.west
                if action == "South":
                    cgstree.south.statevalue = bestStateVal
                    cgstree = cgstree.south
                if action == "Stop":
                    cgstree.stop.statevalue = bestStateVal
                    cgstree = cgstree.stop
            return (cgs, cgstree, action)
            # util.raiseNotDefined()

        # children[][] = (state,action),... , UCB return (statevalue, action)
        
        # UCB1 with HeuristicFunction
        def UCB(children):
            i = 0
            while i < len(children):
                if children[i][0] is None or children[i][1] == "Stop":
                    children.pop(i)
                else:
                    i = i+1
            children_UCB = [] # children_UCB is a list of tuples, i.e. children_UCB[][] = (statevalue, action),(statevalue, action),...
            for i in range (len(children)):
                value = (children[i][0].numerator / children[i][0].denominator + sqrt(2 * log(children[i][0].parent.denominator)/ log(2.71828) / children[i][0].denominator) ), children[i][1]
                children_UCB.append(value)
            # 如果所有UCB值相同,e-greedy,以0.7的概率选最优，0.3的概率随机
            equal_counter = 1
            for i in range(len(children_UCB)-1):
                if children_UCB[i][0] == children_UCB[i+1][0]:
                    equal_counter += 1
            if equal_counter == len(children_UCB):
                maxIndex = 0
                randompointer = random.random()
                if randompointer < 0.85: 
                    eval_list = []
                    maxIndex_list = []
                    for i in range(len(children)):
                        eval_list.append(HeuristicFunction(children[i][0].statevalue))
                    # find the max index in eval_list and add to the maxIndex_list
                    maxIndex_list.append(eval_list.index(max(eval_list)))
                    maxVal = eval_list.pop(maxIndex_list[-1])
                    eval_list.insert(maxIndex_list[-1],-9999)
                    while maxVal in eval_list:
                        maxIndex_list.append(eval_list.index(max(eval_list)))
                        eval_list.pop(maxIndex_list[-1])
                        eval_list.insert(maxIndex_list[-1],-9999)
                    maxIndex = random.choice(maxIndex_list)
                else:
                    maxIndex = random.randint(0,len(children)-1)
            else:
                maxVal = -float('inf')
                maxIndex = 0
                for i in range(len(children_UCB)):
                    if children_UCB[i][0] > maxVal:
                        maxVal = children_UCB[i][0]
                        maxIndex = i
            return (children[maxIndex][0].statevalue, children[maxIndex][1])
        
        def Expansion(cgs, cgstree):
            "*** YOUR CODE HERE ***"
            # gameState = data[0] = self.statevalue
            legalMoves = cgstree.statevalue.getLegalActions(0)
            for action in legalMoves:
                newdata = [cgstree.statevalue.generateSuccessor(0, action), 0, 1]
                newNode = Node(newdata)
                if action == "North":
                    cgstree.north = newNode
                    cgstree.north.parent = cgstree
                elif action == "East":
                    cgstree.east = newNode
                    cgstree.east.parent = cgstree
                elif action == "West":
                    cgstree.west = newNode
                    cgstree.west.parent = cgstree
                elif action == "South":
                    cgstree.south = newNode
                    cgstree.south.parent = cgstree
                elif action == "Stop":
                    cgstree.stop = newNode
                    cgstree.stop.parent = cgstree
            # util.raiseNotDefined()

        def Simulation(cgs, cgstree):
            "*** YOUR CODE HERE ***"
            WinorLose = 0
            if cgstree.statevalue.isWin():
                WinorLose = 1
            elif cgstree.statevalue.isLose():
                WinorLose = 0
            else:
                WinorLose = 1 - 1 / (HeuristicFunction(cgs) + 1)
            return WinorLose, cgstree
            # util.raiseNotDefined()

        def Backpropagation(cgstree, WinorLose):
            "*** YOUR CODE HERE ***"
            while cgstree.parent is not None:
                cgstree.numerator = cgstree.numerator + WinorLose
                cgstree.denominator = cgstree.denominator + 1
                cgstree = cgstree.parent
            return cgstree
            # util.raiseNotDefined()

        def HeuristicFunction(currentGameState):
            "*** YOUR CODE HERE ***"
            pacman_pos = currentGameState.getPacmanPosition()
            food_pos = currentGameState.getFood().asList()
            # 计算距离最近的食物点的距离
            min_food_distance = float('inf')
            for food in food_pos:
                min_food_distance = min(min_food_distance, manhattanDistance(pacman_pos, food))
            # 计算所有鬼的距离
            ghost_distance = []
            ghost_pos = currentGameState.getGhostPositions()
            for ghost in ghost_pos:
                distance = manhattanDistance(pacman_pos, ghost)
                if(distance < 1):
                    return -float('inf')
                ghost_distance.append(distance)
            food = currentGameState.getNumFood()
            pellet = len(currentGameState.getCapsules())
            food_coefficient = 999999
            pellet_coefficient = 19999
            food_distance_coefficient = 999
            answer = (1.0 / (food + 1) * food_coefficient) + min(ghost_distance) + (
                1.0 / (min_food_distance + 1) * food_distance_coefficient) + (
                1.0 / (pellet + 1) * pellet_coefficient)
            return answer
            # util.raiseNotDefined()

        def endSelection(cgs, cgstree):
            if cgstree.north is not None or cgstree.east is not None or cgstree.west is not None or cgstree.south is not None:
                children = [] # children is a list of tuples, i.e. children[][] = (state, action),(state, action),...
                nextStep = (cgstree.north, "North")
                children.append(nextStep)
                nextStep = (cgstree.east, "East")
                children.append(nextStep)
                nextStep = (cgstree.west, "West")
                children.append(nextStep)
                nextStep = (cgstree.south, "South")
                children.append(nextStep)
                nextStep = (cgstree.stop, "Stop")
                children.append(nextStep)
                # children是所有可能行动，通过UCB选出最佳statevalue和action
                bestStateVal, action = UCB(children)
                return action

        "*** YOUR CODE HERE ***"
        i = 0
        for i in range(80):
            gameState, cgstree, action = Selection(gameState, cgstree)
            Expansion(gameState, cgstree)
            WinorLose, cgstree = Simulation(gameState, cgstree)
            cgstree = Backpropagation(cgstree,WinorLose)
            gameState = cgstree.statevalue
        return endSelection(gameState, cgstree)
        #util.raiseNotDefined()

