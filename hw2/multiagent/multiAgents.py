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
import random, util

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        nextPacmanPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        "*** YOUR CODE HERE ***"
        ghostPos = currentGameState.getGhostPosition(1)
        distToGhost = util.manhattanDistance(ghostPos, nextPacmanPos)
        curScore = successorGameState.getScore()
        foods = newFood.asList()
        0

        """find closest food"""
        def closestFood(foods, nextPacmanPos):
            closestfood = 70
            for food in foods:
                thisdist = util.manhattanDistance(food, nextPacmanPos)
                if (thisdist < closestfood):
                    closestfood = thisdist
            return closestfood

        if(distToGhost == 1):
            curScore = curScore - 101

        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            curScore = curScore + 100

        curScore = curScore - (2 * closestFood(foods, nextPacmanPos))

        return curScore

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

def terminalTest(state):
            return state.isWin() or state.isLose()

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
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



        def minAgent(state, depth):
            # first check if we are done evaluating

            if terminalTest(state):
                return state.getScore()
            # make a list of possible actions from the current state
            actions = state.getLegalActions(0)
            # initialize smallest score possible, goes up from there
            maxScore = float("-inf")
            # initialize default best action, obviously standing still is not optimal
            possibleAction = Directions.STOP
            # go through actions and find the best score by getting the maxAgent to search the subtree
            for action in actions:
                # tried to find the max score of all the maxAgent calls, but we need to keep track of the
                # best action as well, so instead of doing one max call I loop through actions in a for loop
                score = maxAgent(state.generateSuccessor(0, action), depth, 1);
                #curScore = max(curScore, maxAgent(state.generateSuccessor(0, action), depth, 1))
                if score > maxScore:
                    maxScore = score
                    possibleAction = action
            #
            if depth == 0:
                return possibleAction
            return maxScore

        def maxAgent(state, depth, numGhost):
            # first check if we are done evaluating
            if terminalTest(state):
                return state.getScore()
            # initialize largest score possible, goes down from there
            # get all possible actions for current character
            # if we are pacman then we need to go through the depth and increment
            actions = state.getLegalActions(numGhost)
            minScore = float("inf")
            # if next person is the player then we need to switch
            curChar = numGhost + 1
            if numGhost == state.getNumAgents() - 1:
                curChar = 0

            # for successor state and action
            for action in actions:
                #if pacman and done exploring
                if curChar == 0 and depth == self.depth - 1:
                        curScore = self.evaluationFunction(state.generateSuccessor(numGhost, action))
                elif curChar == 0:
                        curScore = minAgent(state.generateSuccessor(numGhost, action), depth + 1)
                #for ghosts we want to find a move for them with maxAgent
                else: #else if ghost
                    curScore = maxAgent(state.generateSuccessor(numGhost, action), depth, curChar)
                # if we've found a new best score we can pass it back
                if curScore < minScore:
                    minScore = curScore

            return minScore
        return minAgent(gameState, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # adverserial search
        def maxAgent(state, depth, alpha, beta):
            if terminalTest(state):
                return state.getScore()
            # get all possible actions and initialze the local alpha value to -inf
            actions = state.getLegalActions(0)
            curMin = float("-inf")
            nextBestAct = Directions.STOP
            # for each actions we get the utility value, and once we get new utility values we can get a new action
            for action in actions:
                utilityVal = minAgent(state.generateSuccessor(0, action), depth, 1, alpha, beta)
                curMin = max(curMin, utilityVal);
                # get a new action which scores more points than previously
                if(curMin == utilityVal):
                    nextBestAct = action
                # if we found a new current min then we don't have to go over the next tree
                if curMin > beta:
                    return curMin
                # set new alpha which allows me to make the value searching for next iteration smaller
                alpha = max(alpha, curMin)
            # need to return an action if we're done searching or for ghost
            if depth == 0:
                return nextBestAct
            return curMin

        def minAgent(state, depth, numGhost, alpha, beta):
            if terminalTest(state):
                return state.getScore()
            # the next ghost we will consider
            next_ghost = numGhost + 1
            if numGhost == state.getNumAgents() - 1:
                next_ghost = 0
            actions = state.getLegalActions(numGhost)
            utilityVal = float("inf")
            # for successor state and action, we generate a successor state based on the action we choose
            for action in actions:
                if next_ghost != 0:  # We are on the last ghost and it will be Pacman's turn next.
                    score = minAgent(state.generateSuccessor(numGhost, action), depth, next_ghost, alpha, beta)
                # passes on next alpha and beta values, be careful for local/global variables
                elif depth == self.depth - 1: #consider the ghost scenario
                    score = self.evaluationFunction(state.generateSuccessor(numGhost, action))
                else:
                    score = maxAgent(state.generateSuccessor(numGhost, action), depth + 1, alpha, beta)
                # if we've found a new best score
                if score < utilityVal:
                    utilityVal = score
                # if we found new higher utility val we can skip the rest of the tree
                if utilityVal < alpha:
                    return utilityVal
                beta = min(beta, utilityVal)
                #compare with alpha to see if we need to continue going down the tree

            return utilityVal
        # initialize Alpha-Beta call with alpha -infinity and beta as infinity
        return maxAgent(gameState, 0, float("-inf"), float("inf"))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def starter(state, depth, agent):
            # we are currently on the last ghost and need to switch to pacman
            if agent >= state.getNumAgents():
                agent = 0
                depth = 1 + depth
            # if done exploring or game ended
            if (terminalTest(state) or depth == self.depth):
                return self.evaluationFunction(state)
            elif (agent == 0):
                # if we are on mr pacman then we begin
                return maxFinder(state, depth, agent)
            # pacs job to predict other events and get max score
            return expectedFinder(state, depth, agent)
        # ghosts part of environment, don't count into expectedVal
        def maxFinder(state, depth, agent):

            nextAct = Directions.STOP
            v = -float("inf")
            actions = state.getLegalActions(agent)
            for action in actions:
                # for each action predict its effects and the successors states and scores
                # when found a new best state we set it
                curState = state.generateSuccessor(agent, action)
                curVal = starter(curState, depth, agent + 1)
                # check to see if we have gotten a score or an action and associated score
                if isinstance(curVal, (int, long, float, complex)):
                    val = curVal
                else:
                    val = curVal[1]
                # found new best action, return it and continue recursively
                if val > v:
                    nextAct = action
                    v = val
            return [nextAct, v]

        def expectedFinder(state, deepness, agent):
            nextAct = Directions.STOP
            v = 0
            actions = state.getLegalActions(agent)
            # give uniform probability and return expectations
            probability = 1.0 / len(actions)
            for action in actions:
                curState = state.generateSuccessor(agent, action)
                curVal = starter(curState, deepness, agent + 1)
                # check to see if we have gotten a score or an action and associated score
                if isinstance(curVal, (int, long, float, complex)):
                    val = curVal
                else:
                    val = curVal[1]
                v += val * probability
                nextAct = action

            return [nextAct, v]

        bestAct = starter(gameState, 0, 0)
        return bestAct[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

