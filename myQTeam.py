# myTeam.py
# ---------
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

from captureAgents import CaptureAgent
from baselineTeam import ReflexCaptureAgent
import distanceCalculator
from game import Directions
import game
from util import nearestPoint
from game import Actions
import random, time, util, sys


'''
Agents created using approximate q-agent

Both can go offense defense
One agent prefers the top, one preferers the bottom
Similar to zone defence in basketball

Tried to blitz the other side to get as many pellets as possible
But will do a little defense once acquired a few pellets
To ensure that the other side is managed a little
'''

#initialize beliefs 
beliefs = []
#and keep another set for initialized beliefs 
beliefsInitialized = []

#create teams 
def createTeam(firstIndex, secondIndex, isRed,
               first = 'TopAgent', second = 'BottomAgent', **args):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

#create our approximate q-agent
class ApproximateQAgent(CaptureAgent):

  def __init__( self, index ):
    CaptureAgent.__init__(self, index)
    #create weights counter
    self.weights = util.Counter()
    self.numTraining = 0
    self.episodesSoFar = 0
    self.epsilon = 0.05
    self.discount = 0.8
    self.alpha = 0.2

  def getSuccessor(self, gameState, action):
    #Finds the next successor which is a grid position (location tuple).
    successor = gameState.generateSuccessor(self.index, action)
    #find position
    pos = successor.getAgentState(self.index).getPosition()
    #get successor game state
    if pos != nearestPoint(pos):
      # only half the grid was considered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def registerInitialState(self, gameState):
    #get starting position
    self.start = gameState.getAgentPosition(self.index)
    #no last action 
    self.lastAction = None
    #registers initial state 
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, state):
    # Append game state to observation history...
    self.observationHistory.append(state)
    # Pick Action
    legalActions = state.getLegalActions(self.index)
    action = None
    if len(legalActions):
      #flip based on q-values and epsilon
        action = self.computeActionFromQValues(state)
    
    #find how many pellets are left
    foodLeft = len(self.getFood(state).asList())

    #store last action
    self.lastAction = action

    # if we have two or less pellets left, go towards start (more defensive)
    if foodLeft <= 2:
      #initialize distance away
      bestDist = 10000
      for a in legalActions:
        #get successor states and position
        successor = self.getSuccessor(state, a)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          # if actual distance less than best distance we choose action
          action = a
          bestDist = dist
    #return action
    return action

  def getFeatures(self, gameState, action):
    #initialize features 
    features = util.Counter()
    #generate successor state
    successor = self.getSuccessor(gameState, action)
    #find 
    features['score'] = self.getScore(successor)
    #if not red, score is not negative 
    if not self.red:
      features['score'] *= -1
    #find available choices 
    features['choices'] = len(successor.getLegalActions(self.index))
    #return featuresf
    return features

  def computeActionFromQValues(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    #set initial action to none
    bestActions = None
    #set initial value to big negative
    bestValue = -100000
    #for all legal actions
    for action in state.getLegalActions(self.index):
      #find the action's value
      value = self.getQValue(state, action)
      #if value better than value
      if value > bestValue:
        #update best action and value accordingly
        bestActions = [action]
        bestValue = value
      #else if its tied with best actions 
      elif value == bestValue:
        bestActions.append(action)
        #no legal actions, stop
    if bestActions == None:
      return Directions.STOP 
    #else choose a random best action (from tied best value)
    return random.choice(bestActions) 


  def getWeights(self):
    #return weights of self
    return self.weights

  def computeValueFromQValues(self, state):
    """
      Returns max-value of actions 
    """
    #set initial best Value to big negative
    bestValue = -100000
    #set legal actions to true, will update if action is available
    noLegalActions = True
    for action in state.getLegalActions(self.index):
      #if there is an action, find value and update
      #also make nolegalActions false because there is a legal action 
      noLegalActions = False
      value = self.getQValue(state, action)
      #update best value when appropriate
      if value > bestValue:
        bestValue = value
    #return value of 0 if no legal actions found
    if noLegalActions:
      return 0 
    #else return the best value found 
    return bestValue

  def getQValue(self, state, action):
    """
      returns q-value
    """
    total = 0
    weights = self.getWeights()
    features = self.getFeatures(state, action)
    for feature in features:
      # Implements the Q calculation
      total += features[feature] * weights[feature]
    return total

  def getReward(self, gameState):
    foodList = self.getFood(gameState).asList()
    return -len(foodList)

  def observationFunction(self, gameState):
    if len(self.observationHistory) > 0 and self.isTraining():
      self.update(self.getCurrentObservation(), self.lastAction, gameState, self.getReward(gameState))
    return gameState.makeObservation(self.index)

  def isTraining(self):
    return self.episodesSoFar < self.numTraining

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    #find difference 
    difference = (reward + self.discount * self.computeValueFromQValues(nextState))
    #subtract own state's q-value 
    difference -= self.getQValue(state, action)

    # Only calculate the difference once, not in the loop.
    newWeights = self.weights.copy()
    # Same with weights and features. 
    features = self.getFeatures(state, action)
    for feature in features:
      # Implements the weight updating calculations
      newWeight = newWeights[feature] + self.alpha * difference * features[feature]
      newWeights[feature]  = newWeight
    #updates weights 
    self.weights = newWeights.copy()


'''
Our gameAgent holds many weights and draws from the q-agent:
'legalActions': We value going to spaces that have more actions
'powerPelletValue': Power pellets are valued
'ghostDistance' - We value moving away from enemy non-scared ghosts
'backToSafeZone': Valued when chased by ghost or holding many pellets
'chaseEnemyValue': Chase Enemy if close 
'successorScore' - we highly value eating our pellets
'distanceToFood' - we value moving closer to a certain pellet
'stop': We penalize no movement
'''
class gameAgent(ApproximateQAgent):
  
  def registerInitialState(self, gameState):
    ApproximateQAgent.registerInitialState(self, gameState)
    self.favoredY = 0.0
    self.defenseTimer = 0.0
    self.lastNumReturnedPellets = 0.0
    self.getLegalPositions(gameState)

  def __init__( self, index ):
    ApproximateQAgent.__init__(self, index)
    self.weights = util.Counter()
    self.weights['successorScore'] = 100
    self.weights['distanceToFood'] = -1
    self.weights['ghostDistance'] = 5
    self.weights['stop'] = -1000
    self.weights['legalActions'] = 60
    self.weights['powerPelletValue'] = 100
    self.distanceToTrackPowerPelletValue = 3
    self.weights['backToSafeZone'] = -1
    self.minPelletsToCashIn = 9
    self.weights['chaseEnemyValue'] = -200
    self.chaseEnemyDistance = 5
    self.threatenedDistance = 5
    self.legalActionMap = {}
    self.legalPositionsInitialized = False
    self.weights['eatValue'] = 100

  def getWinningBy(self, gameState):
    #if red, score is score
    if self.red:
      return gameState.getScore()
      #else if blue, the score should be negative 
    else:
      return -1 * gameState.getScore()

  def getLegalActions(self, gameState):
    #find current position
    currentPos = gameState.getAgentState(self.index).getPosition()
    #get legal actions 
    if currentPos not in self.legalActionMap:
      self.legalActionMap[currentPos] = gameState.getLegalActions(self.index)
    return self.legalActionMap[currentPos]

  def getLegalPositions(self, gameState):
    #find legal positions
    #taking into account walls 
    if not self.legalPositionsInitialized:
      self.legalPositions = []
      walls = gameState.getWalls()
      for x in range(walls.width):
        for y in range(walls.height):
          if not walls[x][y]:
            self.legalPositions.append((x, y))
      self.legalPositionsInitialized = True
    return self.legalPositions

  # If we are near the end of the game, we determine if we should just cash in pellets
  def shouldRunHome(self, gameState):
    #find how much winning by
    winningBy = self.getWinningBy(gameState)
    #find num carrying of agent
    numCarrying = gameState.getAgentState(self.index).numCarrying
    #if time beforef
    return (gameState.data.timeleft < 83
      #and not winning 
      and winningBy <= 0 
      #and carrying something 
      and numCarrying > 0 
      #and carrying greater than winningby
      and numCarrying >= abs(winningBy))
    #run home

  def getFeatures(self, gameState, action):
    #observe the opponents
    self.observeAllOpponents(gameState)
    #initialize all features, positions, state, and successors
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True, but better safe than sorry
      distance = min([self.getDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = distance

    # Find all enemies and their ghost/pacman states
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    enemyPacmen = [a for a in enemies if a.isPacman and a.getPosition() != None]
    nonScaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and not a.scaredTimer > 0]
    scaredGhosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]

    # computes distance of ghosts that are 'seen'
    dists = []
    #for each opponent
    for index in self.getOpponents(successor):
      #get state
      enemy = successor.getAgentState(index)
      if enemy in nonScaredGhosts:
        #not a scary ghost
        dists.append(self.getMazeDistance(myPos, self.getMostLikelyGhostPosition(index)))
    # Use the smallest distance
    if len(dists) > 0:
      #find smallest Dist
      smallestDist = min(dists)
      #add to features 
      features['ghostDistance'] = smallestDist

    #update features of pellets and enemies accordingly
    features['powerPelletValue'] = self.getPowerPelletValue(myPos, successor, scaredGhosts)
    features['chaseEnemyValue'] = self.getChaseEnemyWeight(myPos, enemyPacmen)
    
    # If we cashed in any pellets, we shift over to defense mode for a time
    #if myState.numReturned != self.lastNumReturnedPellets:
      #do some defense for a time
      #self.defenseTimer = 100
      #self.lastNumReturnedPellets = myState.numReturned
    # If on defense, heavily value chasing after enemies
    #if self.defenseTimer > 0:
      #self.defenseTimer -= 1
      #increase chase enemy value
      #features['chaseEnemyValue'] *= 100
    #print(features['chaseEnemyValue'])

    # Check if we eat enemy
    if(self.getEatValue(myPos, enemyPacmen)):
      print('eat')
      features['eatValue'] = 1

    # If our opponents ate almost all our food, chase the enemy 
    if len(self.getFoodYouAreDefending(successor).asList()) <= 2:
      #increase chase enemy value
      features['chaseEnemyValue'] *= 100

    # Stopping is penalized
    if action == Directions.STOP: 
      #set stop to 1 if action is stop
        features['stop'] = 1
    
    # total of the legal actions you can take from where you are and the legal actions you can take in all future states
    features['legalActions'] = self.getLegalActionModifier(gameState, 1)

    # pellets eaten adds value
    features['backToSafeZone'] = self.getCashInValue(myPos, gameState, myState)
    
    # going back to safe zone adds value
    features['backToSafeZone'] += self.getBackToStartDistance(myPos, features['ghostDistance'])

    # should run home impacts backToSafeZone
    if self.shouldRunHome(gameState):
      features['backToSafeZone'] = self.getMazeDistance(self.start, myPos) * 10000

    #return all features 
    return features


  # Adds (maze distance) to (the difference in y between the food and our favored y)
  def getDistance(self, myPos, food):
    return self.getMazeDistance(myPos, food) + abs(self.favoredY - food[1])

  def getPowerPelletValue(self, myPos, successor, scaredGhosts):
    #we want to eat power pellets if no scared ghosts
    powerPellets = self.getCapsules(successor)
    #initialize minimum distance
    minDistance = 0
    #do the pellet math
    if len(powerPellets) > 0 and len(scaredGhosts) == 0:
      distances = [self.getMazeDistance(myPos, pellet) for pellet in powerPellets]
      minDistance = min(distances)
    return max(self.distanceToTrackPowerPelletValue - minDistance, 0)

  def getCashInValue(self, myPos, gameState, myState):
    # if we have enough pellets, try to return to safe zone 
    if myState.numCarrying >= self.minPelletsToCashIn:
      #also account for distance away
      return self.getMazeDistance(self.start, myPos)
    else:
      #else has no weight 
      return 0

  #return the weights
  def getWeights(self):
    return self.weights

  # get chasing enemy weight 
  def getChaseEnemyWeight(self, myPos, enemyPacmen):
    if len(enemyPacmen) > 0:
      # Computes distance to enemy pacmen that we can see
      dists = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemyPacmen]
      # Use the smallest distance to chase enemy
      if len(dists) > 0:
        smallestDist = min(dists)
        return smallestDist
    return 0

  # determines if you eat a PacMan
  def getEatValue(self, myPos, enemyPacmen):
    if len(enemyPacmen) > 0:
      # Computes distance to enemy pacmen that we can see
      for enemy in enemyPacmen:
        #print('my pos:')
        #print(myPos)
        #print('enemy pos:')
        #print(enemy.getPosition())
        if(self.getMazeDistance(myPos, enemy.getPosition()) == 0 ):
          return True
    return False

  # find distance back if threatened/cornered
  def getBackToStartDistance(self, myPos, smallestGhostPosition):
    if smallestGhostPosition > self.threatenedDistance or smallestGhostPosition == 0:
      return 0
    else:
      return self.getMazeDistance(self.start, myPos) * 1000

  # Find most likely ghost position
  def getMostLikelyGhostPosition(self, ghostAgentIndex):
    return max(beliefs[ghostAgentIndex])

  # find possible actions and modify 
  def getLegalActionModifier(self, gameState, numLoops):
    legalActions = self.getLegalActions(gameState)
    numActions = len(legalActions)
    for legalAction in legalActions:
      if numLoops > 0:
        newState = self.getSuccessor(gameState, legalAction)
        numActions += self.getLegalActionModifier(newState, numLoops - 1)
    return numActions

  '''
  Alter beliefs of opponents
  '''

  #initialize our beliefs for opponents
  def initializeBeliefs(self, gameState):
    beliefs.extend([None for x in range(len(self.getOpponents(gameState)) + len(self.getTeam(gameState)))])
    for opponent in self.getOpponents(gameState):
      self.initializeBelief(opponent, gameState)
    beliefsInitialized.append('done')

  #belief based on belief from position
  def initializeBelief(self, opponentIndex, gameState):
    belief = util.Counter()
    for p in self.getLegalPositions(gameState):
      belief[p] = 1.0
    belief.normalize()
    beliefs[opponentIndex] = belief

    #find one opponnent
  def observeOneOpponent(self, gameState, opponentIndex):
    #get our position
    pacmanPosition = gameState.getAgentPosition(self.index)
    allPossible = util.Counter()
    # potential position 
    maybeDefinitePosition = gameState.getAgentPosition(opponentIndex)
    #if there is one
    if maybeDefinitePosition != None:
      allPossible[maybeDefinitePosition] = 1
      beliefs[opponentIndex] = allPossible
      #return modified beliefs
      return
    noisyDistance = gameState.getAgentDistances()[opponentIndex]
    for p in self.getLegalPositions(gameState):
      # fach legal ghost position, calculate distance to that ghost
      #use manhattan distance
      trueDistance = util.manhattanDistance(p, pacmanPosition)
      #get the probability as well 
      modelProb = gameState.getDistanceProb(trueDistance, noisyDistance) 
      if modelProb > 0:
        oldProb = beliefs[opponentIndex][p]
        #add a minimum probability in case so it doesn't do something stupid if prob super low
        allPossible[p] = (oldProb + 0.0001) * modelProb
      else:
        allPossible[p] = 0
    allPossible.normalize()
    beliefs[opponentIndex] = allPossible

  #for all opponents 
  def observeAllOpponents(self, gameState):
    if len(beliefsInitialized):
      #add opponent to observedf
      for opponent in self.getOpponents(gameState):
        self.observeOneOpponent(gameState, opponent)
    else:
      self.initializeBeliefs(gameState)


    #use q agent
  def observationFunction(self, gameState):
    return ApproximateQAgent.observationFunction(self, gameState)

# Agent that favors lower side
class BottomAgent(gameAgent):
  def registerInitialState(self, gameState):
    gameAgent.registerInitialState(self, gameState)
    self.favoredY = 0.0

# Agent that favors higher side
class TopAgent(gameAgent):
  def registerInitialState(self, gameState):
    gameAgent.registerInitialState(self, gameState)
    self.favoredY = gameState.data.layout.height
    
