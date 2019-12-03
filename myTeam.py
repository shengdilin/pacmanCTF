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
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    We should have offence and defense/ also remember to check teams
    '''

    return random.choice(actions)


#-----------------------------------OUR AGENTS-----------------------------------------------


class MyCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  capsuleTimer = 0.0

 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    action = random.choice(bestActions)

    #Count down capsule timer if it is on
    if(self.capsuleTimer > 0):
      self.capsuleTimer -= 1

    if(self.getSuccessor(gameState, action).getAgentState(self.index).getPosition() in self.getFood(gameState).asList()):
      self.dotsEaten += 1

    return action

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}



class OffensiveAgent(MyCaptureAgent):
  """
  We want our offensive agent to do the following:

  -Collect dots on the enemy side while avoiding enemy ghosts, unless they eat a capsule
    -Use inference if they are over 5 away, otherwise avoid via direct observation
    -Return to their side after collecting a certain number of dots


  """
  dotsEaten = 0.0


  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    prevFoodList = self.getFood(gameState).asList()
    features['successorScore'] = self.getScore(successor)
    prevCapsuleList = self.getCapsules(gameState)
    capsuleList = self.getCapsules(successor)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in prevFoodList])
      features['distanceToFood'] = minDistance

    # Compute amount of food remaining to be eaten
    #features['foodLeft'] = len(foodList)

    # Check if action involves eating food
    

    #Reset dots eaten counter
    if(self.getScore(successor) > self.getScore(gameState)):
      self.dotsEaten = 0
    if(myPos == self.start):
      self.dotsEaten = 0

    #Compute distance to friendly side (spawn)
    features['distanceHome'] = self.getMazeDistance(myPos, self.start)

    # Compute distance to capsules
    if len(capsuleList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
      features['distanceToCapsule'] = minDistance

    #Give capsule status
    if(len(prevCapsuleList) > len(capsuleList)):
      capsuleTimer = 40

    #Compute distance to nearest enemy (if visible)
    visibleEnemies = []

    #Only count if pacman is on opposite side
    if(myState.isPacman):
      for enemy in self.getOpponents(successor):
        if gameState.getAgentPosition(enemy) is not None:
          visibleEnemies.append(gameState.getAgentPosition(enemy))

      enemyDists = []
      for enemy in visibleEnemies:
        enemyDists.append(self.getMazeDistance(successor.getAgentPosition(self.index), enemy))

      if(len(enemyDists) > 0):
        features['enemyDistance'] = min(enemyDists)

        if(min(enemyDists) < 3):
          features['enemyNear'] = 1
        else:
          features['enemyNear'] = 0
      else:
        features['enemyDistance'] = 0
    #If you are on same side, reward eating enemies

    #Use inference to estimate distance otherwise

    return features

  def getWeights(self, gameState, action):
    

    #Change sign/magnitude of certain weights if pacman has eaten the capsule or if it collects a certain number of dots
    if(self.dotsEaten > 5):
      return {'distanceHome': -10, 'enemyDistance': -50}
    if(self.capsuleTimer > 0):
      print('ghost')
      return {'successorScore': 100, 'distanceToFood': -1, 'distanceToCapsule': 0, 'enemyDistance': -10, 'enemyNear': 0}
    else:
      return {'successorScore': 100, 'distanceToFood': -1, 'distanceToCapsule': -1, 'enemyDistance': 10, 'enemyNear': -20}




class DefensiveAgent(MyCaptureAgent):
  """
  We want our defensive agent to do the following:

  -Follow the enemy agent closest to their side - use inference if they are over 5 away, otherwise follow directly
    -We can check the change in dots on our side to precisely see where an enemy is

  -Stay on their side, play defensively


  """
  def getFeatures(self, gameState, action):
    #features of defender
    features = util.Counter()
    #successor state
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    #determine invaders
    invaders = [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() != None]
    #keeps record of how many invaders
    features['numInvaders'] = len(invaders)
    #if there is at least one invader 
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  

