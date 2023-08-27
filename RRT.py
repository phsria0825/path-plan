# RRT algorithm Code
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import rcParams

np.set_printoptions(precision=3, suppress=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams['font.size'] = 22


# treeNode class
class treeNode():
    def __init__(self, locationX, locationY):
        self.locationX = locationX  # X Location
        self.locationY = locationY  # Y Location
        self.children = []  # children list
        self.parent = None  # parent node reference


# RRT Algorithm class
class RRTAlgorithm():
    def __init__(self, start, goal, numIterations, grid, stepSize):
        self.randomTree = treeNode(start[0], start[1])                  # The RRT (root position) or (Current position)
        self.goal = treeNode(goal[0], goal[1])                          # goal position
        self.nearestNode = None                                         # nearest node (이전에 찾아둔 가장 가까운 노드) : 현재 위치라고 생각하면 된다.
        self.iterations = numIterations  # number of iterations to run
        self.grid = grid  # the map
        self.rho = stepSize  # length of each branch (노드 확장 및 생성 단계에서 랜덤 위치의 방향으로 얼마의 거리를 이동할 것 인지 크기를 설정)
        self.path_distance = 0  # total path distance
        self.nearestDist = np.inf  # distance to nearest node (initialize with large value)
        self.numWaypoints = 0  # number of waypoints
        self.Waypoints = []  # the waypoints

    # add the node to the nearest node, and add goal if necessary
    def addChild(self, locationX, locationY):
        """
        현재의 위치에서 child 노드와 parent노드를 생성

        Args:
            locationX (float): 현재 노드의 X
            locationY (float): 현재 노드의 Y
        """

        ########################################################################
        # 현재 위치가 목표 위치와 동일한지 확인
        # 만약 동일하다면, 아래의 코드를 실행하여 목표 노드를 가장 가까운 노드의 자식 노드로 추가하고,
        # 목표 노드의 부모를 가장 가까운 노드로 설정
        ########################################################################
        if (locationX == self.goal.locationX and locationY == self.goal.locationY):

            # append goal to nearestNode's children
            # 목표위치를 nearestNode의 children 노드로 추가
            self.nearestNode.children.append(self.goal)

            # and set goal's parent to nearestNode
            # 목표 노드의 부모 노드를 가장 가까운 노드로 설정
            self.goal.parent = self.nearestNode

        ########################################################################
        # 현재 위치가 목표 위치와 동일한지 확인
        # 만약 다르면, 새로운 treeNode 객체를 생성하고,
        # 가장 가까운 노드의 자식 목록에 추가한 후,
        # 해당 노드의 부모를 가장 가까운 노드로 설정
        ########################################################################
        else:
            # create a tree node from locationX, locationY
            node = treeNode(locationX, locationY)  # 랜덤으로 선택된 위치

            # append name 'node' to nearestNode's children
            self.nearestNode.children.append(node)

            # set the parent to nearestNode
            node.parent = self.nearestNode

    # sample random point within grid limits
    def sampleAPoint(self):
        """
        랜덤으로 x,y값을 생성하는 함수

        Returns:
            array : [x, y]
        """
        x = random.randint(1, grid.shape[1])  # 랜덤 x
        y = random.randint(1, grid.shape[0])  # 랜덤 y
        point = np.array([x, y])
        return point

    # steer a distance stepSize from start location to end location (keep in mind the grid limits)
    def steerToPoint(self, locationStart, locationEnd):
        """
        locationStart, locationEnd 위치 사이의 거리가 rho인 새로운 노드를 생성.
        실제로 RRT 알고리즘이 동작할 때, 'nearestNode'에서 'random_point'방향으로 지정된 거리(rho 혹은 stepSize)만큼 이동한 위치에 새로운 노드를 생성.

        현재 노드에서 랜덤한 방향으로 한 걸음 이동한 위치를 계산하고, 그 위치를 트리에 새로운 노드로 추가하는 역할을 한다.

        Args:
            locationStart (object): nearestNode의 위치 값
            locationEnd (array): random_point의 위치 값

        Returns:
            array: offset 된 위치배열
        """
        # 시작, 종료위치 사이의 단위 벡터의 스칼라배
        offsetVector = self.rho * self.unitVector(locationStart, locationEnd)

        # 새로운 점위치 계산, 기존의 위치에서 offsetVector의 [x, y]만큼 이동
        point = np.array([locationStart.locationX + offsetVector[0], locationStart.locationY + offsetVector[1]])

        # 새로운 점의 위치가 grid영억안에 있도록 위치 수정
        if point[0] >= grid.shape[1]:
            point[0] = grid.shape[1] - 1
        if point[1] >= grid.shape[0]:
            point[1] = grid.shape[0] - 1
        return point

    # check if obstacle lies between the start and end point of the edge
    def isInObstacle(self, locationStart, locationEnd):
        """
        'locationStart'와 'locationEnd' 직선경로 사이에 장애물이
        존재하는지 판단하는 함수

        'nearestNode'와 'random_point'에서 생성된 노드의 위치가 장애물위에 있는지 판단

        Args:
            locationStart (list): 시작위치
            locationEnd (list): 종료위치

        Returns:
            bool : 논리값을 반환
        """
        u_hat = self.unitVector(locationStart, locationEnd)  # locationStart에서 locationEnd로 이동하는 단위 벡터를 계산
        testPoint = np.array([0.0, 0.0])  # 장애물을 판단할 변수 초기화
        for i in range(self.rho):  # locationStart와 locationEnd 사이의 각 지점을 self.rho 거리만큼 반복하여 체크

            # testPoint의 x와 y 좌표를 계산합니다. u_hat에 i를 곱하고 locationStart의 x, y 좌표에 더하여 새로운 위치를 계산
            # min 함수를 사용하는 이유는 testPoint가 그리드의 범위를 벗어나지 않게 한다.
            testPoint[0] = min(grid.shape[1] - 1, locationStart.locationX + i * u_hat[0])
            testPoint[1] = min(grid.shape[0] - 1, locationStart.locationY + i * u_hat[1])

            # 장애물과 충돌하면 참, 아니면 거짓을 반환
            if self.grid[np.int64(round(testPoint[1])), np.int64(round(testPoint[0]))] == 1:
                return True
        return False

    # find the unit vector between 2 locations
    def unitVector(self, locationStart, locationEnd):
        """
        두 위치 사이의 단위벡터를 계산

        Args:
            locationStart (list): 시작위치
            locationEnd (list): 종료위치

        Returns:
            list : 단위벡터 반환
        """
        v = np.array([locationEnd[0] - locationStart.locationX, locationEnd[1] - locationStart.locationY])  # 위치벡터 계산
        u_hat = v / np.linalg.norm(v)  # 정규화
        return u_hat

    # find the nearest node from a given (unconnected) point (Euclidean distance)
    def findNearest(self, root, point):
        """
        주어진 점(지금까지지 생성된 경로)에서 가장 가까이 샘플링된 노드를 찾는 함수이다.
        KD-tree나 Quad-tree와 같은 공간 분할 트리 구조에서 일반적으로 사용
        'randomTree'(현재까지 선택된 노드들)와 'random_point'(샘플링된 노드)들 중 가장 가까은 노드를 탐색
        """

        # 트리가 비었는지 체크
        if not root:
            return

        # find distance between root and point use distance method,
        # root 노드와 주어진 점 사이의 거리를 계산
        distance = self.distance(root, point)

        # if it's lower than or equal to nearestDist then update nearestNode to root
        # 측정한 거리가 현재까지 발견한 가장 짧은 거리보다 짧거나 같으면, 가장 가까운 노드와 그 거리를 갱신
        if distance <= self.nearestDist:
            self.nearestNode = root
            self.nearestDist = distance

        # update nearestDist to the distance (it's recursion)
        # 마지막으로, root 노드의 각 자식 노드에 대해 findNearest 함수를 재귀적으로 호출하여,
        # 주어진 점에서 가장 가까운 노드를 탐색
        for child in root.children:
            self.findNearest(child, point)

    # find euclidean distance between a node object and an XY point
    def distance(self, node1, point):
        dist = np.sqrt((node1.locationX - point[0]) ** 2 + (node1.locationY - point[1]) ** 2)
        return dist

    # check if the goal is within stepsize (rho) distance from point, return true if so otherwise false
    def goalFound(self, point):
        if self.distance(self.goal, point) <= self.rho:
            return True
        return False

    # reset: set nearestNode to None and nearestDistance to 10000
    def resetNearestValues(self):
        self.nearestNode = None
        self.nearestDist = 10000

    # trace the path from goal to start
    def retraceRRTPath(self, goal):

        if goal.locationX == self.randomTree.locationX:
            return

        # add 1 to numWaypoints
        self.numWaypoints += 1

        # extract the X Y location of goal in a numpy array
        current_point = np.array([goal.locationX, goal.locationY])

        # insert this array to waypoints (from the beginning)
        self.Waypoints.insert(0, current_point)

        # add rho to path_distance
        self.path_distance += self.rho

        # reculsive next Node
        self.retraceRRTPath(goal.parent)

    # end of class definitions


# ------------------------------------------------------------------------------------------------------------------------#

# load the grid, set start and goal <x, y> positions, number of iterations, step size
grid = np.load('cspace.npy')
start = np.array([100.0, 100.0])
goal = np.array([1600.0, 750.0])
numIterations = 200
stepSize = 200
goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill=False)

fig = plt.figure("RRT Algorithm")
plt.imshow(grid, cmap='binary')
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'bo')
ax = fig.gca()
ax.add_patch(goalRegion)
plt.xlabel('X-axis $(m)$')
plt.ylabel('Y-axis $(m)$')

# Begin
rrt = RRTAlgorithm(start, goal, numIterations, grid, stepSize)
plt.pause(2)

# RRT algorithm
# iterate
for i in range(rrt.iterations):
    # Reset nearest values, call the resetNearestValues method
    rrt.resetNearestValues()
    print("Iteration: ", i)

    # sample a point (use the appropriate method, store the point in variable)- call this variable 'random_point' to match Line 151
    random_point = rrt.sampleAPoint()

    # find the nearest node w.r.t to the point (just call the method do not return anything)
    """
    findNearest 함수는 트리에서 가장 가까운 노드를 찾는 역할을 하며, 
    steerToPoint 함수는 가장 가까운 노드(즉, findNearest 함수에 의해 찾아진 노드)에서 랜덤하게 선택한 점(random_point) 방향으로 새로운 노드를 생성하는 역할을 한다.
    이 두 과정을 통해 RRT 트리는 랜덤하게 샘플링된 점들을 통해 확장되며, 이는 결국 탐색 공간을 효과적으로 탐색하게 된다.
    """
    rrt.findNearest(rrt.randomTree, random_point)
    new = rrt.steerToPoint(rrt.nearestNode, random_point)  # steer to a point, return as 'new'

    # if not in obstacle
    if not rrt.isInObstacle(rrt.nearestNode, new):

        # add new to the nearestnode (addChild), again no need to return just call the method
        rrt.addChild(new[0], new[1])
        plt.pause(0.10)
        plt.plot([rrt.nearestNode.locationX, new[0]], [rrt.nearestNode.locationY, new[1]], 'go', linestyle="--")

        # if goal found (new is within goal region)
        if (rrt.goalFound(new)):
            # append goal to path
            rrt.addChild(goal[0], goal[1])
            # retrace
            rrt.retraceRRTPath(rrt.goal)
            break

# Add start to waypoints
rrt.Waypoints.insert(0, start)
print("Number of waypoints: ", rrt.numWaypoints)
print("Path Distance (m): ", rrt.path_distance)
print("Waypoints: ", rrt.Waypoints)

# plot the waypoints in red
for i in range(len(rrt.Waypoints) - 1):
    plt.plot([rrt.Waypoints[i][0], rrt.Waypoints[i + 1][0]], [rrt.Waypoints[i][1], rrt.Waypoints[i + 1][1]], 'ro',
             linestyle="--")
    plt.pause(0.10)