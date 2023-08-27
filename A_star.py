import math  # 수학 연산을 위한 파이썬 내장 모듈
import heapq  # 힙 큐 알고리즘을 위한 파이썬 내장 모듈
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 matplotlib의 pyplot 모듈
import numpy as np  # 효과적인 수치 연산을 위한 numpy 모듈

class HBF:  # Hybrid A*를 구현하는 메인 클래스
    def __init__(self):  # 초기화 함수
        self.NUM_THETA_CELLS = 90  # 각도의 셀 수
        self.SPEED = 1.45  # 기본 이동 속도
        self.LENGTH = 0.5  # 차량 길이

    class maze_s:  # 지도를 표현하는 클래스
        def __init__(self, f, g, x, y, theta):  # 초기화 함수
            self.f = f  # 총 비용 (g+h)
            self.g = g  # 이동 비용
            self.x = x  # x 좌표
            self.y = y  # y 좌표
            self.theta = theta  # 각도

        def __lt__(self, other):  # 두 개체를 비교하는 함수 (힙 큐를 위해 필요)
            return self.f < other.f  # 비용 f를 기준으로 비교

    class maze_path:  # 최종 경로를 표현하는 클래스
        def __init__(self, closed, came_from, final):  # 초기화 함수
            self.closed = closed  # 방문한 노드를 저장하는 배열
            self.came_from = came_from  # 경로 추적을 위한 배열
            self.final = final  # 최종 노드

    def theta_to_stack_number(self, theta):  # 각도를 셀 번호로 변환하는 함수
        new_theta = (theta + 2 * math.pi) % (2*math.pi)  # 2π로 나눈 나머지를 이용해 각도를 [0, 2π) 범위로 변환
        stack_number = round(new_theta * self.NUM_THETA_CELLS / (2*math.pi)) % self.NUM_THETA_CELLS  # 각도에 해당하는 셀 번호 계산
        return stack_number  # 셀 번호 반환

    def idx(self, float_num):  # 소수점 이하를 버리는 함수
        return int(math.floor(float_num))  # 소수점 이하 버림

    def heuristic(self, x, y, goal):  # 휴리스틱 함수 (목표까지의 추정 비용)
        return abs(y - goal[0]) + abs(x - goal[1])  # 맨해튼 거리 사용

    def expand(self, state, goal):  # 주어진 상태에서 가능한 다음 상태를 확장하는 함수
        g = state.g  # 현재까지의 이동 비용
        x = state.x  # 현재 위치 x
        y = state.y  # 현재 위치 y
        theta = state.theta  # 현재 각도
        g2 = g + 1  # 다음 상태의 이동 비용
        next_states = []  # 다음 상태를 저장할 리스트
        for delta_i in range(-35, 40, 5):  # -35도에서 35도까지, 5도 간격으로 루프
            delta = math.pi / 180.0 * delta_i  # 각도를 라디안으로 변환
            omega = self.SPEED / self.LENGTH * math.tan(delta)  # 각속도 계산
            theta2 = theta + omega  # 다음 상태의 각도
            if theta2 < 0:
                theta2 += 2*math.pi  # 각도가 음수인 경우, 2π를 더해 [0, 2π) 범위로 변환
            x2 = x + self.SPEED * math.cos(theta)  # 다음 상태의 x 좌표
            y2 = y + self.SPEED * math.sin(theta)  # 다음 상태의 y 좌표
            state2 = self.maze_s(g2 + self.heuristic(x2, y2, goal), g2, x2, y2, theta2)  # 새로운 상태 생성
            next_states.append(state2)  # 다음 상태 리스트에 추가
        return next_states  # 다음 상태 리스트 반환

    def reconstruct_path(self, came_from, start, final):  # 경로를 재구성하는 함수
        path = [final]  # 경로 리스트, 최종 노드부터 시작
        stack = self.theta_to_stack_number(final.theta)  # 최종 노드의 셀 번호 계산
        current = came_from[stack][self.idx(final.x)][self.idx(final.y)]  # 최종 노드의 이전 노드
        stack = self.theta_to_stack_number(current.theta)  # 이전 노드의 셀 번호 계산
        x = current.x  # 이전 노드의 x 좌표
        y = current.y  # 이전 노드의 y 좌표
        while x != start[0] or y != start[1]:  # 시작 노드에 도달할 때까지 반복
            path.append(current)  # 경로에 현재 노드 추가
            current = came_from[stack][self.idx(x)][self.idx(y)]  # 현재 노드의 이전 노드
            x = current.x  # 이전 노드의 x 좌표
            y = current.y  # 이전 노드의 y 좌표
            stack = self.theta_to_stack_number(current.theta)  # 이전 노드의 셀 번호 계산
        return path  # 경로 반환

    def search(self, grid, start, goal):  # A* 탐색 알고리즘 구현 함수
        # 각 그리드 셀에 대해 각각의 방향에 대해 그리드 셀을 방문했는지 여부를 저장하기 위한 3차원 리스트를 초기화합니다.
        closed = [[[0 for x in range(len(grid[0]))] for y in range(len(grid))] for cell in range(self.NUM_THETA_CELLS)]

        # 각 그리드 셀에 대해 각각의 방향에 대해 그리드 셀로 이동하기 위해 어디서 왔는지를 저장하기 위한 3차원 리스트를 초기화합니다.
        came_from = [[[0 for x in range(len(grid[0]))] for y in range(len(grid))] for cell in
                     range(self.NUM_THETA_CELLS)]

        # 시작 상태의 방향(theta)를 가져옵니다.
        theta = start[2]

        # 시작 상태의 방향에 대한 스택 번호를 계산합니다.
        stack = self.theta_to_stack_number(theta)

        # 시작 상태에서의 비용을 0으로 설정합니다.
        g = 0

        # 시작 상태를 생성합니다. 휴리스틱 값, 이동 비용, x 위치, y 위치, 방향을 가지고 있습니다.
        state = self.maze_s(g + self.heuristic(start[0], start[1], goal), g, start[0], start[1], theta)

        # 시작 상태를 닫힌 리스트에 추가합니다.
        closed[stack][self.idx(state.x)][self.idx(state.y)] = 1

        # 시작 상태의 이전 상태를 자기 자신으로 설정합니다.
        came_from[stack][self.idx(state.x)][self.idx(state.y)] = state

        # 닫힌 리스트의 상태 개수를 1로 설정합니다.
        total_closed = 1

        # 열린 리스트를 생성하고 시작 상태를 추가합니다.
        opened = [state]

        # 목표를 찾았는지의 여부를 저장하는 변수를 설정합니다.
        finished = False

        # 열린 리스트에 상태가 있을 동안 실행합니다.
        while not len(opened) == 0:
            # 비용이 가장 낮은 상태를 가져와서 현재 상태로 설정합니다.
            opened.sort()
            current = opened.pop(0)

            # 현재 상태의 x, y 좌표를 가져옵니다.
            x = current.x
            y = current.y

            # 현재 상태가 목표 상태이면 경로를 찾았다는 메시지를 출력하고 경로를 반환합니다.
            if self.idx(x) == goal[0] and self.idx(y) == goal[1]:
                print("found path to goal in {} expansions".format(total_closed))
                path = self.maze_path(closed, came_from, current)
                return path

            # 가능한 다음 상태들을 가져옵니다.
            next_states = self.expand(current, goal)

            # 가능한 다음 상태들에 대해 반복합니다.
            for i in range(len(next_states)):
                # 다음 상태의 비용, x, y 좌표, 방향을 가져옵니다.
                g2 = next_states[i].g
                x2 = next_states[i].x
                y2 = next_states[i].y
                theta2 = next_states[i].theta

                # 다음 상태의 x, y 좌표가 그리드 범위를 벗어나면 해당 상태를 무시합니다.
                if x2 < 0 or x2 >= len(grid) or y2 < 0 or y2 >= len(grid[0]):
                    continue

                # 다음 상태의 방향에 대한 스택 번호를 계산합니다.
                stack2 = self.theta_to_stack_number(theta2)

                # 다음 상태가 닫힌 리스트에 없고, 그리드 셀이 장애물이 아니라면 다음 상태를 열린 리스트에 추가하고, 닫힌 리스트에 추가하고, 이전 상태를 현재 상태로 설정합니다.
                if closed[stack2][self.idx(x2)][self.idx(y2)] == 0 and grid[self.idx(x2)][self.idx(y2)] == 0:
                    opened.append(next_states[i])
                    closed[stack2][self.idx(x2)][self.idx(y2)] = 1
                    came_from[stack2][self.idx(x2)][self.idx(y2)] = current
                    total_closed += 1

        # 열린 리스트가 비어있다면 유효한 경로가 없다는 메시지를 출력하고, 시작 상태부터의 경로를 반환합니다.
        print("no valid path.")
        path = self.maze_path(closed, came_from, state)
        return path

def draw(grid, path, start, goal):
    grid = np.array(grid)
    path = [(step.x, step.y) for step in path]

    plt.imshow(grid, cmap=plt.cm.Dark2)
    plt.scatter(start[1], start[0], marker='o', color='blue', s=200)
    plt.scatter(goal[1], goal[0], marker='o', color='red', s=200)

    path_points = list(zip(*path))
    plt.plot(path_points[1], path_points[0], color='blue')

    plt.show()

# 테스트 코드
if __name__ == "__main__":
    X = 1
    _ = 0

    original_GRID = np.array([
        [_, X, X, _, _, _, _, _, _, _, X, X, _, _, _, _],
        [_, X, X, _, _, _, _, _, _, X, X, _, _, _, _, _],
        [_, X, X, _, _, _, _, _, X, X, _, _, _, _, _, _],
        [_, X, X, _, _, _, _, X, X, _, _, _, X, X, X, _],
        [_, X, X, _, _, _, X, X, _, _, _, X, X, X, _, _],
        [_, X, X, _, _, X, X, _, _, _, X, X, X, _, _, _],
        [_, X, X, _, X, X, _, _, _, X, X, X, _, _, _, _],
        [_, X, X, X, X, _, _, _, X, X, X, _, _, _, _, _],
        [_, X, X, X, _, _, _, X, X, X, _, _, _, _, _, _],
        [_, X, X, _, _, _, X, X, X, _, _, X, X, X, X, X],
        [_, X, _, _, _, X, X, X, _, _, X, X, X, X, X, X],
        [_, _, _, _, X, X, X, _, _, X, X, X, X, X, X, X],
        [_, _, _, X, X, X, _, _, X, X, X, X, X, X, X, X],
        [_, _, X, X, X, _, _, X, X, X, X, X, X, X, X, X],
        [_, X, X, X, _, _, _, _, _, _, _, _, _, _, _, _],
        [X, X, X, _, _, _, _, _, _, _, _, _, _, _, _, _]])

    # Creating the expanded grid
    expanded_GRID = np.repeat(np.repeat(original_GRID, 5, axis=0), 5, axis=1)

    # print(expanded_GRID)

    # 시작 지점, 목표 지점 설정
    START = [0.0, 0.0, 0.0]
    GOAL = [len(expanded_GRID)-1, len(expanded_GRID[0])-1]

    hbf = HBF()
    get_path = hbf.search(expanded_GRID, START, GOAL)

    show_path = hbf.reconstruct_path(get_path.came_from, START, get_path.final)

    print("show path from start to finish")
    for i in range(len(show_path)-1, -1, -1):
        step = show_path[i]
        print("##### step {} #####".format(step.g))
        print("x {}".format(step.x))
        print("y {}".format(step.y))
        print("theta {}".format(step.theta))

    draw(expanded_GRID, show_path, START, GOAL)