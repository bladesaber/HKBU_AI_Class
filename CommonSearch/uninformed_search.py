from CommonSearch.node import Node
import copy

class BFS:
    # Breath First Search

    def __init__(self, maze, start, end):
        '''
        :param maze: a map
        :param start: (position1, position2)
        :param end: [(position1, position2), (position3, position4)]
        '''
        self.maze = maze
        self.start = start
        self.end = end

        self.h, self.w = self.maze.shape

    def isValid(self, point):
        return point[0] >= 0 and point[0] < self.h \
               and point[1] >= 0 and point[1] < self.w \
               and self.maze[point[1]][point[0]] == 0

    def start_search(self):
        position_list = [Node(self.start, [], 1)]
        explored = 0

        while len(position_list) > 0:
            current = position_list.pop(0)
            x, y = current.point

            if current.point == self.end:
                current.path.append(current.point)
                self.maze[y][x] = 3
                explored += 1
                break

            path = current.path[:]
            path.append(current.point)

            # up
            if self.isValid((x, y - 1)):
                position_list.append(Node((x, y - 1), path, current.cost + 1))
                self.maze[y - 1][x] = 2
            # down
            if self.isValid((x, y + 1)):
                position_list.append(Node((x, y + 1), path, current.cost + 1))
                self.maze[y + 1][x] = 2
            # left
            if self.isValid((x - 1, y)):
                position_list.append(Node((x - 1, y), path, current.cost + 1))
                self.maze[y][x - 1] = 2
            # right
            if self.isValid((x + 1, y)):
                position_list.append(Node((x + 1, y), path, cost=current.cost + 1))
                self.maze[y][x + 1] = 2

            self.maze[y][x] = 3
            explored += 1

        return current, explored

class DFS:
    # Depth First Search

    def __init__(self, maze, start, end):
        '''
        :param maze: a map
        :param start: (position1, position2)
        :param end: [(position1, position2), (position3, position4)]
        '''
        self.maze = maze
        self.start = start
        self.end = end

        self.h, self.w = self.maze.shape

    def isValid(self, point):
        return point[0] >= 0 and point[0] < self.h \
               and point[1] >= 0 and point[1] < self.w \
               and self.maze[point[1]][point[0]] == 0

    def start_search(self):
        position_list = [Node(self.start, [], 1)]
        explored = 0

        while len(position_list) > 0:
            current = position_list.pop()
            x, y = current.point

            if current.point == self.end:
                current.path.append(current.point)
                self.maze[y][x] = 3
                explored += 1
                break

            path = current.path[:]
            path.append(current.point)

            # up
            if self.isValid((x, y - 1)):
                position_list.append(Node((x, y - 1), path, current.cost + 1))
                self.maze[y - 1][x] = 2
            # down
            if self.isValid((x, y + 1)):
                position_list.append(Node((x, y + 1), path, current.cost + 1))
                self.maze[y + 1][x] = 2
            # left
            if self.isValid((x - 1, y)):
                position_list.append(Node((x - 1, y), path, current.cost + 1))
                self.maze[y][x - 1] = 2
            # right
            if self.isValid((x + 1, y)):
                position_list.append(Node((x + 1, y), path, cost=current.cost + 1))
                self.maze[y][x + 1] = 2

            self.maze[y][x] = 3
            explored += 1

        return current, explored

class DLS:
    # Depth Limited Search

    def __init__(self, maze, start, end):
        '''
        :param maze: a map
        :param start: (position1, position2)
        :param end: [(position1, position2), (position3, position4)]
        '''
        self.maze = maze
        self.start = start
        self.end = end

        self.h, self.w = self.maze.shape

    def isValid(self, point):
        return point[0] >= 0 and point[0] < self.h \
               and point[1] >= 0 and point[1] < self.w \
               and self.maze[point[1]][point[0]] == 0

    def start_search(self, limit):
        assert limit>0

        position_list = [Node(self.start, [], 1)]
        explored = 0
        isFound = False

        while len(position_list) > 0:
            current = position_list.pop()
            x, y = current.point

            if current.point == self.end:
                current.path.append(current.point)
                self.maze[y][x] = 3
                explored += 1
                isFound = True
                break

            path = current.path[:]
            path.append(current.point)

            # up
            if self.isValid((x, y - 1)) and current.depth <= limit:
                position_list.append(Node((x, y - 1), path, current.cost + 1, depth=current.depth + 1))
                self.maze[y - 1][x] = 2
            # down
            if self.isValid((x, y + 1)) and current.depth <= limit:
                position_list.append(Node((x, y + 1), path, current.cost + 1, depth=current.depth + 1))
                self.maze[y + 1][x] = 2
            # left
            if self.isValid((x - 1, y)) and current.depth <= limit:
                position_list.append(Node((x - 1, y), path, current.cost + 1, depth=current.depth + 1))
                self.maze[y][x - 1] = 2
            # right
            if self.isValid((x + 1, y)) and current.depth <= limit:
                position_list.append(Node((x + 1, y), path, cost=current.cost + 1, depth=current.depth + 1))
                self.maze[y][x + 1] = 2

            self.maze[y][x] = 3
            explored += 1

        return current, explored, isFound

class IDS:
    # Iterative Deepening Search

    def __init__(self, maze, start, end):
        '''
        :param maze: a map
        :param start: (position1, position2)
        :param end: [(position1, position2), (position3, position4)]
        '''
        self.maze = maze
        self.start = start
        self.end = end

        self.h, self.w = self.maze.shape

    def isValid(self, point):
        return point[0] >= 0 and point[0] < self.h \
               and point[1] >= 0 and point[1] < self.w \
               and self.maze[point[1]][point[0]] == 0

    def depth_limited_search(self, limit):
        maze = copy.deepcopy(self.maze)
        position_list = [Node(self.start, [], 1)]
        explored = 0
        isFound = False
        end_search_list = []

        while len(position_list) > 0:
            current = position_list.pop()
            x, y = current.point

            if current.point == self.end:
                current.path.append(current.point)
                maze[y][x] = 3
                explored += 1
                isFound = True
                break

            path = current.path[:]
            path.append(current.point)

            # up
            if self.isValid((x, y - 1)) and current.depth <= limit:
                next_node = Node((x, y - 1), path, current.cost + 1, depth=current.depth + 1)
                position_list.append(next_node)
                maze[y - 1][x] = 2

                if current.depth+1>limit:
                    end_search_list.append(next_node)

            # down
            if self.isValid((x, y + 1)) and current.depth <= limit:
                next_node = Node((x, y + 1), path, current.cost + 1, depth=current.depth + 1)
                position_list.append(next_node)
                maze[y + 1][x] = 2

                if current.depth+1>limit:
                    end_search_list.append(next_node)

            # left
            if self.isValid((x - 1, y)) and current.depth <= limit:
                next_node = Node((x - 1, y), path, current.cost + 1, depth=current.depth + 1)
                position_list.append(next_node)
                maze[y][x - 1] = 2

                if current.depth+1>limit:
                    end_search_list.append(next_node)

            # right
            if self.isValid((x + 1, y)) and current.depth <= limit:
                next_node = Node((x + 1, y), path, cost=current.cost + 1, depth=current.depth + 1)
                position_list.append(next_node)
                maze[y][x + 1] = 2

                if current.depth+1>limit:
                    end_search_list.append(next_node)

            maze[y][x] = 3
            explored += 1

        return current, explored, isFound, end_search_list

    def start_search(self):
        limit = 1
        isFound = False
        while not isFound:
            current, explored, isFound, end_search_list = self.depth_limited_search(limit=limit)
            limit += 1
