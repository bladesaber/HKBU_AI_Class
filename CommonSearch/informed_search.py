from CommonSearch.node import Node

class A_Star:
    # A Star Search

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

    def get_heuristic_loss(self, point, goal):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def get_move_cost(self, current_node, next_node):
        return abs(current_node[0]-next_node[0]) + abs(current_node[1]-next_node[1])

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

            current_total_loss = current.cost+current.heuristic

            # up
            if self.isValid((x, y - 1)):
                move_cost = self.get_move_cost(current.point, (x, y-1))+current.cost
                heuristic_cost = self.get_heuristic_loss((x, y-1), self.end)
                move_cost = max(current_total_loss, move_cost+heuristic_cost)-heuristic_cost

                position_list.append(Node((x, y - 1), path,
                                          cost=move_cost,
                                          heuristic=heuristic_cost,
                                          depth=current.depth+1))
                self.maze[y - 1][x] = 2
            # down
            if self.isValid((x, y + 1)):
                move_cost = self.get_move_cost(current.point, (x, y + 1)) + current.cost
                heuristic_cost = self.get_heuristic_loss((x, y + 1), self.end)
                move_cost = max(current_total_loss, move_cost + heuristic_cost) - heuristic_cost

                position_list.append(Node((x, y + 1), path,
                                          cost=move_cost,
                                          heuristic=heuristic_cost,
                                          depth=current.depth+1))
                self.maze[y + 1][x] = 2
            # left
            if self.isValid((x - 1, y)):
                move_cost = self.get_move_cost(current.point, (x-1, y)) + current.cost
                heuristic_cost = self.get_heuristic_loss((x-1, y), self.end)
                move_cost = max(current_total_loss, move_cost + heuristic_cost) - heuristic_cost

                position_list.append(Node((x - 1, y), path,
                                          cost=move_cost,
                                          heuristic=heuristic_cost,
                                          depth=current.depth + 1))
                self.maze[y][x - 1] = 2
            # right
            if self.isValid((x + 1, y)):
                move_cost = self.get_move_cost(current.point, (x + 1, y)) + current.cost
                heuristic_cost = self.get_heuristic_loss((x + 1, y), self.end)
                move_cost = max(current_total_loss, move_cost + heuristic_cost) - heuristic_cost

                position_list.append(Node((x + 1, y), path,
                                          cost=move_cost,
                                          heuristic=heuristic_cost,
                                          depth=current.depth + 1))
                self.maze[y][x + 1] = 2

            position_list = sorted(position_list, key=lambda x: x.cost+x.heuristic)

            self.maze[y][x] = 3
            explored += 1

        return current, explored

class Greedy:
    # A Greedy Search

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

    def get_heuristic_loss(self, point, goal):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def get_move_cost(self, current_node, next_node):
        return abs(current_node[0]-next_node[0]) + abs(current_node[1]-next_node[1])

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
                position_list.append(Node((x, y - 1), path,
                                          cost=self.get_move_cost(current.point, (x, y-1))+current.cost,
                                          heuristic=self.get_heuristic_loss((x, y-1), self.end),
                                          depth=current.depth+1))
                self.maze[y - 1][x] = 2
            # down
            if self.isValid((x, y + 1)):
                position_list.append(Node((x, y + 1), path,
                                          cost=self.get_move_cost(current.point, (x, y+1))+current.cost,
                                          heuristic=self.get_heuristic_loss((x, y+1), self.end),
                                          depth=current.depth+1))
                self.maze[y + 1][x] = 2
            # left
            if self.isValid((x - 1, y)):
                position_list.append(Node((x - 1, y), path,
                                          cost=self.get_move_cost(current.point, (x-1, y))+current.cost,
                                          heuristic=self.get_heuristic_loss((x, y + 1), self.end),
                                          depth=current.depth + 1))
                self.maze[y][x - 1] = 2
            # right
            if self.isValid((x + 1, y)):
                position_list.append(Node((x + 1, y), path,
                                          cost=self.get_move_cost(current.point, (x+1, y))+current.cost,
                                          heuristic=self.get_heuristic_loss((x, y + 1), self.end),
                                          depth=current.depth + 1))
                self.maze[y][x + 1] = 2

            position_list = sorted(position_list, key=lambda x: x.heuristic)

            self.maze[y][x] = 3
            explored += 1

        return current, explored

class Beam:
    # Beam Search

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

    def get_heuristic_loss(self, point, goal):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def get_move_cost(self, current_node, next_node):
        return abs(current_node[0]-next_node[0]) + abs(current_node[1]-next_node[1])

    def start_search(self, length):
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

            current_total_loss = current.cost+current.heuristic

            # up
            if self.isValid((x, y - 1)):
                move_cost = self.get_move_cost(current.point, (x, y-1))+current.cost
                heuristic_cost = self.get_heuristic_loss((x, y-1), self.end)
                move_cost = max(current_total_loss, move_cost+heuristic_cost)-heuristic_cost

                position_list.append(Node((x, y - 1), path,
                                          cost=move_cost,
                                          heuristic=heuristic_cost,
                                          depth=current.depth+1))
                self.maze[y - 1][x] = 2
            # down
            if self.isValid((x, y + 1)):
                move_cost = self.get_move_cost(current.point, (x, y + 1)) + current.cost
                heuristic_cost = self.get_heuristic_loss((x, y + 1), self.end)
                move_cost = max(current_total_loss, move_cost + heuristic_cost) - heuristic_cost

                position_list.append(Node((x, y + 1), path,
                                          cost=move_cost,
                                          heuristic=heuristic_cost,
                                          depth=current.depth+1))
                self.maze[y + 1][x] = 2
            # left
            if self.isValid((x - 1, y)):
                move_cost = self.get_move_cost(current.point, (x-1, y)) + current.cost
                heuristic_cost = self.get_heuristic_loss((x-1, y), self.end)
                move_cost = max(current_total_loss, move_cost + heuristic_cost) - heuristic_cost

                position_list.append(Node((x - 1, y), path,
                                          cost=move_cost,
                                          heuristic=heuristic_cost,
                                          depth=current.depth + 1))
                self.maze[y][x - 1] = 2
            # right
            if self.isValid((x + 1, y)):
                move_cost = self.get_move_cost(current.point, (x + 1, y)) + current.cost
                heuristic_cost = self.get_heuristic_loss((x + 1, y), self.end)
                move_cost = max(current_total_loss, move_cost + heuristic_cost) - heuristic_cost

                position_list.append(Node((x + 1, y), path,
                                          cost=move_cost,
                                          heuristic=heuristic_cost,
                                          depth=current.depth + 1))
                self.maze[y][x + 1] = 2

            position_list = sorted(position_list, key=lambda x: x.cost+x.heuristic)[:length+1]

            self.maze[y][x] = 3
            explored += 1

        return current, explored
