class Node:
    def __init__(self, point, path, cost, heuristic=100, depth=0):
        self.point = point
        self.path = path
        self.cost = cost
        self.heuristic = heuristic
        self.depth = depth