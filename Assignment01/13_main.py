import ast
from collections import deque
import heapq

class graph:
    def __init__(self, array):
        self.n = len(array) * len(array)
        self.row = len(array)
        self.col = len(array)
        self.graph = self.build_graph(array)
        self.start, self.end = self.find_terminals(array)
        self.last_path = []
        # self.visit_freq = None
        self.visit_frequency = None
        
    def find_terminals(self, array):
        start = None
        end = None
        for i in range(self.row):
            for j in range(self.col):
                if array[i][j][0] == 0:
                    start = i*self.row + j
                if array[i][j][0] == 1:
                    end = i*self.row + j
                if start and end:
                    break    
        return start, end
        
    def build_graph(self, array):
        grid = [[] for _ in range(self.n * 2)]
        for i in range(self.row):
            for j in range(self.col):
                id = i * self.row + j 
                if array[i][j][1] >= 1:
                    grid[id].append(((i-1)*self.row + j, array[i][j][1]))
                if array[i][j][2] >= 1:
                    grid[id].append((i * self.row + j+1, array[i][j][2]))
                if array[i][j][3] >= 1:
                    grid[id].append(((i+1) * self.row + j, array[i][j][3]))
                if array[i][j][4] >= 1:
                    grid[id].append((i*self.row + j-1, array[i][j][4]))
        return grid
    
    def bfs(self):
        visited = [False] * self.n * 2
        parent = [-1] * self.n * 2
        queue = deque()
        queue.append((self.start, 0, 0)) # (node, path lenght, total cost)
        visited[self.start] = True
        self.visit_frequency = [[0 for _ in range(self.col * 2)] for _ in range(self.row * 2)]
        self.visit_frequency[self.start // self.row][self.start % self.col] += 1
        
        
        while queue:
            node, path_len, cost = queue.popleft()
            self.visit_frequency[node//self.row][node%self.col] += 1
            if node == self.end:
                path = []
                cur = node
                while cur != -1:
                    path.append((cur // self.row, cur % self.col))
                    cur = parent[cur]
                path.reverse
                self.last_path = path
                print("BFS Path:")
                print(path)
                print(f"Path length = {path_len}, Total cost = {cost}")
                return path_len, cost
            
            for neighbour, edge_cost in self.graph[node]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    parent[neighbour] = node
                    queue.append((neighbour, path_len+1, cost+edge_cost))
                    
        print("BFS: No path found!")
        return -1, -1
    
    def dfs_main(self):
        visited = [False] * self.n * 2
        parent = [-1] * self.n * 2
        result = []
        self.visit_frequency = [[0 for _ in range(self.col * 2)] for _ in range(self.row * 2)]
        self.visit_frequency[self.start // self.row][self.start % self.col] += 1
        
        def dfs_visit(node, lenght, cost):
            self.visit_frequency[node//self.row][node%self.col] += 1
            if node == self.end:
                result.append((lenght, cost))
                return True
            
            visited[node] = True
            for neighbor, edge_cost in self.graph[node]:
                if not visited[neighbor]:
                    parent[neighbor] = node
                    if dfs_visit(neighbor, lenght+1, cost + edge_cost):
                        return True
            return False
        
        if dfs_visit(self.start, 0, 0):
            path = []
            cur = self.end
            while cur != -1:
                path.append((cur // self.row, cur % self.col))
                cur = parent[cur]
            path.reverse()
            self.last_path = path
            print("DFS Path:")
            print(path)
            print(f"Path length = {result[0][0]}, Total cost = {result[0][1]}")
            return result[0] # (length, cost)
        else:
            print("DFS: No path found!")
            return -1 -1

    def dls(self, limit):
        visited = [False] * self.n * 2
        path = []
        found = [False]
        self.visit_frequency = [[0 for _ in range(self.col*2)] for _ in range(self.row*2)]
        self.visit_frequency[self.start // self.row][self.start % self.col] += 1
        
        def dfs(node, depth, cost):
            self.visit_frequency[node//self.row][node%self.col] += 1
            if depth > limit or found[0]:
                return
            visited[node] = True
            path.append((node // self.row, node % self.col))
            
            if node == self.end:
                self.last_path = path.copy()
                print("DLS Path:", path)
                print(f"Path Length: {len(path) - 1}")
                print("Total Cost:", cost)
                print(f"Depth limit: {limit}")
                found[0] = True
                return
            
            for neighbor, weight in self.graph[node]:
                if not visited[neighbor]:
                    if dfs(neighbor, depth+1, cost+weight):
                        return True 
            path.pop()
            visited[node] = False
        
        dfs(self.start, 0, 0)

        if not found[0]:
            print(f"No path found within DLS limit {limit}")

    def ucs(self):
        dist = [float('inf')] * self.n * 2
        parent = [-1] * self.n * 2
        dist[self.start] = 0
        self.visit_frequency = [[0 for _ in range(self.col*2)] for _ in range(self.row*2)]
        self.visit_frequency[self.start // self.row][self.start % self.col] += 1
        
        pq = [(0, self.start)] # (cost, node)
        
        while pq:
            cost, node = heapq.heappop(pq)
            self.visit_frequency[node//self.row][node%self.col] += 1
            if node == self.end:
                break
            if cost > dist[node]:
                continue
            
            for neighbor, weight in self.graph[node]:
                if dist[neighbor] > dist[node] + weight:
                    dist[neighbor] = dist[node] + weight
                    parent[neighbor] = node
                    heapq.heappush(pq, (dist[neighbor], neighbor))
        
        if dist[self.end] == float('inf'):
            print("UCS: No path found!!")
            return
        
        path = []
        current = self.end
        while current != -1:
            path.append((current // self.row, current % self.col))
            current = parent[current]
        path.reverse()  
        self.last_path = path
        print("UCS Path:")
        print(path)
        path_len = len(path) - 1
        print(f"Path length: {path_len}")
        print(f"Total cost: {dist[self.end]}")   

    def ids(self, max_depth = 1000):
        for limit in range(50, max_depth + 1, 50):
            visited = [False] * self.n * 2
            path = []
            found = [False]
            self.visit_frequency = [[0 for _ in range(self.col*2)] for _ in range(self.row*2)]
            self.visit_frequency[self.start // self.row][self.start % self.col] += 1
            
            def dfs(node, depth, cost):
                self.visit_frequency[node//self.row][node%self.col] += 1
                if depth > limit or found[0]:
                    return
                visited[node] = True
                path.append((node // self.row, node % self.col))
                
                if node == self.end:
                    self.last_path = path.copy()
                    print(f"IDS (Depth {limit}) path: {path}")
                    print(f"Path length: {len(path)-1}")
                    print(f"Total cost: {cost}")
                    found[0] = True
                    return
                
                for neighbor, weight in self.graph[node]:
                    if not visited[neighbor]:
                        dfs(neighbor, depth+1, cost+weight)
                        
                path.pop()
                visited[node] = False
            
            dfs(self.start, 0, 0)
            
            if found[0]:
                return
        print("No path found within max depth limit!")

    def best_first(self):
        def get_coordinate(node):
            return node//self.row, node %self.col
        
        def heuristic(node):
            x1, y1 = get_coordinate(node)
            x2, y2 = get_coordinate(self.end)
            return abs(x1-x2) + abs(y1-y2) # Manhattan distance
        
        visited = [False] * self.n * 2
        pq = [(heuristic(self.start), self.start, [self.start])] # (heuristic, node, path)
        self.visit_frequency = [[0 for _ in range(self.col*2)] for _ in range(self.row*2)]
        self.visit_frequency[self.start // self.row][self.start % self.col] += 1
        
        while pq:
            h, u, path = heapq.heappop(pq)
            self.visit_frequency[u//self.row][u%self.col] += 1
            if visited[u]:
                continue
            visited[u] = True
            
            if u == self.end:
                path_cor = [(cur // self.row, cur % self.col) for cur in path]
                self.last_path = path_cor
                print(f"Best-First Search Path: {path_cor}")
                print(f"Path Length: {len(path) - 1}")
                return
            
            for v, _ in self.graph[u]:
                if not visited[v]:
                    heapq.heappush(pq, (heuristic(v), v, path + [v]))
        
        print("No path found using Best-First Search!")

    def a_star(self):
        def get_coordinate(node):
            return node//self.row, node %self.col
        
        def heuristic(node):
            x1, y1 = get_coordinate(node)
            x2, y2 = get_coordinate(self.end)
            return abs(x1-x2) + abs(y1-y2) # Manhattan distance
        
        visited = [False] * self.n * 2
        open_set = [(heuristic(self.start), 0, self.start, [self.start])] 
        self.visit_frequency = [[0 for _ in range(self.col*2)] for _ in range(self.row*2)]
        self.visit_frequency[self.start // self.row][self.start % self.col] += 1
        
        while open_set:
            f, g, u, path = heapq.heappop(open_set)
            self.visit_frequency[u//self.row][u%self.col] += 1
            if visited[u]:
                continue
            visited[u] = True
            
            if u == self.end:
                path_cor = [(cur // self.row, cur % self.col) for cur in path]
                self.last_path = path_cor
                print(f"A* path: {path_cor}")
                print(f"Total cost: {g}")
                print(f"Path length: {len(path)-1}")
                return
            
            for v, cost in self.graph[u]:
                if not visited[v]:
                    new_g = g + cost
                    new_f = new_g + heuristic(v)
                    heapq.heappush(open_set, (new_f, new_g, v, path + [v]))
                    
        print("No path found using A* search!!")

    def weighted_a_star(self, w):
        def get_coordinate(node):
            return node//self.row, node %self.col
        
        def heuristic(node):
            x1, y1 = get_coordinate(node)
            x2, y2 = get_coordinate(self.end)
            return abs(x1-x2) + abs(y1-y2) # Manhattan distance
        
        visited = [False] * self.n * 2
        open_set = [(heuristic(self.start), 0, self.start, [self.start])] 
        self.visit_frequency = [[0 for _ in range(self.col*2)] for _ in range(self.row*2)]
        self.visit_frequency[self.start // self.row][self.start % self.col] += 1
        
        while open_set:
            f, g, u, path = heapq.heappop(open_set)
            self.visit_frequency[u//self.row][u%self.col] += 1
            if visited[u]:
                continue
            visited[u] = True
            
            if u == self.end:
                path_cor = [(cur // self.row, cur % self.col) for cur in path]
                self.last_path = path_cor
                print(f"A* path: {path_cor}")
                print(f"Total cost: {g}")
                print(f"Path length: {len(path)-1}")
                return
            
            for v, cost in self.graph[u]:
                if not visited[v]:
                    new_g = g + cost
                    new_f = new_g + w * heuristic(v)
                    heapq.heappush(open_set, (new_f, new_g, v, path + [v]))
                    
        print("No path found using weighted-A* search!!")


if __name__ == "__main__":
    with open("testcase1", "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().replace('\x00', '')
    array = ast.literal_eval(content)
    grid = graph(array)
    
    # print(grid.start)
    # print(grid.end)
    
    length, cost = grid.bfs()
    print()
    length, cost = grid.dfs_main()
    print()
    grid.dls(100)
    print()
    grid.ucs()
    print()
    grid.ids()
    print()
    grid.best_first()
    print()
    grid.a_star()
    
    
    