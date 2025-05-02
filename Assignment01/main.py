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
        
        while queue:
            node, path_len, cost = queue.popleft()
            if node == self.end:
                path = []
                cur = node
                while cur != -1:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse
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
    
    def dfs(self):
        visited = [False] * self.n * 2
        parent = [-1] * self.n * 2
        result = []
        
        def dfs_visit(node, lenght, cost):
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
                path.append(cur)
                cur = parent[cur]
            path.reverse()
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
        
        def dfs(node, depth, cost):
            if depth > limit or found[0]:
                return
            visited[node] = True
            path.append(node)
            
            if node == self.end:
                print("DLS Path:", path)
                print("Path Length:", len(path) - 1)
                print("Total Cost:", cost)
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
        
        pq = [(0, self.start)] # (cost, node)
        
        while pq:
            cost, node = heapq.heappop(pq)
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
            path.append(current)
            current = parent[current]
        path.reverse()  
        
        print("UCS Path:")
        print(path)
        print(f"Path length: {len(path) - 1}")
        print(f"Total cost: {dist[self.end]}")   

    def ids(self, max_depth = 10000000):
        for limit in range(max_depth + 1):
            visited = [False] * self.n * 2
            path = []
            found = [False]
            
            def dfs(node, depth, cost):
                if depth > limit or found[0]:
                    return
                visited[node] = True
                path.append(node)
                
                if node == self.end:
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

if __name__ == "__main__":
    with open("tc1.in", "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().replace('\x00', '')
    array = ast.literal_eval(content)
    grid = graph(array)
    
    # print(grid.graph)
    # print(grid.start)
    # print(grid.end)
    
    # len, cost = grid.bfs()
    # len, cost = grid.dfs()
    # grid.dls(40)
    # grid.ucs()
    # grid.ids()
    
    