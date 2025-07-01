import random
from collections import deque

def generate_grid(n, percentage):
    dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
    opp = [2, 3, 0, 1]

    def random_cost(): return random.randint(1, 10)

    def in_bounds(i, j): return 0 <= i < n and 0 <= j < n
    
    si, sj, ei, ej = 1, 1, n-2, n-2

    # BFS to build a guaranteed path
    visited = [[False]*n for _ in range(n)]
    parent = dict()
    q = deque()
    q.append((si, sj))
    visited[si][sj] = True
    found = False

    while q and not found:
        i, j = q.popleft()
        random.shuffle(dirs)
        for d, (di, dj) in enumerate(dirs):
            ni, nj = i + di, j + dj
            if in_bounds(ni, nj) and not visited[ni][nj]:
                visited[ni][nj] = True
                parent[(ni, nj)] = (i, j)
                q.append((ni, nj))
                if (ni, nj) == (ei, ej):
                    found = True
                    break

    if not found:
        print("path not found error")
        return generate_grid(n)  # Retry if no path found
    
    path = [(si, sj)]
    cur = (si, sj)
    # while cur != (si, sj):
    #     cur = parent[cur]
    #     path.append(cur)
    # path.reverse()
    # path_set = set(path)
    
    while cur != (n-1, sj):
        ex, ey = cur
        cur = (ex+1, ey)
        path.append(cur)
    while cur != (n-1, ej):
        ex, ey = cur
        cur = (ex, ey+1)
        path.append(cur)
    while cur != (ei, ej):
        ex, ey = cur
        cur = (ex-1, ey)
        path.append(cur)
    
    # print(path)
    path_set = set(path)

    # # Step 1: Fill everything as type-2 with full costs
    grid = [[[2] + [-10 for _ in range(4)] for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i > 0:
                if j > 0:
                    grid[i][j] = [2, grid[i-1][j][3], random_cost(), random_cost(), grid[i][j-1][2]]
                else:
                    grid[i][j] = [2, grid[i-1][j][3], random_cost(), random_cost(), -1]
            else:
                if j > 0:
                    grid[i][j] = [2, -1, random_cost(), random_cost(), grid[i][j-1][2]]
                else:
                    # print("ok")
                    grid[i][j] = [2, -1, random_cost(), random_cost(), -1]

    # Step 3: Add obstacles (type-3) not in path
    for i in range(n):
        for j in range(n):
            if (i, j) not in path_set and random.random() < percentage/100:
                grid[i][j] = [3, -1, -1, -1, -1]
                if i > 0:
                    grid[i-1][j][3] = -1
                if i < n-1:
                    grid[i+1][j][1] = -1
                if j > 0:
                    grid[i][j-1][2] = -1
                if j < n-1:
                    grid[i][j+1][4] = -1
    
    # Set start and end
    grid[si][sj][0] = 0
    grid[ei][ej][0] = 1

    return grid

# Example usage
n = 20
grid = generate_grid(n, 40)
print(grid)
