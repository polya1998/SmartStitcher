import numpy as np
from queue import Queue

def bfs(matrix, start, end):
    queue = Queue()
    queue.put((start, []))
    visited = set()
    while not queue.empty():
        (x, y), path = queue.get()
        if (x, y) == end:
            return path
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and (nx, ny) not in visited:
                queue.put(((nx, ny), path + [(nx, ny)]))
                visited.add((nx, ny))
    return []

def connectIslands(matrix):
    # 找到所有的岛屿
    islands = []
    visited = set()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1 and (i, j) not in visited:
                island = [(i, j)]
                stack = [(i, j)]
                visited.add((i, j))
                while stack:
                    x, y = stack.pop()
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and matrix[nx][ny] == 1 and (nx, ny) not in visited:
                            stack.append((nx, ny))
                            visited.add((nx, ny))
                            island.append((nx, ny))
                islands.append(island)

    # 找到面积最大的岛屿
    max_island = max(islands, key=len)

    # 连接其他岛屿到最大岛屿
    for island in islands:
        if island is not max_island:
            start = island[0]
            end = max_island[0]
            path = bfs(matrix, start, end)
            for x, y in path:
                matrix[x][y] = 1

    return matrix

def main():
    matrix = np.array([
        [1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 1]
    ])
    result = connectIslands(matrix)
    print(result)

if __name__ == "__main__":
    main()
