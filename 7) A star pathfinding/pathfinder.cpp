#include "pathfinder.h"
#include <algorithm>
#include <iostream>

Node::Node(int _x, int _y)
    : x(_x), y(_y), f(0), g(0), h(0) {}

bool Node::operator>(const Node& other) const {
    return f > other.f;
}

bool Node::operator==(const Node& other) const {
    return x == other.x && y == other.y;
}

std::vector<Node> FindPath(const std::vector<std::vector<int>>& graph, const Node& start, const Node& goal) {
    const int directionX[] = {-1, 0, 1, 0};
    const int directionY[] = {0, 1, 0, -1};

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> openList;
    std::vector<std::vector<bool>> closedList(graph.size(), std::vector<bool>(graph[0].size(), false));
    std::vector<std::vector<Node>> parentMap(graph.size(), std::vector<Node>(graph[0].size(), Node(-1, -1)));

    openList.push(start);

    while (!openList.empty()) {
        Node current = openList.top();
        openList.pop();

        if (current == goal) {
            std::vector<Node> path;
            while (!(current == start)) {
                path.push_back(current);
                current = parentMap[current.x][current.y];
            }
            path.push_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }

        closedList[current.x][current.y] = true;

        for (int i = 0; i < 4; ++i) {
            int newX = current.x + directionX[i];
            int newY = current.y + directionY[i];

            if (newX >= 0 && newX < graph.size() && newY >= 0 && newY < graph[0].size()) {
                if (graph[newX][newY] == 0 && !closedList[newX][newY]) {
                    Node neighbor(newX, newY);
                    int newG = current.g + 1;

                    neighbor.g = newG;
                    neighbor.h = abs(newX - goal.x) + abs(newY - goal.y);
                    neighbor.f = neighbor.g + neighbor.h;

                    // Only add to open list if not already closed or better path found
                    if (parentMap[newX][newY].x == -1 || newG < parentMap[newX][newY].g) {
                        parentMap[newX][newY] = current;
                        openList.push(neighbor);
                    }
                }
            }
        }
    }

    return std::vector<Node>(); // No path found
}

void PrintPath(const std::vector<Node>& path) {
    for (const Node& node : path) {
        std::cout << "(" << node.x << ", " << node.y << ") ";
    }
    std::cout << std::endl;
}
