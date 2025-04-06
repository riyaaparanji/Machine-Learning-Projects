#include "pathfinder.h"
#include <iostream>

int main() {
    std::vector<std::vector<int>> graph = {
        {0, 0, 0, 0, 0},
        {1, 1, 0, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 0, 0, 0}
    };

    Node start(0, 0);
    Node goal(4, 4);

    std::vector<Node> path = FindPath(graph, start, goal);

    if (!path.empty()) {
        std::cout << "Path found:\n";
        PrintPath(path);
    } else {
        std::cout << "No path found.\n";
    }

    return 0;
}