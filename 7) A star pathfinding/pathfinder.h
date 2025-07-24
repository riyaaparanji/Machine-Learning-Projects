#pragma once

#include <vector>
#include <queue>
#include <cmath>

// Define a structure to represent a node in the graph
struct Node
{
    int x, y;     // Coordinates of the node
    int f, g, h;  // Cost values for A*

    Node(int r = 0, int c = 0);

    // Overload comparison operators for priority queue
    bool operator>(const Node& other) const;
    bool operator==(const Node& other) const;
};

// A* pathfinding algorithm
std::vector<Node> FindPath(const std::vector<std::vector<int>>& graph, const Node& start, const Node& goal);

// Function to print the path
void PrintPath(const std::vector<Node>& path);
