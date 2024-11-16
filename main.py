import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = defaultdict(lambda: defaultdict(int))
        self.original_graph = defaultdict(lambda: defaultdict(int))

    # Add an edge to the graph
    def add_edge(self, source, destination, capacity):
        self.graph[source][destination] = capacity
        self.original_graph[source][destination] = capacity

    # Depth-first search to find a path from source to sink
    def _dfs(self, start, end, visited, path):
        visited[start] = True

        # If we reached the end node, we found a path
        if start == end:
            return True
        
        # For each neighbor of the current node, check if it has capacity
        # and we haven't visited it yet, then add it to the path
        for neighbor, capacity in self.graph[start].items():
            if not visited[neighbor] and capacity > 0:
                path[neighbor] = start
                if self._dfs(neighbor, end, visited, path):
                    return True
                
        return False

    # Ford-Fulkerson algorithm to find the maximum flow from source to sink
    def ford_fulkerson(self, start, end):
        max_flow = 0
        path = [-1] * self.vertices

        while self._dfs(start, end, [False] * self.vertices, path):
            old_end = end
            path_flow = float('Inf')

            # Find the minimum capacity while the end node is not the start node
            while old_end != start:
                capacity = self.graph[path[old_end]][old_end]
                path_flow = min(path_flow, capacity)
                old_end = path[old_end]

            # Update the reverse edges capacities
            current_node = end
            while current_node != start:
                prev_node = path[current_node]
                self.graph[prev_node][current_node] -= path_flow
                self.graph[current_node][prev_node] += path_flow
                current_node = path[current_node]

            # Add the path flow to the total flow
            max_flow += path_flow

        return max_flow

    '''
    Visualize the flow through the graph
    :param flow_data: A dictionary containing the flow data for each edge
    '''
    def visualize_flow(self, flow_data):
        directed_graph = nx.DiGraph()

        for vertice_index in range(self.vertices):
            for vertice, capacity in self.original_graph[vertice_index].items():
                if capacity > 0:
                    directed_graph.add_edge(vertice_index, vertice, capacity=capacity)

        positions = nx.spring_layout(directed_graph)

        # Draw the graph with edge labels for capacities
        plt.figure(figsize=(10, 8))
        labels = nx.get_edge_attributes(directed_graph, 'capacity')

        # Draw each edge with its capacity
        nx.draw(
          directed_graph, 
          positions, 
          with_labels=True, 
          node_size=700, 
          node_color='green', 
          font_weight='bold', 
          font_size=12
        )

        # Draw edge labels
        nx.draw_networkx_edge_labels(directed_graph, positions, edge_labels=labels)

        # Display flow on edges (using flow_data)
        for (source, _), flow in flow_data.items():
          plt.text(
            positions[source][0], 
            positions[source][1] - 0.05,
            f'Flow: {flow}', 
            color='red', 
            fontsize=10
          )

        plt.title("Network Flow")
        plt.show()

if __name__ == "__main__":
    graph = Graph(6)
    edges = [
        (0, 1, 11),
        (0, 2, 12),
        (1, 3, 12),
        (2, 1, 1),
        (2, 4, 11),
        (3, 5, 19),
        (4, 3, 7),
        (4, 5, 4)
    ]

    # Add the edges to the graph
    for source, destination, capacity in edges:
        graph.add_edge(source, destination, capacity)

    start = 0
    end = 5

    # Find the maximum flow from source to sink
    max_flow = graph.ford_fulkerson(start, end)
    print(f"The maximum possible flow is {max_flow}")

    # Now, visualize the flow through the graph
    flow_data = {}
    for source in range(graph.vertices):
        for target in graph.graph[source].keys():
            flow = graph.graph[target][source]
            if flow > 0:
                flow_data[(target, source)] = flow

    graph.visualize_flow(flow_data)
