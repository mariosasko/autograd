from .node import Node


def grad_sort(top_node):
    parent_nodes = lambda parents: [node for node in parents if isinstance(node, Node)]

    def topological_sort(node, visited):
        if node in visited:
            return []
        visited.append(node)

        parents = parent_nodes(node)
        if not parents:
            return [node]
        
        nodes = []
        for parent in parents:
            nodes += topological_sort(parent, visited)
        
        return nodes

    return reversed(topological_sort(top_node, []))


def grad(top_node):
    pass