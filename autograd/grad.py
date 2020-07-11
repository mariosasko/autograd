from .node import Node
from .vector import Vector


def grad_sort(top_node):
    def topological_sort(node, visited):
        if node in visited:
            return []
        visited.append(node)

        if not node.parents:
            return [node]

        nodes = []
        for parent in node.parents:
            nodes += topological_sort(parent, visited)
        
        return nodes + [node]

    return reversed(topological_sort(top_node, []))


def grad(top_node):
    # if top_node.dim != 1:
    #     raise RuntimeError('grad can be created only for scalar outputs')

    # TODO: add warning when requiring grad of leaf nodes (leaf node is a top node)

    dct = {}
    dct[top_node] = Vector.ones(top_node.dim)

    for node in grad_sort(top_node):
        if node.is_leaf:
            continue

        for parent in node.parents:
            dct[parent] = (dct.get(parent, Vector.zeros(parent.dim)) 
                           + node.partial_derivative(dct[node], parent))
            
            if parent.is_leaf:
                parent.grad = dct[parent]