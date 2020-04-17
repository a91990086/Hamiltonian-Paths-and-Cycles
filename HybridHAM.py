import numpy as np
import random

def rotate(P, j, used_edge):

    # Assume P is the path [1 -> 2 -> 3 -> 4 -> 5] and j is 3 
    # i.e. we want to go back to 3 after 5

    # The solution is to go to 5 from 3 first 
    # and then reverse the path if an edge exists between 3 and 5

    # i.e. P becomes [1 -> 2 -> 3 -> 5 -> 4]
    # Note: If we enter this function, we already confirmed that
    # the edge (i.e. edge between 3 & 5) exists

    # New path after rotation
    rotated_P = []

    # Get the index of j, which is the node that is supposed to be the next head
    j_idx = P.index(j)
    
    # Mark j -> j+1 as unused
    used_edge[j, P[j_idx + 1]] = 2
    used_edge[P[j_idx + 1], j] = 2
    
    # Mark j -> k(last node in original path) as used
    used_edge[j, P[-1]] = 1
    used_edge[P[-1], j] = 1
    
    # The path from start until j should stay the same
    for i in range (j_idx + 1):

        rotated_P.append(P[i])

    # We then jump from j to end of original list P, then we go in reverse order
    for i in range (len(P) - 1, j_idx, -1):
        rotated_P.append(P[i])

    return rotated_P

def create_vertices_order_list(adj):

    # List of vertices degree
    Va = {}

    for i in range(adj.shape[0]):

        # Check vertex order
        order = list(adj[i]).count(1)
        Va[i] = order
    
    # Sort the list according to vertex orders, return the indices
    Va = sorted(Va.items(), key=lambda x: x[1])

    return Va

def check_unreachable(path, next_head, used_edge, adj):

    # Initially we set the reachable flag to true
    reachable = 1

    # All nodes adj to next head
    adj_node_to_next_head = [i for i, e in enumerate(adj[next_head]) if e == 1]

    count = 0
    
    # Iterate through each node
    for node in adj_node_to_next_head:
        
        if (node in path):

            count = count + 1
            continue

        # How many unvisted neighbours does such node have
        num_unvisited_adj = list(used_edge[node]).count(2)

        # If travelling from head to next head makes one of next head's neightbor
        # unreachable
        if (num_unvisited_adj == 1):

            reachable = 0
            break

    # If all adj nodes to next head are in path
    if (count == len(adj_node_to_next_head)):

        reachable = 0

    return reachable

def greedy_depth_first_search(Va, adj, used_edge, num_of_nodes, P):

    # Head is the end of the path
    head = P[-1]

    # Get all nodes adjacent to head
    adj_node_to_head = [i for i, e in enumerate(adj[head]) if e == 1]

    vertices_order, _ = zip(*Va)

    # Get the rank of thier degree order
    degree_order_rank = [vertices_order.index(i) for i in adj_node_to_head]

    while (1):

        # If head is dead end
        if (len(adj_node_to_head) == 0):

            break

        # Set the adj node with least degree the next head
        next_head = adj_node_to_head.pop(np.argmin(degree_order_rank))

        degree_order_rank.pop(np.argmin(degree_order_rank))

        # If we already have a hamiltonian path and next head completes the cycle
        # or the path is 1 element short from completing a hamiltonian path
        if (((next_head not in P) and (len(P) == num_of_nodes - 1)) or \
            ((next_head == P[0]) and (len(P) == num_of_nodes))):

            # Append the path
            P.append(next_head)

            # Mark the edge as used
            used_edge[head, next_head] = 1
            used_edge[next_head, head] = 1

            break

        # Else if we are nowhere near completing hamiltonian path/cycle but
        # next head is in path
        elif (next_head in P): 

            continue

        # If going to next head still makes all its neighbor reachable
        if (check_unreachable(P, next_head, used_edge, adj)):
      
            # Add to path
            P.append(next_head)

            # Mark the edge as used
            used_edge[head, next_head] = 1
            used_edge[next_head, head] = 1

            # Set next head as head
            head = next_head

            # Get all unvisited nodes adjacent to head
            adj_node_to_head = [i for i, e in enumerate(used_edge[head]) if e == 2]

            # Get the rank of thier degree order
            degree_order_rank = [vertices_order.index(i) for i in adj_node_to_head]

    return P

def convert_to_hamiltonian(Va, path, used_edge, adj, num_of_nodes, phase):

    # phase = 2 --> select highest degree end for rotation
    # phase = 3 --> select smallest degree end for rotation

    vertices_order, _ = zip(*Va)

    while (1):

        # If we are at phase 2 and we get a hamiltonian path
        if (phase == 2 and len(path) == num_of_nodes):

            break

        # If we are at phase 3 and we get a hamiltonian cycle
        elif (phase == 3 and len(path) == num_of_nodes + 1 and path[0] == path[-1]):

            break

        # Degree order for the first node in path
        start_degree = vertices_order.index(path[0])
    
        # Degree order for the last node in path
        end_degree = vertices_order.index(path[-1])

        # We want the end node to be the highest degree of the two 
        # if we are at phase 2
        if (phase == 2 and start_degree > end_degree):

            path.reverse()

        # We want the end node to be the highest degree of the two
        # if we are at phase 2
        elif (phase == 3 and start_degree < end_degree):

            path.reverse()

        # All of end node's neighbors
        all_adj_to_end = [i for i, e in enumerate(adj[path[-1]]) if e == 1]

        # Which of these neighbors are in the path
        visited_adj_to_end = [i for i in all_adj_to_end if i in path]

        # If such neighbors exist
        if (visited_adj_to_end):

            # Choose from the neighbors at random
            j = random.choice(visited_adj_to_end)

            # Perform rotation
            path = rotate(path, j, used_edge)

        else:

            return 0

        # Continue to extend the path with greedy depth first search
        path = greedy_depth_first_search(Va, adj, used_edge, num_of_nodes, path)

    return path

def construct_used_edge_matrix(adj, path):

    # Construct the used edge matrix
    used_edge = adj * 2
        
    for i in range(len(path) - 1):

        # Mark the corresponding used edges
        used_edge[path[i], path[i + 1]] = 1
        used_edge[path[i + 1], path[i]] = 1

    return used_edge

def hybrid_ham(adj):

    path = []

    # How many nodes (vertices)
    num_of_nodes = adj.shape[0]

    # Create a list of vertex orders
    Va = create_vertices_order_list(adj)

    nodes_with_highest_order = []
    highest_order = Va[-1][1]

    # Calculate how many nodes have the highest degree vertex
    for v in reversed(Va):

        if (v[1] == highest_order):

            nodes_with_highest_order.append(v[0])

        else:
            
            break

    # Repeat for each of the vertices with highest degree
    for i, node in enumerate(nodes_with_highest_order):
    
        initial_path = [node]

        path.append([])

        # A path induced by such vertex
        temp_path = greedy_depth_first_search(Va, adj, adj * 2, num_of_nodes, initial_path)

        # Append to our list
        path[i].append(temp_path)

        # If |P| = n, go to phase three
        if (len(temp_path) == num_of_nodes):

            # Construct used edge
            used_edge = construct_used_edge_matrix(adj, temp_path)

            # After phase 3, we should get a hamiltonian cycle
            path = convert_to_hamiltonian(Va, temp_path, used_edge, adj, num_of_nodes, phase = 3)

            break

    # If none of the paths induced by the highest degree vertices
    # makes a hamiltonian path

    ######################### NOTE: SO FAR ALL TESTED GRAPHS DID NOT ENTER PHASE 2 ####################
    else:

        length_of_path = []

        for p in path:

            length_of_path.append(len(p))

        # Indices of the longest paths
        longest_path_idx = [i for i, j in enumerate(length_of_path) if j == max(length_of_path)]

        # If only 1 longest path
        if (len(longest_path_idx) == 1):

            path_to_be_extended = path[longest_path_idx]

        else:

            # If multiple paths with same length, choose 1 randomly
            path_to_be_extended = random.choice(path)
            path_to_be_extended = path_to_be_extended[-1]

        # Construct the used edge matrix
        used_edge = construct_used_edge_matrix(adj, temp_path)

        # Go to phase 2
        path = convert_to_hamiltonian(Va, path_to_be_extended, used_edge, adj, num_of_nodes, phase = 2)

        # We should have a hamiltonian path after phase 2, chechk to 
        # see if there is an edge connecting the first and last vertices for such path
        if (adj[path[0], path[-1]] == 1):

            # Complete the hamiltonian cycle
            path.append(path[0])

        else:

            # Construct the used edge matrix after phase 2
            used_edge = construct_used_edge_matrix(adj, path)

            # Go to phase 3
            path = convert_to_hamiltonian(Va, path, used_edge, adj, num_of_nodes, phase = 3)

    return path

def make_equivalent_path(cycles, node_num = 0):

    # We want to move the current node to the end of the path list
    # until 0 becomes the first element. Note that since we have
    # a Hamiltonian Cycle, shifting the order doesn't matter
    # e.g. [1 -> 2 -> 0 -> 1] is equavalent to [0 -> 1 -> 2 -> 0]

    for i, cy in enumerate(cycles):

        # While the first node in the cycle is not node_num
        while (cycles[i][0] != node_num):

            # Pop the first element in the cycle and append it to the end
            cycles[i].append(cycles[i].pop(0))

    return cycles

def find_unique_path(cycles):

    unique_path = []
    unique_path.append(cycles[0])

    for i, cy in enumerate(cycles):

        if cy not in unique_path:

            unique_path.append(cy)

    return unique_path

def edgelist_to_adj(edge_list):

    n = max(max(edge_list))

    # Create empty adj matrix
    adj = np.zeros((n, n))

    # Populate the corresponding entries
    for e in edge_list:
        adj[e[0] - 1, e[1] - 1] = 1

    return adj

def create_node_between_vertices(start, end, adj):

    # The idea is to add a node between the start and end vertices to our graph
    # If we can find a Hamiltonian cycle within the modified graph, we can simply
    # delete the node between start and end after obtaining the Hamiltonian cycle
    # and get a Hamiltonian path

    # Concatenate column
    col = np.zeros((adj.shape[0], 1))
    col[start, :] = 1
    col[end, :] = 1
    adj = np.concatenate((adj, col), axis = 1)
    
    # Concatenate row
    row = np.zeros((1, adj.shape[1]))
    row[:, start] = 1
    row[:, end] = 1
    adj = np.concatenate((adj, row), axis = 0)

    return adj

def hybrid_ham_repetative(adj, start, end, iter = 10):
    
    cycles = []
    
    # Run ham for iter number of times
    for i in range(iter):

        # Find  valid Hamiltonian cycle
        while(1):

            path = hybrid_ham(adj)

            if path:

                break
        
        # Make cycle Hamiltonian path
        path = ham_path_post_processing(path, start, end)

        cycles.append(path)

    return cycles

def ham_path_post_processing(path, start, end):

    # Remove the last node, which is the first node since this path
    # is a Hamiltonian cycle
    path.pop(-1)

    # Check path is indeed a Hamiltonian cycle
    assert len(path) == len(set(path))

    # Pop the added node
    start_idx = path.index(start)
    end_idx = path.index(end)

    if (start_idx > end_idx):
        pop_idx = end_idx + 1
    else:
        pop_idx = start_idx + 1

    path.pop(pop_idx)

    # Shuffle the path so that start becomes the first element
    # While the first node in the cycle is not node_num
    while (path[0] != start and path[-1] != end):

        # Pop the first element in the cycle and append it to the end
        path.append(path.pop(0))

    return path

def find_unique_path(cycles):

    unique_path = []
    unique_path.append(cycles[0])

    for i, cy in enumerate(cycles):

        if cy not in unique_path:

            unique_path.append(cy)

    return unique_path

def main():

    # Some simple test cases

    #adj = np.array([[0, 1, 0, 0, 1, 1],
    #                [1, 0, 1, 1, 1, 0],
    #                [0, 1, 0, 1, 1, 1],
    #                [0, 1, 1, 0, 1, 0],
    #                [1, 1, 1, 1, 0, 0],
    #                [1, 0, 1, 0, 0, 0]])
    
    #adj = np.array([[0, 1, 1, 1, 1, 1, 1, 1],
    #                [1, 0, 1, 0, 0, 0, 0, 0],
    #                [0, 1, 0, 0, 1, 0, 1, 1],
    #                [1, 0, 0, 0, 0, 0, 1, 1],
    #                [1, 0, 1, 0, 0, 0, 1, 0],
    #                [0, 0, 0, 0, 0, 0, 1, 1],
    #                [1, 0, 1, 1, 1, 1, 0, 0],
    #                [0, 0, 1, 1, 0, 1, 0, 0]])
    
    # Test with large data sets

    # Read edge list from file
    edge_list = np.genfromtxt(r'tsphcp/GP3_126.hcp', dtype = int, skip_header = 6, skip_footer = 2)
    edge_list = list(map(tuple, edge_list))

    # Create adjacency matrix from edge list
    adj = edgelist_to_adj(edge_list)

    # The idea is to add a node between the start and end vertices to our graph
    # If we can find a Hamiltonian cycle within the modified graph, we can simply
    # delete the node between start and end after obtaining the Hamiltonian cycle
    # and get a Hamiltonian path
    start = 0
    end = 1

    # Now create the node between start and end
    adj = create_node_between_vertices(start, end, adj)

    # Find the hamiltonian cycle
    path = hybrid_ham_repetative(adj, start, end, iter = 5)

    # Now delete the repetative paths
    path = find_unique_path(path)

    print("We found ", len(path), "Hamiltonian path(s) from node ", start, "to node ", end)
    for p in path:

        print(p)
        print("\n")

    return 0

if __name__ == '__main__':
    main()
