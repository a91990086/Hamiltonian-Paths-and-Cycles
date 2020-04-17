import numpy as np
import random

def is_hamilton(P, v, num_of_nodes):

    # If next node is the starting node and we have
    # traveled through every other node
    #if (P[0] == v and len(P) == num_of_nodes):
    if (len(P) == num_of_nodes):
        return 1

    else:
        return 0

def rotate(P, j, used_edge):

    # Assume P is the path [1 -> 2 -> 3 -> 4 -> 5] and j is 3 
    # i.e. we want to go back to 3 after 5

    # The solution is to go to 5 from 3 first 
    # and then reverse the path if an edge exists between 3 and 5

    # i.e. P becomes [1 -> 2 -> 3 -> 5 -> 4]
    # Note: If we enter this function, we already confirmed that the edge exists

    # New path after rotation
    rotated_P = []

    # Get the index of j, which is the node that is supposed to be the next head
    j_idx = P.index(j)
    
    # Mark j -> j+1 as unused
    #used_edge[j, P[j_idx + 1]] = 2
    
    # Mark j -> k(last node in original path) as used
    used_edge[j, P[-1]] = 1
    
    # The path from start until j should stay the same
    for i in range (j_idx + 1):

        rotated_P.append(P[i])

    # We then jump from j to end of original list P, then we go in reverse order
    for i in range (len(P) - 1, j_idx, -1):
        rotated_P.append(P[i])

        # Mark k-1 -> k as unused
        #used_edge[i - 1, i] = 2

        # Mark k -> k-1 as used
        try:

            used_edge[P[i + 1], P[i]] = 1
        
        except:

            continue

    return rotated_P

def reverse_path(P, used_edge):

    # Reverse the path list
    P.reverse()

    # Mark the corresponding edges as used
    for i in range(len(P) - 1):

        used_edge[P[i], P[i + 1]] = 1

    return P

def choose_next_head(v, adj, used_edge):

    # List of traversable and unused nodes adjacent to current head v
    reachable_and_unused = list(adj[v] * used_edge[v])

    # We want to get the node number 
    adj_reachable_node_to_v = [i for i, e in enumerate(reachable_and_unused) if e == 1]

    # Choose 1 node randomly from the aforementioned list to be the next head
    next_head = random.choice(adj_reachable_node_to_v)

    return next_head

def choose_next_head_from_used_or_unused(v, adj, used_edge, use_idx):

    # use_idx = 2 --> choose from previously not visited
    # use_idx = 1 --> choose from previously visited

    # List of traversable and unused nodes adjacent to current head v
    reachable = list(used_edge[v])

    # We want to get the node number 
    adj_reachable_node_to_v = [i for i, e in enumerate(reachable) if e == use_idx]

    # Choose 1 node randomly from the aforementioned list to be the next head
    next_head = random.choice(adj_reachable_node_to_v)

    return next_head

def choose_action(used_edge_len, num_of_nodes):

    # The idea is to generate a list for probability calculation
    # We will be using 1, 2, 3 to denote choosing action 1, 2, 3 respectively
    # e.g. We will be choosing uniformly from the prob_list = [1 2 3 3 3 3], 
    # which means that
    # we will choose action 1 with prob 1/6
    # we will choose action 2 with prob 1/6
    # we will choose action 3 with prob 4/6

    # (1/n) prob of reversing the list
    prob_list = [1]

    # (used_edge_len/n) prob of choosing an used edge
    for i in range(used_edge_len):
        prob_list.append(2)

    # (n - 1 - used_edge_len)/n prob of choosing an unused edge
    for i in range(num_of_nodes - 1 - used_edge_len):
        prob_list.append(3)

    # Now choose from the generated list uniformly
    action = random.choice(prob_list)

    return action

def ham(adj, start):

    # How many nodes
    num_of_nodes = adj.shape[0]

    # Duplicate adjacency matrix to mark the edges we have used
    # unused edges are marked 2, used edges are marked 1
    # Initially the matrix will be the exact same as adj matrix since
    # we haven't visited any of the nodes yet
    used_edge = adj*2

    # Path Change: Add an edge between start and end (if it doesn't exist already)
    # The idea is the keep this edge and find a Hamiltonian cycle
    # After such cycle is found we can simply remove the edge 
    # and obtain a Hamiltonian path
    #used_edge[start, end] = 2
    #used_edge[end, start] = 2

    # The path we have so far
    path = [start]

    # Choose the next head from unused
    # (use_idx = 1: choose from use, 2: choose from unused)
    # Path Change: 
    head = choose_next_head_from_used_or_unused(start, adj, used_edge, use_idx = 2)

    # Path Change: Set end as head
    #head = end
    
    # Mark the edge as used and update path
    used_edge[start, head] = 1
    path.append(head)

    while(2 in list(used_edge[head])):
        a = len(path)
        # Choose which action
        # 1: Reverse the path
        # 2: Choose uniformly from used edge and rotate
        # 3: Choose unifromly from unused edge, rotate if neccessary
        action = choose_action(list(used_edge[head]).count(1), num_of_nodes)

        # The most likely to happen action = 3
        if(action == 3):

            # Choose the next head from unused
            next_head = choose_next_head_from_used_or_unused(head, adj, used_edge, use_idx = 2)

        # The second most likely to happen action = 2
        elif(action == 2):

            # Choose the next head from used
            next_head = choose_next_head_from_used_or_unused(head, adj, used_edge, use_idx = 1)

        else:

            # Reverse the path
            reverse_path(path, used_edge)

            # Mark head
            head = path[-1]

            continue

        # If next head completes a hamiltonian cycle
        if (is_hamilton(path, next_head, num_of_nodes)):
            break
        
        # If next head is already in the path (already visited)
        # and next head is not adjacent to head in our path
        elif (next_head in path):
            if abs(path.index(head) - path.index(next_head)) != 1:

                # Path Change: Check rotation condition as we don't want to 
                # break the edge between start and end
                #idx = path.index(next_head)

                #if (not(next_head == start and path[idx + 1] == end) and \
                #    not(next_head == end and path[idx + 1] == start)):

                path = rotate(path, next_head, used_edge)

        # If next head is not in the path
        else:
            path.append(next_head)
            used_edge[head, next_head] = 1
            
        # Set new head
        head = path[-1]

    # If we end the for loop naturally
    # i.e. we cant find hamiltonian cycle
    else:
        return 0

    return path

def ham_repetative(adj, start, iter = 1):
    
    cycles = []
    
    # Run ham for iter number of times
    for i in range(iter):

        # Find  valid Hamiltonian cycle
        while(1):

            path = ham(adj, start)

            if path:

                break

        cycles.append(path)

    return cycles

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

    # Number of vertices
    n = edge_list[-1][0]

    # Create empty adj matrix
    adj = np.zeros((n, n))

    # Populate the corresponding entries
    for e in edge_list:
        adj[e[0] - 1, e[1] - 1] = 1

    return adj

def main():

    adj = np.array([[0, 1, 0, 0, 1],
                    [1, 0, 1, 1, 1],
                    [0, 1, 0, 1, 1],
                    [0, 1, 1, 0, 1],
                    [1, 1, 1, 1, 0]])
    
    #adj = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
    #                [1, 0, 1, 0, 0, 0, 0, 0],
    #                [0, 1, 0, 0, 1, 0, 1, 1],
    #                [1, 0, 0, 0, 0, 0, 1, 1],
    #                [1, 0, 1, 0, 0, 0, 1, 0],
    #                [0, 0, 0, 0, 0, 0, 1, 1],
    #                [1, 0, 1, 1, 1, 1, 0, 0],
    #                [0, 0, 1, 1, 0, 1, 0, 0]])
    
    # HAM heuristic does not work well with large graphs

    #edge_list = np.genfromtxt(r'tsphcp/GP3_126.hcp', dtype = int, skip_header = 6, skip_footer = 2)
    #edge_list = list(map(tuple, edge_list))
    #adj = edgelist_to_adj(edge_list)

    path = ham_repetative(adj, 0, iter = 100)
    path = make_equivalent_path(path)
    path = find_unique_path(path)

    return 0

if __name__ == '__main__':
    main()
