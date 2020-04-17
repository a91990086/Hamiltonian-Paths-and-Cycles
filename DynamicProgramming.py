import numpy as np
import scipy as sp

# This is a function to create the power set lookup table
def create_lookup_table(adj):
    
    # Get value of n from adj, which exists in R^nxn
    n = adj.shape[0]

    # Generate the loop up table, populate all elements with -1
    lookup_table = -1*np.ones((n, 2**n))

    # Populate the initial elements with corresponding starting locations
    for i in range(n):
        lookup_table[i][2**i] = i

    return lookup_table

# This is a function to traverse the power set lookup table and populate the
# corresponding entries in the table
def traverse(adj):
    
    # Generate a lookup table for the powerset of 2^n
    lookup_table = create_lookup_table(adj)

    # Length of n
    n = lookup_table.shape[0]

    # How many subsets in our powerset
    set = lookup_table.shape[1]

    # Each i represents a set    
    for i in range(set):
        
        # Each j a vertex
        for j in range(n):

            # If vertex j exists in set i
            if (1<<j & i):

                # Each k is also a vertex
                for k in range (n):

                    # If k which is not equal to j is also in the set i and there exists a path between j and k
                    # If j excluded from set i, and there exists a path which traverses every vertices
                        # in set {i excluding j) and end up at k, then there exists a path which traverses 
                        # every vertices in set i and end at k
                    if (adj[j][k] and 1<<k & i and k != j and lookup_table[k][(2**j)^i] != -1):

                        # Assign k to lookup table
                        lookup_table[j][i] = k
                        break


    return lookup_table

# This is a function to trace our path from the lookup table
def path_traceback(lookup_table):

    path = []
    n = lookup_table.shape[0]

    for i in range (n):
        
        # Look at the last power set
        prev_node = lookup_table[i][-1]
        path.append([])

        # If we end up at node i after the last power set
        if (prev_node != -1):

            # Append to path
            path[i].append(i)
            path[i].append(prev_node)

            # The power set index when prev_node is excluded
            # i.e. If the last power set is {1, 2, 3, 4} and 
            # the node we end up at is 1, then we look for the index
            # of the power set {2, 3, 4} (i.e. power set excluding node i)
            idx = (2**i) ^ (2**n - 1)

            for j in range (n-2):
                
                # This is the precceding node of node i
                current_node = lookup_table[int(prev_node)][idx]

                # Append this to the path as well
                path[i].append(current_node)

                # New power set index
                idx = 2**int(prev_node) ^ idx

                prev_node = current_node

    return path

def main():

    adj = np.array([[0, 1, 0, 0],
                    [1, 0, 1, 1],
                    [0, 1, 0, 1],
                    [0, 1, 1, 0]])

    #adj = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
    #                [1, 0, 1, 0, 0, 0, 0, 0],
    #                [0, 1, 0, 0, 1, 0, 1, 1],
    #                [1, 0, 0, 0, 0, 0, 1, 1],
    #                [1, 0, 1, 0, 0, 0, 1, 0],
    #                [0, 0, 0, 0, 0, 0, 1, 1],
    #                [1, 0, 1, 1, 1, 1, 0, 0],
    #                [0, 0, 1, 1, 0, 1, 0, 0]])

    # Generate a lookup table for the adjacency matrix
    lookup = traverse(adj)

    # Find the path
    # This will return ALL hamiltonian paths
    path = path_traceback(lookup)

    return 0

if __name__ == '__main__':
    main()
