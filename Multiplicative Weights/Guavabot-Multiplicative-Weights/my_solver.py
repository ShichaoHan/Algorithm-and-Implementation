
import networkx as nx
import numpy as np
from sortedcontainers import SortedDict
import math
def solve(client):
    '''
    :param client: an instance of Guavabot graph; see project description for more details
    The solver function would perform scout, remote and other operations on the client
    instance in order to get all the Guavabots in the client instance back to home.
    '''
    ##Getting start with a new client instance
    client.end()
    client.start()
    ##Getting the networkx graph instance for the city
    guavaGraph = client.G
    ##Getting the homenode for the city
    guavaHome = client.home
    ##Getting the nodes list of the city
    mst = nx.minimum_spanning_tree(guavaGraph)
    mst_leaves = []
    for node in mst.nodes():
        if len(list(mst.neighbors(node))) == 1:
            mst_leaves += [node]

    predecessor = nx.dfs_predecessors(mst, guavaHome)
    topological_order = list(nx.dfs_postorder_nodes(mst, guavaHome))
    all_students = list(np.arange(client.k)+1)
    ## Initiate a dictionary that stores the total number of times that the students made mistake
    mistake = {}
    for student in all_students:
        mistake[student] = 1

    ## Keep track of the nodes that must have some bots on it
    must_have = {}
    for node in guavaGraph.nodes:
        must_have[node] = 0

    ##Overall scout results:
    overall_results = {}

    ##A dictionary; key: node and value: whether the node can be scouted in the future
    cannotscout = {}
    for node in guavaGraph.nodes:
        cannotscout[node] = False

    ##A dictionary; key: node and value: whether the node has already been explored
    explored = {}
    for node in guavaGraph.nodes():
        explored[node] = False

    ##The list of nodes that have been remoted
    remoted_before = []


    def scout_nodes(nodes, students):
        '''
        :param nodes: A list of nodes that the students scout on
        :param students: A list of students to perform the scout task
        :return: None
        '''
        nonlocal overall_results, stop_scouting
        for node in nodes:
            overall_results[node]= client.scout(node, students)

    def calculate_confidence(theNode):
        '''
        :param scoutResults: A dictionary of scouting results; key: student; value:result
        :param mistakeDict: An array of students' mistake count; indexed by student - 1
        :return: a float in [0,1] that indicates the weighted confidence in that the current node has bot
        '''
        nonlocal overall_results, mistake, confid
        weighted_confidence = 0
        current_node_result = overall_results[theNode]
        if current_node_result is None:
            return int(client.bot_count[theNode] != 0)
        total_weights = 0
        for s in list(current_node_result.keys()):
            total_weights += mistake[s]
        for s in list(current_node_result.keys()):
            weighted_confidence += current_node_result[s] * mistake[s] / total_weights
        confid += [weighted_confidence]
        return weighted_confidence

    def remote_and_update(fromNode, toNode, epsilon):
        '''
        THIS PART IS EQUIVALENT TO EXPLOITATION AT DAT T AND LEARNING THE LOSS OF THE DAY
        :param fromNode: node to perform remote from
        :param toNode: node to perform remote to
        Remote and update function perform remote from fromNode to toNode.
        Based on the  result of the remote operation, we update the weights of each corresponding student.
        Use the negative entropy loss function and penalize the students who make correct predictions.
        '''
        nonlocal mistake, cannotscout, overall_results, stop_scouting, must_have, num_bots, explored, remoted_before
        remote_result = client.remote(fromNode, toNode)
        remoted_before += [fromNode]
        explored[fromNode] = True
        cannotscout[fromNode] = True
        if remote_result > 0:
            cannotscout[toNode] = True
            must_have[toNode] += remote_result
            must_have[fromNode] = 0


        if fromNode in list(overall_results.keys()):
            expectedAnswer = (remote_result > 0)
            updateResults = overall_results[fromNode]
            if not updateResults:
                return
            for student in list(updateResults.keys()):
                if updateResults[student] == expectedAnswer:
                    mistake[student] *= (1 - epsilon)
                else:
                    mistake[student] *= (1 - epsilon)**(-1)


    def my_solver(ep):
        nonlocal overall_results, topological_order, explored, guavaHome
        nonlocal guavaGraph, predecessor, mistake, total_vert, explored
        nonlocal all_students, confid_2, pct, remoted_before, node_confidence


        for i in range(len(topological_order) - 1):
            stu_to_scout = all_students
            scout_nodes([topological_order[i]], stu_to_scout)
        while client.bot_count[client.home] != client.l:
            highest_confid = 0
            highest_node = 0
            for node in list(overall_results.keys()):
                current_confid = calculate_confidence(node)
                node_confidence[node] = current_confid
            for n in list(node_confidence.keys()):
                if node_confidence[n] >= highest_confid:
                    highest_confid = node_confidence[n]
                    highest_node = n



            curNei = guavaGraph.neighbors(highest_node)

            shortest_nei = 0
            shortest_len = float('inf')

            for nei in curNei:
                dist = guavaGraph[highest_node][nei]['weight']
                if dist < shortest_len:
                    shortest_len = dist
                    shortest_nei = nei
            if shortest_nei != client.home:
                cutoff = 0.4
                if len(all_students) == 20:
                    cutoff = 0.6
                elif len(all_students) == 10:
                    cutoff = 0.7
                else:
                    cutoff = 0.5
                if shortest_len / guavaGraph[client.home][highest_node]['weight'] >= cutoff:
                    shortest_nei = client.home
            remote_and_update(highest_node, shortest_nei, ep)
            overall_results[highest_node] = None
            if client.bot_count[shortest_nei] == 0:
                continue
            if shortest_nei == client.home:
                continue
            path = nx.shortest_path(guavaGraph, shortest_nei, client.home)
            for k in range(len(path) - 1):
                remote_and_update(path[k], path[k + 1], ep)

    ## Setting different values of epsilon for instances with different number of students
    if len(all_students) == 10:
        epsilo = 0.01
    elif len(all_students) == 20:
        epsilo = 0.03
    else:
        epsilo = 0.05
    #exploration(deg, epsilo)
    my_solver(epsilo)
    client.end()
