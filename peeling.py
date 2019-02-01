#!/usr/bin/python2.7

import os
import heapq
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

# node = 1559667
node = 1557943
# node = 10

# get the complete graph list, this graph list is an adjacency list of the graph
# eg: graph_list[2] - 4,5 => there are two edges 2->4 and 2->5
graph_list = [[] for x in xrange(node)]
# with open("/home/akhil/Documents/hadoop_project/graph_ab_ba_all_edges_converted.txt", "r") as f:
with open("/home/akhil/Documents/hadoop_project/graph_ab_ba_distinct_converted.txt", "r") as f:
    for line in f:
        line = line.strip(" '\n")
        line = line.split(" ")
        line = map(int, line)
        for i in range(1, len(line)):
            graph_list[line[0]].append(line[i])

# We find cores of the given graph with specified number of nodes according to the algorithm
# specified in the paper. Graph is assumed to be of the form of graph_list given above.
# Return value is a degree array with the core number for each node
def cores(graph, node):
    size, md, d = 0, 0, 0
    deg = [0]*node
    vert = [0]*node
    pos = [0]*node

    for i in range(1, node):
        d = 0
        deg[i] = len(graph[i])
        if deg[i] > md: md = deg[i]

    bins = [0]*(md+1)

    for v in range(1, int(node)):
        bins[deg[v]] = bins[deg[v]] + 1

    start = 1
    for d in range(0, len(bins)):
        num = bins[d]
        bins[d] = start
        start = start + num

    for v in range(1, int(node)):
        pos[v] = bins[deg[v]]
        vert[pos[v]] = v
        start = start + num
        bins[deg[v]] = bins[deg[v]] + 1

    for d in range(md, 1, -1):
        bins[d] =  bins[d-1]
    bins[0] = 1

    for i in range(1, int(node)):
        v = vert[i]
        for u in graph[v]:
            position = u
            if deg[position] > deg[v]:
                du = deg[position]
                pu = pos[position]
                pw = bins[du]
                w = vert[pw]
                if position != w:
                    pos[position] = pw
                    vert[pu] = w
                    pos[w] = pu
                    vert[pw] = position
                bins[du] = bins[du] + 1
                deg[position] = deg[position] - 1
    return(deg)


# These are two sample graphs, for which this code has been tested.
# graph_list = [[],
# [3,5],
# [3,4],
# [1,2,4,6],
# [2,3],
# [1],
# [3,7,8,9],
# [6,8,9],
# [6,7,9],
# [6,7,8]]

# graph_list = [[],
# [2, 3],
# [1, 3, 4, 8],
# [1, 2],
# [2, 5, 6, 7, 8],
# [4],
# [4],
# [4],
# [2, 4]]

# As name suggests, returns the maximum core value of the graph
def getMaxCore(graph_list, node):
    max_cores = 0
    deg = cores(graph_list, node)
    for u in deg:
        if u > max_cores:
            max_cores = u
    return max_cores

max_cores = getMaxCore(graph_list, node) + 1

# v_core contains vertices in a particular peel layer
# eg: v_core[2] - 4,5 => second peel layer has vertex 4 and 5
v_core = [[] for x in xrange(max_cores)]

# e_core contains the edges in a particular peel layer
# eg: e_core[2] - (4,5), (6,7) => second peel layer has edge(4,5) and edge(6,7)
e_core = [[] for x in xrange(max_cores)]

def flatten(seq,container=None):
    if container is None:
        container = []
    for s in seq:
        if hasattr(s,'__iter__'):
            flatten(s,container)
        else:
            container.append(s)
    return container

# This function is used to check if peeling is finished or not. Empty graph_list implies that
# all the layers have been peeled.
def isListEmpty(inList):
    if isinstance(inList, list):    # Is a list
        return all( map(isListEmpty, inList) )
    return False

k = node
# Peeling loop as specified in the paper starts from here.
while isListEmpty(graph_list) == False:
    # We calculate cores each time the edges have been peeled
    deg = cores(graph_list, node)
    # creating a heap for all the vertices with its core number as key
    cores_heap = []
    heapq.heapify(cores_heap)
    for i in range(1, node):
        # We multiply by -1, since we want to create a max-heap using a min-heap.
        heapq.heappush(cores_heap, (-1*deg[i], i))

    k_ele = heapq.heappop(cores_heap)
    # since heap is formed using negative of the core values, we multiply by -1
    k = -1*k_ele[0]
    v_core[k].append(k_ele[1])
    print("k-value " + str(k))
    # We pop the highest core value repeatedly until there is no more highest core value left
    # in the graph. This collection of vertices form the vertex peel decomposition of a
    # particular layer.
    while cores_heap[0][0] == -1*k:
        k_ele = heapq.heappop(cores_heap)
        v_core[k].append(k_ele[1])
    # We use set operations, as they are convenient for edge deletions
    a = set(v_core[k])
    for u in v_core[k]:
        b = set(graph_list[u])
        c = a.intersection(b)
        for d in list(c):
            # We add the relevant edges to the edge decomposition peel layer
            e_core[k].append((u,d))
        # removing the edges from the current_graph
        graph_list[u] = list(b.difference(c))

no_components = list()
size_components = list()
# This function finds connected components in graph_list with specified number of nodes
# using Union-Find data structure.
# Return value is the number of connected components in the graph_list.
def connected_components(edge_list, nodes):
    A = [0]*nodes
    rank = [1]*nodes
    def unionSet(a, b):
        x = find(a)
        y = find(b)
        if x != y:
            if rank[x] > rank[y]:
                A[y] = x
                rank[x] = rank[x] + rank[y]
            else:
                A[x] = y
                rank[y] = rank[x] + rank[y]

    def find(a):
        i = a
        while A[i] != i:
            i = A[i]
        A[a] = i
        return i

    def createSet():
        for i in range(0, nodes):
            A[i] = i

    createSet()
    # We create a set a, so that we can record all the elements that we have
    # actually seen.
    a = set()
    for edge in edge_list:
        unionSet(edge[0], edge[1])
        a.add(edge[0])
        a.add(edge[1])
    components = [[] for x in xrange(nodes)]
    # We run find once for all the elements, so that path compression can take
    # place, and all the nodes directly point to the root of the tree.
    for i in range(1, nodes):
        find(i)
    # print("array A is " + str(A))
    # All nodes that are in the same component are grouped together in a list.
    # We put a check of its presence in set a, so as to omit nodes that are not
    # present in the edge_list
    for i in range(1, nodes):
        if i in a:
            components[A[i]].append(i)

    list3 = [x for x in components if x != []]
    list4 = [len(x) for x in list3]
    no_components.append(len(list3))
    size_components.append(list4)
    return(len(list3))

def connected_comp(e_core, max_cores, nodes):
    for i in range(0, max_cores):
        if len(e_core[i]) != 0:
            print("Layer " + str(i) + " has " + str(connected_components(e_core[i], nodes)) +
            " connected components")
connected_comp(e_core, max_cores, node)
print("number of components " + str(no_components))
print("size of components " + str(size_components))

max_size = max(flatten(size_components)) + 1
print(max_size)
max_size = 500

whatever =  list()
for what in size_components:
    whatever.append(Counter(what))
print(whatever)

f = open("/home/akhil/Documents/hadoop_project/peel_outputB.txt", "w")

p1 = list()
for i in range(0, max_size):
    p1.append(whatever[0][i])
    if whatever[0][i] != 0:
        f.write("1 " + str(i) + " " + str(whatever[0][i]) + "\n")
p2 = list()
for i in range(0, max_size):
    p2.append(whatever[1][i])
    if whatever[1][i] != 0:
        f.write("2 " + str(i) + " " + str(whatever[1][i]) + "\n")
p3 = list()
for i in range(0, max_size):
    p3.append(whatever[2][i])
    if whatever[2][i] != 0:
        f.write("3 " + str(i) + " " + str(whatever[2][i]) + "\n")
p4 = list()
for i in range(0, max_size):
    p4.append(whatever[3][i])
    if whatever[3][i] != 0:
        f.write("4 " + str(i) + " " + str(whatever[3][i]) + "\n")
p5 = list()
for i in range(0, max_size):
    p5.append(whatever[4][i])
    if whatever[4][i] != 0:
        f.write("5 " + str(i) + " " + str(whatever[4][i]) + "\n")
p6 = list()
for i in range(0, max_size):
    p6.append(whatever[5][i])
    if whatever[5][i] != 0:
        f.write("6 " + str(i) + " " + str(whatever[5][i]) + "\n")
p7 = list()
for i in range(0, max_size):
    p7.append(whatever[6][i])
    if whatever[6][i] != 0:
        f.write("7 " + str(i) + " " + str(whatever[6][i]) + "\n")
p8 = list()
for i in range(0, max_size):
    p8.append(whatever[7][i])
    if whatever[7][i] != 0:
        f.write("8 " + str(i) + " " + str(whatever[7][i]) + "\n")
p9 = list()
for i in range(0, max_size):
    p9.append(whatever[8][i])
    if whatever[8][i] != 0:
        f.write("9 " + str(i) + " " + str(whatever[8][i]) + "\n")
p10 = list()
for i in range(0, max_size):
    p10.append(whatever[9][i])
    if whatever[9][i] != 0:
        f.write("10 " + str(i) + " " + str(whatever[9][i]) + "\n")
p11 = list()
for i in range(0, max_size):
    p11.append(whatever[10][i])
    if whatever[10][i] != 0:
        f.write("11 " + str(i) + " " + str(whatever[10][i]) + "\n")
p12 = list()
for i in range(0, max_size):
    p12.append(whatever[11][i])
    if whatever[11][i] != 0:
        f.write("12 " + str(i) + " " + str(whatever[11][i]) + "\n")
p13 = list()
for i in range(0, max_size):
    p13.append(whatever[12][i])
    if whatever[12][i] != 0:
        f.write("13 " + str(i) + " " + str(whatever[12][i]) + "\n")
p14 = list()
for i in range(0, max_size):
    p14.append(whatever[13][i])
    if whatever[13][i] != 0:
        f.write("14 " + str(i) + " " + str(whatever[13][i]) + "\n")
p15 = list()
for i in range(0, max_size):
    p15.append(whatever[14][i])
    if whatever[14][i] != 0:
        f.write("15 " + str(i) + " " + str(whatever[14][i]) + "\n")
p16 = list()
for i in range(0, max_size):
    p16.append(whatever[15][i])
    if whatever[15][i] != 0:
        f.write("16 " + str(i) + " " + str(whatever[15][i]) + "\n")
p17 = list()
for i in range(0, max_size):
    p17.append(whatever[16][i])
    if whatever[16][i] != 0:
        f.write("17 " + str(i) + " " + str(whatever[16][i]) + "\n")
p18 = list()
for i in range(0, max_size):
    p18.append(whatever[17][i])
    if whatever[17][i] != 0:
        f.write("18 " + str(i) + " " + str(whatever[17][i]) + "\n")

f.close()
print(len(p2))
print(p2[4])
print(p2[3])
print(p2[2])
header = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
'11', '12', '13', '14', '15', '16', '17', '18']
X = range(max_size)
q1 = plt.bar(X, p1, color = (0.2, 0.2, 1))
q2 = plt.bar(X, p2, bottom = p1, color = (0.2, 1, 0.2))
q3 = plt.bar(X, p3, bottom = p2, color = (1, 0.2, 0.2))
q4 = plt.bar(X, p4, bottom = p3, color = (0.2, 0.2, 0.2))
q5 = plt.bar(X, p5, bottom = p4, color = (0.2, 1, 0.2))
q6 = plt.bar(X, p6, bottom = p5, color = (1, 0.2, 0.2))
q7 = plt.bar(X, p7, bottom = p6, color = (0.2, 0.2, 1))
q8 = plt.bar(X, p8, bottom = p7, color = (0.2, 1, 0.2))
q9 = plt.bar(X, p9, bottom = p8, color = (1, 0.2, 0.2))
q10 = plt.bar(X, p10, bottom = p9, color = (0.2, 0.2, 0.2))
q11 = plt.bar(X, p11, bottom = p10, color = (0.2, 1, 0.2))
q12 = plt.bar(X, p12, bottom = p11, color = (1, 0.2, 0.2))
q13 = plt.bar(X, p13, bottom = p12, color = (0.2, 0.2, 1))
q14 = plt.bar(X, p14, bottom = p13, color = (0.2, 1, 0.2))
q15 = plt.bar(X, p15, bottom = p14, color = (1, 0.2, 0.2))
q16 = plt.bar(X, p16, bottom = p15, color = (0.2, 0.2, 0.2))
q17 = plt.bar(X, p17, bottom = p16, color = (0.2, 1, 0.2))
q18 = plt.bar(X, p18, bottom = p17, color = (1, 0.2, 0.2))
plt.ylim([0, 20])
plt.ylabel("No of components")
plt.xlabel("Size of components")
plt.legend((q1[0], q2[0], q3[0], q4[0], q5[0], q6[0], q7[0], q8[0], q9[0]
, q10[0], q11[0], q12[0], q13[0], q14[0], q15[0], q16[0], q17[0], q18[0]), (header[0], header[1], header[2],
header[3], header[4], header[5], header[6], header[7], header[8], header[9], header[10],
header[11], header[12], header[13], header[14], header[15], header[16], header[17]))
plt.show()

# Processing the v_core list so as to delete repeating vertices from lower core numbers
b = set()
for i in range(max_cores - 1, 0, -1):
    a = set(v_core[i])
    a = a.difference(b)
    b = a.union(b)
    v_core[i] = list(a)

f = open("/home/akhil/Documents/hadoop_project/peel_outputA.txt", "w")
j = 0
for i in range(0, max_cores):
    if len(v_core[i]) != 0:
        f.write(str(j) + " " + str(len(v_core[i])) + " " + str(len(e_core[i])) + "\n")
        j = j + 1
f.close()
