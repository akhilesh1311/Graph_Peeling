#!/usr/bin/python2.7

import os
# from graph_tool.all import *
import bisect
import matplotlib.pyplot as plt

# edge = 28280084
node = 1559667
# node = 13

size, md, d = 0, 0, 0
graph = [[] for x in xrange(node)]
vertices = [0]*node
deg = [0]*node
with open("/home/akhil/Documents/hadoop_project/graph_ab_ba_all_edges_python.txt", "r") as f:
    for line in f:
        size = size + 1

i = 1
with open("/home/akhil/Documents/hadoop_project/graph_ab_ba_all_edges_python.txt", "r") as f:
    for line in f:
        d = 0
        line = line.strip("\n")
        line = line.split(",")
        for u in range(0, len(line)):
            d = d + 1
            graph[i].append(int(u))
        vertices[i] = int(line[0])
        deg[i] = d - 1
        i = i + 1
        if d > md: md = d

f = open("/home/akhil/Documents/hadoop_project/degree.txt", "w")
for i in range(1, node):
    f.write(str(i) + "," + str(vertices[i]) + "," + str(deg[i]) + "\n")
f.close()

plt.plot(range(0,node), deg, 'ro')
plt.axis([0, node, 0, 2500])
plt.show()
