#!/usr/bin/python2.7

import os
from graph_tool.all import *

i = 0
max = 0
edges = 0
with open("/home/akhil/Documents/hadoop_project/graph_ab_ba_all_edges_python.txt", "r") as f:
    for line in f:
        i = i + 1
        line = line.split(",")
        edges = edges + len(line) - 1
        for node in line:
            if (int(node) > max):
                max = int(node)
print(i)
print(max)
print(edges)
