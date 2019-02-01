#!/usr/bin/python2.7

import os
import matplotlib.pyplot as plt

edge = 28280084
# edge = 10

A = [0]*edge
rank = [1]*edge
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
    for i in range(0, edge):
        A[i] = i

i = 0
max = 0
createSet()
single_comp = []
with open("/home/akhil/Documents/hadoop_project/graph_ab_ba_all_edges_python.txt", "r") as f:
    for line in f:
        line = line.strip("\n")
        line = line.split(",")
        if len(line) == 2 and int(line[0]) == int(line[1]):
            print(line[0])
            single_comp.append(int(line[0]))
            continue
        for i in range(1,len(line)):
            unionSet(int(line[0]), int(line[i]))
            # print(line)
            # print(line[0], line[i])
# print(A)
del rank
components = [[] for x in xrange(edge)]
for i in range(1, edge):
    find(i)

for i in range(1, edge):
    components[A[i]].append(i)

list2 = [x for x in components if len(x) != 1]
list3 = [x for x in list2 if x != []]

print(list3)

size_components = []
max_size = 0
f = open("/home/akhil/Documents/hadoop_project/connected_components.txt", "w")
for comp in list3:
    size_components.append(len(comp))
    print(len(comp))
    if len(comp) > max_size:
        max_size = len(comp)

print(max_size)
print(size_components)

number_of_comp_of_size = [0]*(max_size+1)
for s in size_components:
    number_of_comp_of_size[s] = number_of_comp_of_size[s] + 1

print(number_of_comp_of_size)

i = 0
for s in number_of_comp_of_size:
    f.write(str(i) + "," + str(number_of_comp_of_size[i]) + "\n")
    i = i + 1

plt.plot(range(0, max_size+1), number_of_comp_of_size, 'ro')
plt.axis([0, max_size, 0, 1500])
plt.show()
