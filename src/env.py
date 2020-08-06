import numpy as np
import queue
from math import *
from copy import deepcopy
import time
from queue import PriorityQueue as PQ
import threading
import pickle
_x = [-1,1,0,0]
_y = [0,0,-1,1]
import ctypes
from numpy.ctypeslib import ndpointer

so = ctypes.CDLL('./search.so')
search = so.search
search.argtypes = [ndpointer(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,ndpointer(ctypes.c_int),ndpointer(ctypes.c_int),ndpointer(ctypes.c_int),ndpointer(ctypes.c_int)]
search.restype = None

search_transpose = so.search_transpose
search_transpose.argtypes = [ndpointer(ctypes.c_int), ctypes.c_int, ctypes.c_int, 
ctypes.c_int, 
ctypes.c_float, ctypes.c_float,
ctypes.c_float, ctypes.c_float, 
ctypes.c_int, ctypes.c_int, ctypes.c_int,
ndpointer(ctypes.c_int),ndpointer(ctypes.c_int), ctypes.c_int,
ndpointer(ctypes.c_float),ndpointer(ctypes.c_float),ndpointer(ctypes.c_int),ndpointer(ctypes.c_int),ndpointer(ctypes.c_int),
ndpointer(ctypes.c_int)]
search_transpose.restype = None

search_transpose_poly = so.search_transpose_poly
search_transpose_poly.argtypes = [ndpointer(ctypes.c_int), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_int), ctypes.c_int, ctypes.c_int, 
ctypes.c_int, ctypes.c_int, 
ndpointer(ctypes.c_float),ndpointer(ctypes.c_float),
ndpointer(ctypes.c_float),ndpointer(ctypes.c_float),
ndpointer(ctypes.c_int),ndpointer(ctypes.c_int), ctypes.c_int,
ndpointer(ctypes.c_float),ndpointer(ctypes.c_float),ndpointer(ctypes.c_int),ndpointer(ctypes.c_int),ndpointer(ctypes.c_int),
ndpointer(ctypes.c_int)]
search_transpose_poly.restype = None

search_transpose_poly_rot = so.search_transpose_poly_rot
search_transpose_poly_rot.argtypes = [ndpointer(ctypes.c_int), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_int), ctypes.c_int, ctypes.c_int, 
ctypes.c_int, ctypes.c_int, 
ndpointer(ctypes.c_float),ndpointer(ctypes.c_float),
ndpointer(ctypes.c_float),ndpointer(ctypes.c_float),
ndpointer(ctypes.c_int),ndpointer(ctypes.c_int), ctypes.c_int,
ndpointer(ctypes.c_float),ndpointer(ctypes.c_float),ndpointer(ctypes.c_int),ndpointer(ctypes.c_int),ndpointer(ctypes.c_int),
ndpointer(ctypes.c_int)]
search_transpose_poly_rot.restype = None


translate = so.translate
translate.argtypes = [ndpointer(ctypes.c_int), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_int), ctypes.c_int, ctypes.c_int, 
ctypes.c_int, ctypes.c_int, ctypes.c_int,
ndpointer(ctypes.c_float), ndpointer(ctypes.c_float),
ndpointer(ctypes.c_int),ndpointer(ctypes.c_int), ctypes.c_int,
ndpointer(ctypes.c_int)]
translate.restype = None

EPS = 1e-4

class ENV:
    def __init__(self, size=(5,5), start=(0,0)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.start = start

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, size): 
        self.map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.mark = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            for gox in range(dx):
                for goy in range(dy):
                    self.map[x + gox,y + goy] = i + 2

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index):
        if index >= len(self.mark) or self.mark[index] == 1:
            return -500, 0
        target = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        Q = queue.Queue()
        self.mark[index] = 1

        # to confirm the starting point is legal
        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[self.start[0]+i,self.start[1]+j] == 1:
                    l = False
                    break
            if not l:
                break
        if l:
            Q.put(self.start)
            dis[self.start[0], self.start[1]] = 0

        # BFS
        while not Q.empty():
            a = Q.get()
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal
            
            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    break

                for i, dx in enumerate(_x):
                    for dy in _y:
                        tx = x + dx
                        ty = y + dy
                        if tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                            Q.put((tx,ty))
                            dis[tx,ty] = d + 1
        
        

        # with open("dis.txt",'a') as fp:
        #     for _ in dis:
        #         for __ in _:
        #             fp.write('%d '%__)
        #         fp.write('\n')
        #     fp.write('\n')
        if dis[target[0],target[1]] == -1:
            return -1000, 1

        for i in range(size[0]):
            for j in range(size[1]):
                self.map[target[0]+i, target[1]+j] = 1

        if np.sum(self.mark) == len(self.mark):
            return 1000-dis[target[0],target[1]], 1

        return -dis[target[0],target[1]], 0

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self):
        import random
        size = self.map_size
        while True:
            dx = np.random.randint(2,4)
            dy = np.random.randint(2,4)
            dx_ = np.random.randint(2,4)
            dy_ = np.random.randint(2,4)
            alter = []

            for i in range(size[0]):
                if i + dx - 1 < size[0]:
                    for j in range(size[1]):
                        if j + dy - 1 < size[1]:
                            for k in range(i+1,size[0]):
                                if k + dx_ - 1 < size[0]:
                                    for l in range(j+1,size[1]):
                                        if l + dy_ - 1 < size[1]:
                                            if not (i + dx - 1 >= k and j + dy - 1 >= l):
                                                alter.append([(i,j),(k,l)])
            
            num = len(alter)
            if num > 0:
                choice = np.random.randint(num)
                pair = alter[choice]
                break
        
        pos_list = []
        size_list = []
        list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))]
        random.shuffle(list_)
        for i in list_:
            pos_list.append(i[0])
            size_list.append(i[1])
        
        self.setmap(pos_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):
        
    def getstate(self):
        return np.array(self.map)

'''
    Environment which has 3 objects.
'''
class ENV3:
    def __init__(self, size=(5,5), start=(0,0)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.start = start

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, size): 
        self.map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.mark = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            for gox in range(dx):
                for goy in range(dy):
                    self.map[x + gox,y + goy] = i + 2

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index):
        if index >= len(self.mark) or self.mark[index] == 1:
            return -500, -1
        target = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        Q = queue.Queue()
        self.mark[index] = 1

        # to confirm the starting point is legal
        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[self.start[0]+i, self.start[1]+j] == 1:
                    l = False
                    break
            if not l:
                break
        if l:
            Q.put(self.start)
            dis[self.start[0], self.start[1]] = 0

        # BFS
        while not Q.empty():
            a = Q.get()
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal

            if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                l = False

            for i in range(size[0]):
                if x + i < self.map_size[0]:
                    for j in range(size[1]):
                        if y + j < self.map_size[1]:
                            if self.map[x+i, y+j] == 1:
                                l = False
                                break
                if not l:
                    break

            # if not l:
            #     print('illegal',x,y)
            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    break

                for i, dx in enumerate(_x):
                    for dy in _y:
                        tx = x + dx
                        ty = y + dy
                        if tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                            Q.put((tx,ty))
                            dis[tx,ty] = d + 1
        
        

        # with open("dis.txt",'a') as fp:
        #     for _ in dis:
        #         for __ in _:
        #             fp.write('%d '%__)
        #         fp.write('\n')
        #     fp.write('\n')
        if dis[target[0],target[1]] == -1:
            return -1000, 1

        for i in range(size[0]):
            for j in range(size[1]):
                self.map[target[0]+i, target[1]+j] = 1

        if np.sum(self.mark) == len(self.mark):
            return 1000-dis[target[0],target[1]], 1

        return -dis[target[0],target[1]], 0

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=3):
        import random
        size = self.map_size
        # print(size)
        pos_list = []
        size_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break

            

        
        random.shuffle(pos_list)
        for i in range(num):
            px, py = pos_list[i]
            
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    if j > len(size_list) -1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            

        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):
        
    def getstate(self):
        return np.array(self.map)

class ENV_M:
    def __init__(self, size=(5,5), start=(0,0)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.start = start
        

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, size): 
        self.map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.mark = np.zeros(len(pos))
        self.dis = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            for gox in range(dx):
                for goy in range(dy):
                    self.map[x + gox,y + goy] = i + 2

        self.check()

    '''
        check the state
    '''
    def check(self):
        to_check = []
        for i in range(len(self.mark)):
            if not self.mark[i]:
                to_check.append(i)
        
        flag = True
        for index in to_check:
            target = self.pos[index]
            size = self.size[index]
            dis = np.ones_like(self.map)
            dis *= -1                   # when it starts, all distance was initialized as -1 
            Q = queue.Queue()

            # to confirm the starting point is legal
            l = True
            for i in range(size[0]):
                for j in range(size[1]):
                    if self.map[self.start[0]+i, self.start[1]+j] == 1:
                        l = False
                        break
                if not l:
                    break
            if l:
                Q.put(self.start)
                dis[self.start[0], self.start[1]] = 0

            # BFS
            while not Q.empty():
                a = Q.get()
                x,y = a
                d = dis[x,y]
                l = True     # whether it is legal

                if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                    l = False

                for i in range(size[0]):
                    if x + i < self.map_size[0]:
                        for j in range(size[1]):
                            if y + j < self.map_size[1]:
                                if self.map[x+i, y+j] == 1:
                                    l = False
                                    break
                    if not l:
                        break

                # if not l:
                #     print('illegal',x,y)
                if l:
                    # dis[x, y] = d + 1
                    if x == target[0] and y == target[1]:
                        break

                    for i, dx in enumerate(_x):
                        for dy in _y:
                            tx = x + dx
                            ty = y + dy
                            if tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                                Q.put((tx,ty))
                                dis[tx,ty] = d + 1
            
            
            if dis[target[0],target[1]] == -1:
                flag = False
                break

            self.dis[index] = dis[target[0],target[1]]

        if len(to_check) == 0:
            return 1

        if flag:
            return 0

        return -1        
        # return to_check            

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index):
        if index >= len(self.mark) or self.mark[index] == 1:
            return -500, -1

        size = self.size[index]
        target = self.pos[index]
        self.mark[index] = 1
        dis = self.dis[index]

        for i in range(size[0]):
            for j in range(size[1]):
                self.map[target[0]+i, target[1]+j] = 1

        # check if legal, calc the distance
        flag = self.check()
        if flag == 1:
            return 1000 - dis, 1
        elif flag == -1:
            return -1000, 1
        else:
            return 50 - dis, 0

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=3):
        import random
        size = self.map_size
        # print(size)
        pos_list = []
        size_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break

            

        
        random.shuffle(pos_list)
        for i in range(num):
            px, py = pos_list[i]
            
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    if j > len(size_list) -1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            

        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):
        
    def getstate(self):
        return np.array(self.map)

class ENV_M_C:
    def __init__(self, size=(5,5), start=(0,0)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.start = start
        

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, size): 
        self.map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.mark = np.zeros(len(pos))
        self.dis = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            for gox in range(dx):
                for goy in range(dy):
                    self.map[x + gox,y + goy] = i + 2

        self.check()

    '''
        check the state
    '''
    def check(self):
        to_check = []
        for i in range(len(self.mark)):
            if not self.mark[i]:
                to_check.append(i)
        
        flag = True
        for index in to_check:
            target = self.pos[index]
            size = self.size[index]
            dis = np.ones_like(self.map)
            dis *= -1                   # when it starts, all distance was initialized as -1 
            Q = queue.Queue()

            # to confirm the starting point is legal
            l = True
            for i in range(size[0]):
                for j in range(size[1]):
                    if self.map[self.start[0]+i, self.start[1]+j] == 1:
                        l = False
                        break
                if not l:
                    break
            if l:
                Q.put(self.start)
                dis[self.start[0], self.start[1]] = 0

            # BFS
            while not Q.empty():
                a = Q.get()
                x,y = a
                d = dis[x,y]
                l = True     # whether it is legal

                if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                    l = False

                for i in range(size[0]):
                    if x + i < self.map_size[0]:
                        for j in range(size[1]):
                            if y + j < self.map_size[1]:
                                if self.map[x+i, y+j] == 1:
                                    l = False
                                    break
                    if not l:
                        break

                # if not l:
                #     print('illegal',x,y)
                if l:
                    # dis[x, y] = d + 1
                    if x == target[0] and y == target[1]:
                        break

                    for i, dx in enumerate(_x):
                        for dy in _y:
                            tx = x + dx
                            ty = y + dy
                            if tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                                Q.put((tx,ty))
                                dis[tx,ty] = d + 1
            
            
            if dis[target[0],target[1]] == -1:
                flag = False
                break

            self.dis[index] = dis[target[0],target[1]]

        if len(to_check) == 0:
            return 1

        if flag:
            return 0

        return -1        
        # return to_check            

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index):
        if index >= len(self.mark) or self.mark[index] == 1:
            return -500, -1

        size = self.size[index]
        target = self.pos[index]
        self.mark[index] = 1
        dis = self.dis[index]

        for i in range(size[0]):
            for j in range(size[1]):
                self.map[target[0]+i, target[1]+j] = 1

        # check if legal, calc the distance
        flag = self.check()
        if flag == 1:
            return 1000 - dis, 1
        elif flag == -1:
            return -1000, 1
        else:
            return 50 - dis, 0

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=3):
        import random
        size = self.map_size
        # print(size)
        pos_list = []
        size_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break

            

        
        random.shuffle(pos_list)
        for i in range(num):
            px, py = pos_list[i]
            
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    if j > len(size_list) -1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            

        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):
        
    def getstate(self):
        state = []
        for i in range(len(self.mark)+1):
            state.append((self.map == (i+1)).astype(np.float32))
        
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):
        return np.array(self.map)

class ENV_M_C_5:
    def __init__(self, size=(5,5), start=(0,0)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.start = start
        

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, size): 
        self.map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.mark = np.zeros(len(pos))
        self.dis = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            for gox in range(dx):
                for goy in range(dy):
                    self.map[x + gox,y + goy] = i + 2

        self.check()

    '''
        check the state
    '''
    def check(self):
        to_check = []
        for i in range(len(self.mark)):
            if not self.mark[i]:
                to_check.append(i)
        
        flag = True
        for index in to_check:
            target = self.pos[index]
            size = self.size[index]
            dis = np.ones_like(self.map)
            dis *= -1                   # when it starts, all distance was initialized as -1 
            Q = queue.Queue()

            # to confirm the starting point is legal
            l = True
            for i in range(size[0]):
                for j in range(size[1]):
                    if self.map[self.start[0]+i, self.start[1]+j] == 1:
                        l = False
                        break
                if not l:
                    break
            if l:
                Q.put(self.start)
                dis[self.start[0], self.start[1]] = 0

            # BFS
            while not Q.empty():
                a = Q.get()
                x,y = a
                d = dis[x,y]
                l = True     # whether it is legal

                if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                    l = False

                for i in range(size[0]):
                    if x + i < self.map_size[0]:
                        for j in range(size[1]):
                            if y + j < self.map_size[1]:
                                if self.map[x+i, y+j] == 1:
                                    l = False
                                    break
                    if not l:
                        break

                # if not l:
                #     print('illegal',x,y)
                if l:
                    # dis[x, y] = d + 1
                    if x == target[0] and y == target[1]:
                        break

                    for i, dx in enumerate(_x):
                        for dy in _y:
                            tx = x + dx
                            ty = y + dy
                            if tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                                Q.put((tx,ty))
                                dis[tx,ty] = d + 1
            
            
            if dis[target[0],target[1]] == -1:
                flag = False
                break

            self.dis[index] = dis[target[0],target[1]]

        if len(to_check) == 0:
            return 1

        if flag:
            return 0

        return -1        
        # return to_check            

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index):
        if index >= len(self.mark) or self.mark[index] == 1:
            return -500, -1

        size = self.size[index]
        target = self.pos[index]
        self.mark[index] = 1
        dis = self.dis[index]

        for i in range(size[0]):
            for j in range(size[1]):
                self.map[target[0]+i, target[1]+j] = 1

        # check if legal, calc the distance
        flag = self.check()
        if flag == 1:
            return 1000 - dis, 1
        elif flag == -1:
            return -1000, 1
        else:
            return 50 - dis, 0

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size)
        pos_list = []
        size_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break

            

        
        random.shuffle(pos_list)
        for i in range(num):
            px, py = pos_list[i]
            
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    if j > len(size_list) -1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            

        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):
        
    def getstate(self):
        state = []
        for i in range(len(self.mark)+1):
            state.append((self.map == (i+1)).astype(np.float32))
        
        return np.transpose(np.array(state),[1,2,0])
    
    def getmap(self):        
        return np.array(self.map) 

class ENV_M_C_L:
    def __init__(self, size=(5,5), start=(0,0)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.start = start
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, size): 
        self.map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.mark = np.zeros(len(pos))
        self.dis = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            self.map[x:x+dx,y:y+dy] = i + 2
            # for gox in range(dx):
            #     for goy in range(dy):
            #         self.map[x + gox,y + goy] = i + 2

        # self.check()

    '''
        check the state
    '''
    def check(self,index):
        flag = True
        target = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        
        mark = np.zeros_like(self.map)
    
        fa = np.ones([*(self.map.shape), 2])
        fa *= -1
        pq = PQ()
        
        # Q = queue.Queue()
        # to confirm the starting point is legal
        l = True

        if (self.map[self.start[0]:self.start[0]+size[0],self.start[1]:self.start[1]+size[1]] == 1).sum() != 0:
            l = False
        
        if l:
            f = abs(self.start[0]-target[0])+abs(self.start[1]-target[1])
            pq.put((f, self.start))
            # Q.put(self.start)
            dis[self.start[0], self.start[1]] = 0

        # BFS
        while not pq.empty():
            a = pq.get()[1]
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal

            if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                l = False

            if (self.map[x:x+size[0],y:y+size[1]] == 1).sum() != 0:
                l = False

            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    mark[x,y] = 1
                    break
                
                if mark[x,y] == 1:
                    continue

                for i, dx in enumerate(_x):
                    dy = _y[i]
                    # for dy in _y:
                    tx = x + dx
                    ty = y + dy
                    if tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                        # Q.put((tx,ty))
                        pq.put((d+1+abs(tx-target[0])+abs(ty-target[1]),(tx,ty)))
                        fa[tx,ty] = (x,y)
                        dis[tx,ty] = d + 1

                mark[x,y] = 1
        
        if dis[target[0], target[1]] == -1:
            flag = False
            # break

        self.dis[index] = dis[target[0], target[1]]

        if flag:
            return 1

        return 0       
        # return to_check            

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index):
        if index >= len(self.mark) or self.mark[index] == 1:
            return -500, -1
        flag = self.check(index)
        if flag:

            size = self.size[index]
            target = self.pos[index]
            self.mark[index] = 1
            dis = self.dis[index]

            self.map[target[0]:target[0]+size[0],target[1]:target[1]+size[1]] = 1
            
            ff = True
            for i in range(len(self.mark)):
                if self.mark[i] != 1:
                    ff = False
                    break
            if ff:
                return 1000-dis, 1
            
            return 50-dis, 0
            
        else:
            return -1000, 1

        # # check if legal, calc the distance
        # flag = self.check()
        # if flag == 1:
        #     return 1000 - dis, 1
        # elif flag == -1:
        #     return -1000, 1
        # else:
        #     return 50 - dis, 0

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')


    # def Asearch(self):
        

    def randominit(self,num=30):
        import random
        size = self.map_size
        # print(size)
        pos_list = []
        size_list = []

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
        
        random.shuffle(pos_list)
        for i in range(num):
            px, py = pos_list[i]
            
            while True:
                dx = np.random.randint(1,64)
                dy = np.random.randint(1,64)
                if px + dx > size[0] or py + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    if j > len(size_list) -1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            

        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,size_list)
    # def randominit(self,num=30):
    #     import random
    #     size = self.map_size
    #     # print(size)
    #     pos_list = []
    #     size_list = []
        

    #     for i in range(num):
    #         while True:
    #             px = np.random.randint(size[0])
    #             py = np.random.randint(size[1])
    #             flag = True

    #             for _ in pos_list:
    #                 dx,dy = _
    #                 if dx == px or dy == py:
    #                     flag = False
    #                     break
    #             if flag:
    #                 pos_list.append((px,py))
    #                 break


        
    #     random.shuffle(pos_list)
    #     for i in range(num):
    #         px, py = pos_list[i]
    #         ux = self.map_size[0] - px
    #         uy = self.map_size[1] - py

    #         for j in range(num):
    #             if j == i:
    #                 continue

    #             px_, py_ = pos_list[j]
    #             if j > len(size_list) -1:
    #                 dx_, dy_ = 1, 1
    #             else:
    #                 dx_, dy_ = size_list[j] 

    #             lx = max(px, px_)
    #             ly = max(py, py_)
    #             rx = min(px + ux, px_ + dx_) - 1
    #             ry = min(py + uy, py_ + dy_) - 1
    #             if lx <= rx and ly <= ry:
    #                 if lx == px:
    #                     uy = ly - py
    #                 elif ly == py:
    #                     ux = lx - px
    #                 else:
    #                     ux = lx - px
    #                     uy = ly - py

            
    #         dx = np.random.randint(1,ux+1)
    #         dy = np.random.randint(1,uy+1)


            
    #         size_list.append((dx,dy))
                

            

    #     # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
    #     # random.shuffle(list_)
    #     # for i in list_:
    #     #     pos_list.append(i[0])
    #     #     size_list.append(i[1])
    #     t = time.time()
    #     self.setmap(pos_list,size_list)
    #     print(time.time()-t)
            
    #         # for i in range(size[0]):
    #         #     for j in range(size[1]):
                    

    #         # for i in range(size[0]):
    #         #     for j in range(size[1]):
        
    def getstate(self):
        state = []
        for i in range(len(self.mark)+1):
            state.append((self.map == (i+1)).astype(np.float32))
        
        return np.transpose(np.array(state),[1,2,0])
    
    def getmap(self):        
        return np.array(self.map) 

class ENV_scene:
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)


    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size): 
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.target = target
        self.dis = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        # self.check()

    '''
        check the state 
        0 represent not accessible
        1 represent accessible
    '''
    def check(self, index):
        flag = True
        target = self.target[index]
        start = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        Q = queue.Queue()

        # to confirm the starting point is legal
        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[start[0]+i, start[1]+j] != 0 and self.map[start[0]+i, start[1]+j] != index + 2:
                    l = False
                    break
            if not l:
                break
        if l:
            Q.put(start)
            dis[start[0], start[1]] = 0

        # BFS
        while not Q.empty():
            a = Q.get()
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal

            if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                l = False

            for i in range(size[0]):
                if x + i < self.map_size[0]:
                    for j in range(size[1]):
                        if y + j < self.map_size[1]:
                            if self.map[x+i, y+j] != 0 and self.map[x+i, y+j] != index + 2:
                                l = False
                                break
                if not l:
                    break

            # if not l:
            #     print('illegal',x,y)
            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    break

                for dx in _x:
                    for dy in _y:
                        tx = x + dx
                        ty = y + dy
                        ff = True

                        for i in range(size[0]):
                            if tx + i < self.map_size[0]:
                                for j in range(size[1]):
                                    if ty + j < self.map_size[1]:
                                        if self.map[tx+i, ty+j] != 0 and self.map[tx+i, ty+j] != index + 2:
                                            ff = False
                                            break
                            if not ff:
                                break

                        if ff and tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                            Q.put((tx,ty))
                            dis[tx,ty] = d + 1
        
        
        if dis[target[0],target[1]] == -1:
            flag = False

        if flag:
            return 1

        return 0        
        # return to_check

    '''
    `    return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -500

        if direction < 4:
            xx, yy = pos
            x = xx + _x[direction]
            y = yy + _y[direction]
            if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                return -500, -1
            
            flag = True
            if _x[direction] == 1:
                for i in range(size[1]):
                    if self.map[x + size[0] - 1, y+i] != 0:
                        flag = False
            
            if _x[direction] == -1:
                for i in range(size[1]):
                    if self.map[x, y+i] != 0:
                        flag = False
            
            if _y[direction] == 1:
                for i in range(size[0]):
                    if self.map[x+i, y + size[1] - 1] != 0:
                        flag = False

            if _y[direction] == -1:
                for i in range(size[0]):
                    if self.map[x+i, y] != 0:
                        flag = False
                        
            if not flag:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)

                if x == target[0] and y == target[1]:
                    return base + 500, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        if pos == target:
            return -500, -1
        # check if legal, calc the distance
        flag = self.check(index)
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0],yy:yy+size[1]] = 0
            self.map[x:x+size[0],y:y+size[1]] = index + 2
            self.pos[index] = target
            
            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            if ff:
                return base + 10000, 1
            else:
                return base + 500, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):
        
    def getstate_1(self):
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):        
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

class ENV_scene_new_action:
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)


    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size): 
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.target = target
        self.dis = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        # self.check()

    '''
        check the state 
        0 represent not accessible
        1 represent accessible
    '''
    def check(self, index):
        flag = True
        target = self.target[index]
        start = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        Q = queue.Queue()

        # to confirm the starting point is legal
        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[start[0]+i, start[1]+j] != 0 and self.map[start[0]+i, start[1]+j] != index + 2:
                    l = False
                    break
            if not l:
                break
        if l:
            Q.put(start)
            dis[start[0], start[1]] = 0

        # BFS
        while not Q.empty():
            a = Q.get()
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal

            if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                l = False

            for i in range(size[0]):
                if x + i < self.map_size[0]:
                    for j in range(size[1]):
                        if y + j < self.map_size[1]:
                            if self.map[x+i, y+j] != 0 and self.map[x+i, y+j] != index + 2:
                                l = False
                                break
                if not l:
                    break

            # if not l:
            #     print('illegal',x,y)
            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    break

                for dx in _x:
                    for dy in _y:
                        tx = x + dx
                        ty = y + dy
                        ff = True

                        for i in range(size[0]):
                            if tx + i < self.map_size[0]:
                                for j in range(size[1]):
                                    if ty + j < self.map_size[1]:
                                        if self.map[tx+i, ty+j] != 0 and self.map[tx+i, ty+j] != index + 2:
                                            ff = False
                                            break
                            if not ff:
                                break

                        if ff and tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                            Q.put((tx,ty))
                            dis[tx,ty] = d + 1
        
        
        if dis[target[0],target[1]] == -1:
            flag = False

        if flag:
            return 1

        return 0        
        # return to_check

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -500

        if direction < 4:
            xx, yy = pos
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
                if _x[direction] == 1:
                    for i in range(size[1]):
                        if self.map[x + size[0] - 1, y+i] != 0:
                            flag = False
                
                if _x[direction] == -1:
                    for i in range(size[1]):
                        if self.map[x, y+i] != 0:
                            flag = False
                
                if _y[direction] == 1:
                    for i in range(size[0]):
                        if self.map[x+i, y + size[1] - 1] != 0:
                            flag = False

                if _y[direction] == -1:
                    for i in range(size[0]):
                        if self.map[x+i, y] != 0:
                            flag = False
                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)

                if x == target[0] and y == target[1]:
                    return base + 500, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target:
            return -500, -1
        # check if legal, calc the distance
        flag = self.check(index)
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0],yy:yy+size[1]] = 0
            self.map[x:x+size[0],y:y+size[1]] = index + 2
            self.pos[index] = target
            
            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            if ff:
                return base + 10000, 1
            else:
                return base + 500, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break


        self.setmap(pos_list,target_list,size_list)

            
    def getstate_1(self):
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):        
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

class ENV_scene_new_action_pre_state:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size): 
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.target = target
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos)+1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod 
        
        return total

    '''
        check the state 
        0 represent not accessible
        1 represent accessible
    '''
    def check(self, index):
        flag = True
        target = self.target[index]
        start = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        Q = queue.Queue()

        # to confirm the starting point is legal
        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[start[0]+i, start[1]+j] != 0 and self.map[start[0]+i, start[1]+j] != index + 2:
                    l = False
                    break
            if not l:
                break
        if l:
            Q.put(start)
            dis[start[0], start[1]] = 0
        
        # BFS
        while not Q.empty():
            a = Q.get()
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal

            if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                l = False

            for i in range(size[0]):
                if x + i < self.map_size[0]:
                    for j in range(size[1]):
                        if y + j < self.map_size[1]:
                            if self.map[x+i, y+j] != 0 and self.map[x+i, y+j] != index + 2:
                                l = False
                                break
                if not l:
                    break

            # if not l:
            #     print('illegal',x,y)
            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    break

                for dx in _x:
                    for dy in _y:
                        tx = x + dx
                        ty = y + dy
                        ff = True

                        for i in range(size[0]):
                            if tx + i < self.map_size[0]:
                                for j in range(size[1]):
                                    if ty + j < self.map_size[1]:
                                        if self.map[tx+i, ty+j] != 0 and self.map[tx+i, ty+j] != index + 2:
                                            ff = False
                                            break
                            if not ff:
                                break

                        if ff and tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                            Q.put((tx,ty))
                            dis[tx,ty] = d + 1
        
        route = []
        
        if dis[target[0],target[1]] == -1:
            flag = False

        if flag:
            # print('start',start,'target',target)
            
            # for r_ in dis:
            #     print(r_)
            # print()

            cur_p = target
            cur_d = dis[target[0],target[1]]
            
            while True:
                route.append(cur_p)
                
                # print(cur_d, cur_p)
                if cur_d == 0:
                    break

                x,y = cur_p
                for dx in _x:
                    tx = x + dx
                    if tx >= 0 and tx < self.map_size[0]:
                        for dy in _y:
                            ty = y + dy
                            if ty >= 0 and ty < self.map_size[1]:
                                if dis[tx,ty] != -1 and dis[tx,ty] < cur_d:
                                    cur_d = dis[tx,ty]
                                    cur_p = (tx,ty)
            return 1, route
                        
        return 0, route
        # return to_check

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -500

        if direction < 4:
            xx, yy = pos
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
                if _x[direction] == 1:
                    for i in range(size[1]):
                        if self.map[x + size[0] - 1, y+i] != 0:
                            flag = False
                
                if _x[direction] == -1:
                    for i in range(size[1]):
                        if self.map[x, y+i] != 0:
                            flag = False
                
                if _y[direction] == 1:
                    for i in range(size[0]):
                        if self.map[x+i, y + size[1] - 1] != 0:
                            flag = False

                if _y[direction] == -1:
                    for i in range(size[0]):
                        if self.map[x+i, y] != 0:
                            flag = False

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1]:
                    return base + 500, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        self.route = route
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            self.map[x:x+size[0], y:y+size[1]] = index + 2
            self.pos[index] = target
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -600, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            if ff:
                return base + 10000, 1
            else:
                return base + 500, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break


        self.setmap(pos_list,target_list,size_list)
            
    def getstate_1(self):
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):        
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

class ENV_scene_new_action_pre_state_penalty:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        
        

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size): 
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.target = target
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos)+1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod 
        
        return total

    '''
        check the state 
        0 represent not accessible
        1 represent accessible
    '''
    def check(self, index):
        flag = True
        target = self.target[index]
        start = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        Q = queue.Queue()

        # to confirm the starting point is legal
        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[start[0]+i, start[1]+j] != 0 and self.map[start[0]+i, start[1]+j] != index + 2:
                    l = False
                    break
            if not l:
                break
        if l:
            Q.put(start)
            dis[start[0], start[1]] = 0
        
        # BFS
        while not Q.empty():
            a = Q.get()
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal

            if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                l = False

            for i in range(size[0]):
                if x + i < self.map_size[0]:
                    for j in range(size[1]):
                        if y + j < self.map_size[1]:
                            if self.map[x+i, y+j] != 0 and self.map[x+i, y+j] != index + 2:
                                l = False
                                break
                if not l:
                    break

            # if not l:
            #     print('illegal',x,y)
            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    break

                for dx in _x:
                    for dy in _y:
                        tx = x + dx
                        ty = y + dy
                        ff = True

                        for i in range(size[0]):
                            if tx + i < self.map_size[0]:
                                for j in range(size[1]):
                                    if ty + j < self.map_size[1]:
                                        if self.map[tx+i, ty+j] != 0 and self.map[tx+i, ty+j] != index + 2:
                                            ff = False
                                            break
                            if not ff:
                                break

                        if ff and tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                            Q.put((tx,ty))
                            dis[tx,ty] = d + 1
        
        route = []
        
        if dis[target[0],target[1]] == -1:
            flag = False

        if flag:
            # print('start',start,'target',target)
            
            # for r_ in dis:
            #     print(r_)
            # print()

            cur_p = target
            cur_d = dis[target[0],target[1]]
            
            while True:
                route.append(cur_p)
                
                # print(cur_d, cur_p)
                if cur_d == 0:
                    break

                x,y = cur_p
                for dx in _x:
                    tx = x + dx
                    if tx >= 0 and tx < self.map_size[0]:
                        for dy in _y:
                            ty = y + dy
                            if ty >= 0 and ty < self.map_size[1]:
                                if dis[tx,ty] != -1 and dis[tx,ty] < cur_d:
                                    cur_d = dis[tx,ty]
                                    cur_p = (tx,ty)
            return 1, route
                        
        return 0, route
        # return to_check

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
                if _x[direction] == 1:
                    for i in range(size[1]):
                        if self.map[x + size[0] - 1, y+i] != 0:
                            flag = False
                
                if _x[direction] == -1:
                    for i in range(size[1]):
                        if self.map[x, y+i] != 0:
                            flag = False
                
                if _y[direction] == 1:
                    for i in range(size[0]):
                        if self.map[x+i, y + size[1] - 1] != 0:
                            flag = False

                if _y[direction] == -1:
                    for i in range(size[0]):
                        if self.map[x+i, y] != 0:
                            flag = False

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1]:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        # return base + 500, 0
                        return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        self.route = route
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            self.map[x:x+size[0], y:y+size[1]] = index + 2
            self.pos[index] = target
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    # return base + 500, 0
                    return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True
                hh = [-1,0,1]
                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                    
                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break

                        if flag == False:
                            break

                    if flag == False:
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                    
                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break

                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            mx = size[0]
            my = size[1]
            for j in range(num):
                if j == i:
                    continue
                tmp = pos_list[j]
                if tmp[0] >= px and tmp[1] >= py:
                    mx = min([mx,tmp[0]-px])
                    my = min([my,tmp[1]-px])
                tmp = target_list[j]
                if tmp[0] >= tx and tmp[1] >= ty:
                    mx = min([mx,tmp[0]-tx])
                    my = min([my,tmp[1]-ty])

            while True:
                dx = np.random.randint(2,mx)
                dy = np.random.randint(2,my)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        # self.finished = np.zeros(num)

        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True
                hh = [-1,0,1]
                for _ in pos_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx - px) + abs(dy - py) < 5:
                        flag = False
                    
                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break

                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False

                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break

                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        # print(pos_list)
        # print(target_list)

        # for i in range(num):
        #     print(i)
        #     px, py = pos_list[i]
        #     tx, ty = target_list[i]
        #     mx = (size[0],size[1])
        #     my = (size[0],size[1])

        #     for j in range(num):
        #         if j == i:
        #             continue
                
        #         tmp = pos_list[j]
        #         tmps = []
        #         tmps.append(tmp)
        #         tmps.append((tmp[0],tmp[1]+1))
        #         tmps.append((tmp[0]+1,tmp[1]))
        #         tmps.append((tmp[0]+1,tmp[1]+1))
        #         if tmp[0] >= px and tmp[1] >= py:
        #             if tmp[0] - px < mx[0]:
        #                 mx = (tmp[0]-px, tmp[1]-py)
        #             if tmp[1] - py < my[1]:
        #                 my = (tmp[0]-px, tmp[1]-py)
                


        #         tmp = target_list[j]
        #         tmps = []
        #         tmps.append(tmp)
        #         tmps.append((tmp[0],tmp[1]+1))
        #         tmps.append((tmp[0]+1,tmp[1]))
        #         tmps.append((tmp[0]+1,tmp[1]+1))
        #         if tmp[0] >= tx and tmp[1] >= ty:
        #             if tmp[0] - tx < mx[0]:
        #                 mx = (tmp[0]-tx, tmp[1]-ty)
        #             if tmp[1] - ty < my[1]:
        #                 my = (tmp[0]-tx, tmp[1]-ty)
            
        #     # if my[1] > mx[0]:
        #     #     mx = my
        #     print(mx,my)
        #     if mx == my:
        #         if mx[0] > mx[1]:
        #             mx = (mx[0], min([size[1]-py,size[1]-ty]))
        #         if mx[0] < mx[1]:
        #             mx = (min([size[0]-px, size[1]-tx]),mx[1])
        #     else:
        #         mx = (my[0],mx[1])

        #     print(mx)
        #     cnt = 0
        #     while True:
        #         dx = np.random.randint(2,mx[0]+1)
        #         dy = np.random.randint(2,mx[1]+1)
        #         # if dx == 2 and dy == 2:
        #             # print(dx, dy)
        #         if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
        #             # print(px,py,dx,dy)
        #             continue

        #         flag = True

        #         for j in range(num):
        #             if j == i:
        #                 continue

        #             px_, py_ = pos_list[j]
        #             tx_, ty_ = target_list[j]
        #             if j > len(size_list) - 1:
        #                 dx_, dy_ = 1, 1
        #             else:
        #                 dx_, dy_ = size_list[j]

        #             lx = max(px, px_)
        #             ly = max(py, py_)
        #             rx = min(px + dx, px_ + dx_) - 1
        #             ry = min(py + dy, py_ + dy_) - 1
        #             if lx <= rx and ly <= ry:
        #                 flag = False
        #                 break

        #             lx = max(tx, tx_)
        #             ly = max(ty, ty_)
        #             rx = min(tx + dx, tx_ + dx_) - 1
        #             ry = min(ty + dy, ty_ + dy_) - 1
        #             if lx <= rx and ly <= ry:
        #                 flag = False
        #                 break

        #         if flag:
        #             size_list.append((dx,dy))
        #             break
                
        #         cnt += 1
            
        #     print(dx,dy)
        #     # print(cnt)
        for i in range(num):
            # print(i)
            px, py = pos_list[i]
            tx, ty = target_list[i]
            mx = (size[0],size[1])
            my = (size[0],size[1])

            for j in range(num):
                if j == i:
                    continue
                
                
            cnt = 0
            while True:
                dx = np.random.randint(2,11)
                dy = np.random.randint(2,11)
                # if dx == 2 and dy == 2:
                    # print(dx, dy)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 2, 2
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break
                
                cnt += 1
            
            # print(dx,dy)
            # print(cnt)

        self.setmap(pos_list,target_list,size_list)
            
    def getstate_1(self):
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.size)

class ENV_scene_new_action_pre_state_penalty_conflict:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        
        

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size): 
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.target = target
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()

    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos)+1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod 
        
        return total

    '''
        check the state 
        0 represent not accessible
        1 represent accessible
    '''
    def check(self, index):
        flag = True
        target = self.target[index]
        start = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        Q = queue.Queue()

        # to confirm the starting point is legal
        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[start[0]+i, start[1]+j] != 0 and self.map[start[0]+i, start[1]+j] != index + 2:
                    l = False
                    break
            if not l:
                break
        if l:
            Q.put(start)
            dis[start[0], start[1]] = 0
        
        # BFS
        while not Q.empty():
            a = Q.get()
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal

            if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                l = False

            for i in range(size[0]):
                if x + i < self.map_size[0]:
                    for j in range(size[1]):
                        if y + j < self.map_size[1]:
                            if self.map[x+i, y+j] != 0 and self.map[x+i, y+j] != index + 2:
                                l = False
                                break
                if not l:
                    break

            # if not l:
            #     print('illegal',x,y)
            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    break

                for dx in _x:
                    for dy in _y:
                        tx = x + dx
                        ty = y + dy
                        ff = True

                        for i in range(size[0]):
                            if tx + i < self.map_size[0]:
                                for j in range(size[1]):
                                    if ty + j < self.map_size[1]:
                                        if self.map[tx+i, ty+j] != 0 and self.map[tx+i, ty+j] != index + 2:
                                            ff = False
                                            break
                            if not ff:
                                break

                        if ff and tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                            Q.put((tx,ty))
                            dis[tx,ty] = d + 1
        
        route = []
        
        if dis[target[0],target[1]] == -1:
            flag = False

        if flag:
            # print('start',start,'target',target)
            
            # for r_ in dis:
            #     print(r_)
            # print()

            cur_p = target
            cur_d = dis[target[0],target[1]]
            
            while True:
                route.append(cur_p)
                
                # print(cur_d, cur_p)
                if cur_d == 0:
                    break

                x,y = cur_p
                for dx in _x:
                    tx = x + dx
                    if tx >= 0 and tx < self.map_size[0]:
                        for dy in _y:
                            ty = y + dy
                            if ty >= 0 and ty < self.map_size[1]:
                                if dis[tx,ty] != -1 and dis[tx,ty] < cur_d:
                                    cur_d = dis[tx,ty]
                                    cur_p = (tx,ty)
            return 1, route
                        
        return 0, route
        # return to_check

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
                if _x[direction] == 1:
                    for i in range(size[1]):
                        if self.map[x + size[0] - 1, y+i] != 0:
                            flag = False
                
                if _x[direction] == -1:
                    for i in range(size[1]):
                        if self.map[x, y+i] != 0:
                            flag = False
                
                if _y[direction] == 1:
                    for i in range(size[0]):
                        if self.map[x+i, y + size[1] - 1] != 0:
                            flag = False

                if _y[direction] == -1:
                    for i in range(size[0]):
                        if self.map[x+i, y] != 0:
                            flag = False

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1]:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        self.route = route
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            self.map[x:x+size[0], y:y+size[1]] = index + 2
            self.pos[index] = target
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        # self.finished = np.zeros(num)

        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break


        self.setmap(pos_list,target_list,size_list)
            
    def getstate_1(self):
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.size)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):
        num = len(self.finished)
        res = np.zeros([num,num,2])
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res


class ENV_scene_new_action_pre_state_penalty_conflict_easy:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        
        

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size): 
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = pos
        self.size = size
        self.target = target
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()

    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos)+1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod 
        
        return total

    '''
        check the state 
        0 represent not accessible
        1 represent accessible
    '''
    def check(self, index):
        flag = True
        target = self.target[index]
        start = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        Q = queue.Queue()

        # to confirm the starting point is legal
        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[start[0]+i, start[1]+j] != 0 and self.map[start[0]+i, start[1]+j] != index + 2:
                    l = False
                    break
            if not l:
                break
        if l:
            Q.put(start)
            dis[start[0], start[1]] = 0
        
        # BFS
        while not Q.empty():
            a = Q.get()
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal

            if x + size[0] - 1 >= self.map_size[0] or y + size[1] - 1 >= self.map_size[1]:
                l = False

            for i in range(size[0]):
                if x + i < self.map_size[0]:
                    for j in range(size[1]):
                        if y + j < self.map_size[1]:
                            if self.map[x+i, y+j] != 0 and self.map[x+i, y+j] != index + 2:
                                l = False
                                break
                if not l:
                    break

            # if not l:
            #     print('illegal',x,y)
            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    break

                for dx in _x:
                    for dy in _y:
                        tx = x + dx
                        ty = y + dy
                        ff = True

                        for i in range(size[0]):
                            if tx + i < self.map_size[0]:
                                for j in range(size[1]):
                                    if ty + j < self.map_size[1]:
                                        if self.map[tx+i, ty+j] != 0 and self.map[tx+i, ty+j] != index + 2:
                                            ff = False
                                            break
                            if not ff:
                                break

                        if ff and tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                            Q.put((tx,ty))
                            dis[tx,ty] = d + 1
        
        route = []
        
        if dis[target[0],target[1]] == -1:
            flag = False

        if flag:
            # print('start',start,'target',target)
            
            # for r_ in dis:
            #     print(r_)
            # print()

            cur_p = target
            cur_d = dis[target[0],target[1]]
            
            while True:
                route.append(cur_p)
                
                # print(cur_d, cur_p)
                if cur_d == 0:
                    break

                x,y = cur_p
                for dx in _x:
                    tx = x + dx
                    if tx >= 0 and tx < self.map_size[0]:
                        for dy in _y:
                            ty = y + dy
                            if ty >= 0 and ty < self.map_size[1]:
                                if dis[tx,ty] != -1 and dis[tx,ty] < cur_d:
                                    cur_d = dis[tx,ty]
                                    cur_p = (tx,ty)
            return 1, route
                        
        return 0, route
        # return to_check

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
                if _x[direction] == 1:
                    for i in range(size[1]):
                        if self.map[x + size[0] - 1, y+i] != 0:
                            flag = False
                
                if _x[direction] == -1:
                    for i in range(size[1]):
                        if self.map[x, y+i] != 0:
                            flag = False
                
                if _y[direction] == 1:
                    for i in range(size[0]):
                        if self.map[x+i, y + size[1] - 1] != 0:
                            flag = False

                if _y[direction] == -1:
                    for i in range(size[0]):
                        if self.map[x+i, y] != 0:
                            flag = False

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1]:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        self.route = route
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            self.map[x:x+size[0], y:y+size[1]] = index + 2
            self.pos[index] = target
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        # self.finished = np.zeros(num)

        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break


        self.setmap(pos_list,target_list,size_list)
            
    def getstate_1(self):
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.size)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):
        num = len(self.finished)
        res = np.zeros([num,num,2])
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res


class ENV_scene_new_action_pre_state_penalty_conflict_heuristic:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size): 
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(pos)
        self.size = deepcopy(size)
        self.target = deepcopy(target)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()

    def hash(self):
        mod = 1000000007
        num = len(self.pos)+1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod 
        
        return total

    '''
        check the state 
        0 represent not accessible
        1 represent accessible
    '''
    def check(self, index):
        flag = True
        target = self.target[index]
        start = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        # Q = queue.Queue()
        Q = PQ()        
        fn = {}
        fa = {}

        # to confirm the starting point is legal

        self.map[start[0]:start[0]+size[0],start[1]:start[1]+size[1]] = 0
        sum_ = np.zeros_like(self.map)
        for i in range(1,sum_.shape[0]):
            sum_[i,0] = sum_[i-1, 0] + self.map[i,0]

        for i in range(1,sum_.shape[1]):
            sum_[0,i] = sum_[0, i-1] + self.map[0,i]

        for i in range(1,sum_.shape[0]):
            for j in range(1,sum_.shape[1]):
                sum_[i,j] = sum_[i,j-1] + sum_[i-1,j] - sum_[i-1,j-1] + self.map[i,j]
        
        # for i in range(sum_.shape[0]):
        #     print(sum_[i])

        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[start[0]+i, start[1]+j] != 0:
                    l = False
                    break
            if not l:
                break
        if l:
            f = 0 + abs(start[0] - target[0]) + abs(start[1] - target[1])
            Q.put((f,0,start))
            dis[start[0], start[1]] = f
        
        # print('====================================================')

        # Heuristic
        while not Q.empty():
            _, d, a = Q.get()
            x, y = a
            # print('x',x,'y',y,'dis',d,'f',_)
            if x == target[0] and y == target[1]:
                break

            for i,dx in enumerate(_x):
                dy = _y[i]
                tx = x + dx
                ty = y + dy
                # print('tx,ty',tx,ty,tx+size[0]-1,ty+size[1]-1)
                if tx + size[0] <= self.map_size[0] and tx >= 0 and ty + size[1] <= self.map_size[1] and ty >= 0:
                    aa = sum_[tx+size[0]-1, ty+size[1]-1]
                    ab = 0
                    ba = 0
                    bb = 0
                    
                    if tx != 0:
                        ba = sum_[tx-1, ty+size[1]-1]
                    
                    if ty != 0:
                        ab = sum_[tx+size[0]-1, ty-1]
                    
                    if tx != 0 and ty != 0:
                        bb = sum_[tx-1, ty-1]
                    
                    res = aa - ab - ba + bb
                    # print('aa,ab,ba,bb,res',aa,ab,ba,bb,res)
                    if res == 0: # If no collision
                        f_ = d + 1 + abs(tx-target[0]) + abs(ty-target[1])
                        # print('dis',dis[tx,ty],'f_',f_)
                        if dis[tx, ty] == -1 or dis[tx,ty] > f_:
                            dis[tx,ty] = f_
                            Q.put((f_, d+1, (tx, ty)))
                            fa[(tx, ty)] = (x, y)
                                



                    


        self.map[start[0]:start[0]+size[0],start[1]:start[1]+size[1]] = index + 2
        route = []
        # print('=============================\n\n')
        
        if dis[target[0],target[1]] == -1:
            flag = False

        if flag:
            # print('start',start,'target',target)
            
            # for r_ in dis:
            #     print(r_)
            # print()

            cur_p = target
            # route.append(cur_p)

            while cur_p != start:
                route.append(cur_p)
                cur_p = fa[cur_p]
                
                
            return 1, route
                        
        return 0, route
        # return to_check

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
                if _x[direction] == 1:
                    for i in range(size[1]):
                        if self.map[x + size[0] - 1, y+i] != 0:
                            flag = False
                
                if _x[direction] == -1:
                    for i in range(size[1]):
                        if self.map[x, y+i] != 0:
                            flag = False
                
                if _y[direction] == 1:
                    for i in range(size[0]):
                        if self.map[x+i, y + size[1] - 1] != 0:
                            flag = False

                if _y[direction] == -1:
                    for i in range(size[0]):
                        if self.map[x+i, y] != 0:
                            flag = False

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1]:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        self.route = route
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            self.map[x:x+size[0], y:y+size[1]] = index + 2
            self.pos[index] = target
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        # self.finished = np.zeros(num)

        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break


        self.setmap(pos_list,target_list,size_list)
            
    def getstate_1(self):
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.size)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):
        num = len(self.finished)
        res = np.zeros([num,num,2])
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res

class ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5),max_num = 5):
        self.map_size = size
        self.map = np.zeros(self.map_size,dtype=np.int32)
        self.target_map = np.zeros(self.map_size,dtype=np.int32)
        self.route = []
        self.max_num = max_num
        


    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size): 
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(pos)
        self.size = deepcopy(size)
        self.target = deepcopy(target)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()

    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos)+1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod 
        
        return total


    def check(self, index):
        # flag = True
        tx,ty = self.target[index]
        sx,sy = self.pos[index]
        h,w = self.size[index]
        n,m = self.map_size
        route = []
        lx = np.zeros([200],dtype=np.int32)
        ly = np.zeros([200],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)

        search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i]))

        return flag[0], route
        # return to_check

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1
        if self.size[index][0] == 0:
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
                if _x[direction] == 1:
                    for i in range(size[1]):
                        if self.map[x + size[0] - 1, y+i] != 0:
                            flag = False
                
                if _x[direction] == -1:
                    for i in range(size[1]):
                        if self.map[x, y+i] != 0:
                            flag = False
                
                if _y[direction] == 1:
                    for i in range(size[0]):
                        if self.map[x+i, y + size[1] - 1] != 0:
                            flag = False

                if _y[direction] == -1:
                    for i in range(size[0]):
                        if self.map[x+i, y] != 0:
                            flag = False

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, -2

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1]:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        self.route = route
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            self.map[x:x+size[0], y:y+size[1]] = index + 2
            self.pos[index] = target
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        # self.finished = np.zeros(num)

        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                if px == 0 and py == 0:
                    continue

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
        
        sub_num = self.max_num - num
        # print(self.max_num, num)
        for _ in range(sub_num):
            pos_list.append((0,0))
        # print(len(pos_list))
        random.shuffle(pos_list)
        pos_list = np.array(pos_list)
        tmp_target_list = np.zeros_like(pos_list)
        random.shuffle(target_list)
        _ = 0
        for target in target_list:
            while pos_list[_][0] == pos_list[_][1] and pos_list[_][0] == 0:
                _ += 1
            tmp_target_list[_] = target
            _ += 1

        target_list = tmp_target_list
        # print(pos_list)
        # print(target_list)

        for i in range(self.max_num):
            px, py = pos_list[i]
            tx, ty = target_list[i]

            if px == py and py == 0:
                size_list.append((0,0))
                continue

            while True:
                dx = np.random.randint(1,size[0]-10)
                dy = np.random.randint(1,size[1]-10)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(self.max_num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]

                    if px_ == py_ and py_ == 0:
                        # size_list.append((0,0))
                        continue

                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break
        pos_list = [(pos[0],pos[1]) for pos in pos_list]
        target_list = [(target[0],target[1]) for target in target_list]

        self.setmap(pos_list,target_list,size_list)
            
    def getstate_1(self):
        
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getstate_3(self, shape=[64,64]):
        state = []
        initial = np.array(self.map).astype(np.float32)
        target = np.array(self.target_map).astype(np.float32)
        initial_a = np.zeros(shape)
        target_a = np.zeros(shape)
        size = initial.shape
        for i in range(shape[0]):
            x = int(i*size[0]/shape[0])
            for j in range(shape[1]):
                y = int(i*size[1]/shape[1])
                initial_a[i,j] = initial[x,y]
        
        state.append(initial_a)
        state.append(target_a)
        return np.transpose(np.array(state),[1,2,0])
        

    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.size)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):
        num = len(self.finished)
        res = np.zeros([num,num,2])
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res

class ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        self.bin = 12
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size, cstate, tstate):
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(pos)
        self.size = deepcopy(size)
        self.target = deepcopy(target)
        self.cstate = deepcopy(cstate).astype(np.int)
        self.tstate = deepcopy(tstate).astype(np.int)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos) + 1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod
        
        return total
    

    def rotate(self, points, size, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        points[:,0] += -0.5 * size[0] + 0.5
        points[:,1] += -0.5 * size[1] + 0.5
        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += 0.5 * size[0] - 0.5 + eps
        points[1,:] += 0.5 * size[1] - 0.5 + eps
        points = np.round(points).astype(np.int)
        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        return points.transpose(), np.array([[xmin, ymin], [xmax, ymax]])

    '''
        check the state
        0 represent not accessible
        1 represent accessible
    '''

    def check(self, index):
        flag = True
        target = self.target[index]
        start = self.pos[index]
        size = self.size[index]
        cstate = self.cstate[index]
        # print(cstate)
        tstate = self.tstate[index]
        dis = np.ones([*self.map.shape, self.bin])
        dis *= -1                   # when it starts, all distance was initialized as -1 
        # Q = queue.Queue()
        Q = PQ()
        fn = {}
        fa = {}

        opoints = []
        for i in range(size[0]):
            for j in range(size[1]):
                opoints.append((i,j))
        
        radian = pi * cstate / self.bin
        points, bbox = self.rotate(opoints, size, radian)


        # to confirm the starting point is legal

        # self.map[start[0]:start[0]+size[0],start[1]:start[1]+size[1]] = 0
        for p in points:
            x, y = p
            x += start[0]
            y += start[1]
            self.map[x, y] = 0

        sum_ = np.zeros_like(self.map)
        for i in range(1,sum_.shape[0]):
            sum_[i,0] = sum_[i-1, 0] + self.map[i,0]

        for i in range(1,sum_.shape[1]):
            sum_[0,i] = sum_[0, i-1] + self.map[0,i]

        for i in range(1,sum_.shape[0]):
            for j in range(1,sum_.shape[1]):
                sum_[i,j] = sum_[i,j-1] + sum_[i-1,j] - sum_[i-1,j-1] + self.map[i,j]
        
        
        # for i in range(sum_.shape[0]):
        #     print(sum_[i])

        l = True
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         if self.map[start[0]+i, start[1]+j] != 0:
        #             l = False
        #             break
        #     if not l:
        #         break

        if l:
            f = 0 + abs(start[0] - target[0]) + abs(start[1] - target[1])
            Q.put((f, 0, start, cstate))
            dis[start[0], start[1], cstate] = f
            fa[(start[0], start[1], cstate)] = (start[0], start[1], cstate)
            x, y = start

            for iii in range(1, self.bin):
                s = (cstate + iii) % self.bin
                if dis[x,y,s] != -1:
                    continue

                ra = pi * s / self.bin
                ps, bx = self.rotate(opoints, size, ra)

                if x+bx[1, 0] < self.map_size[0] and x+bx[0, 0] >= 0 and y+bx[1, 1] < self.map_size[1] and y+bx[0, 1] >= 0:
                    aa = sum_[x+bx[1,0], y+bx[1,1]]
                    ab = 0
                    ba = 0
                    bb = 0

                    if x+bx[0,0] != 0:
                        ba = sum_[x+bx[0, 0] - 1, y+bx[1, 1]]
                    
                    if y+bx[0,1] != 0:
                        ab = sum_[x+bx[1, 0], y+bx[0, 1] - 1]
                    
                    if x+bx[0,0] != 0 and y+bx[0, 1] != 0:
                        bb = sum_[x+bx[0, 0]-1, y+bx[0, 1]-1]

                    res = aa - ab - ba + bb

                    push_flag = False
                    if res == 0: # If no collision
                        push_flag = True
                    else:
                        push_flag = True
                        for p in ps:
                            xx, yy = p
                            xx += x
                            yy += y
                            if self.map[xx,yy] != 0:
                                push_flag = False
                                break

                    if push_flag:
                        dis[x,y,s] = f
                        Q.put((f, 0, (x, y), s))
                        fa[(x, y, s)] = (x, y, cstate)
                    else:
                        break

                else:
                    break

            for iii in range(1, self.bin):
                s = (cstate - iii) % self.bin
                if dis[x,y,s] != -1:
                    continue

                ra = pi * s / self.bin
                ps, bx = self.rotate(opoints, size, ra)

                if x+bx[1, 0] < self.map_size[0] and x+bx[0, 0] >= 0 and y+bx[1, 1] < self.map_size[1] and y+bx[0, 1] >= 0:
                    aa = sum_[x+bx[1,0], y+bx[1,1]]
                    ab = 0
                    ba = 0
                    bb = 0

                    if x+bx[0,0] != 0:
                        ba = sum_[x+bx[0, 0] - 1, y+bx[1, 1]]
                    
                    if y+bx[0,1] != 0:
                        ab = sum_[x+bx[1, 0], y+bx[0, 1] - 1]
                    
                    if x+bx[0,0] != 0 and y+bx[0, 1] != 0:
                        bb = sum_[x+bx[0, 0]-1, y+bx[0, 1]-1]

                    res = aa - ab - ba + bb

                    push_flag = False
                    if res == 0: # If no collision
                        push_flag = True
                    else:
                        push_flag = True
                        for p in ps:
                            xx, yy = p
                            xx += x
                            yy += y
                            if self.map[xx,yy] != 0:
                                push_flag = False
                                break

                    if push_flag:
                        dis[x,y,s] = f
                        Q.put((f, 0, (x, y), s))
                        fa[(x, y, s)] = (x, y, cstate)
                    else:
                        break

                else:
                    break


            
        
        # print('====================================================')

        # Heuristic
        while not Q.empty():
            _, d, a, state = Q.get()
            x, y = a
            # print('x',x,'y',y,'dis',d,'f',_)
            if x == target[0] and y == target[1] and state == tstate:
                break
            
            ra = pi * state / self.bin
            ps, bx = self.rotate(opoints, size, ra)

            for i,dx in enumerate(_x):
                dy = _y[i]
                tx = x + dx
                ty = y + dy
                # print('tx,ty',tx,ty,tx+size[0]-1,ty+size[1]-1)
                if tx+bx[1, 0] < self.map_size[0] and tx+bx[0, 0] >= 0 and ty+bx[1, 1] < self.map_size[1] and ty+bx[0, 1] >= 0:
                    aa = sum_[tx+bx[1,0], ty+bx[1,1]]
                    ab = 0
                    ba = 0
                    bb = 0

                    if tx+bx[0,0] != 0:
                        ba = sum_[tx+bx[0, 0] - 1, ty+bx[1, 1]]
                    
                    if ty+bx[0,1] != 0:
                        ab = sum_[tx+bx[1, 0], ty+bx[0, 1] - 1]
                    
                    if tx+bx[0,0] != 0 and ty+bx[0, 1] != 0:
                        bb = sum_[tx+bx[0, 0]-1, ty+bx[0, 1]-1]
                    
                    res = aa - ab - ba + bb
                    # print('aa,ab,ba,bb,res',aa,ab,ba,bb,res)
                    push_flag = False
                    if res == 0: # If no collision
                        push_flag = True
                    else:
                        push_flag = True
                        for p in ps:
                            xx, yy = p
                            xx += tx
                            yy += ty
                            if self.map[xx,yy] != 0:
                                push_flag = False
                                break
                        
                    if push_flag:
                        f_ = d + 1 + abs(tx - target[0]) + abs(ty - target[1])
                        # print('dis',dis[tx,ty],'f_',f_)
                        if dis[tx, ty, state] == -1 or dis[tx, ty, state] > f_:
                            dis[tx, ty, state] = f_
                            Q.put((f_, d + 1, (tx, ty), state))
                            fa[(tx, ty, state)] = (x, y, state)

                            for iii in range(1, self.bin):
                                s = (state + iii) % self.bin
                                if dis[tx,ty,s] != -1:
                                    if dis[tx,ty,s] > f_:
                                        dis[tx,ty,s] = f_
                                        Q.put((f_, d + 1, (tx, ty), s))
                                        fa[(tx, ty, s)] = (x, y, state)
                                    continue

                                ra = pi * s / self.bin
                                ps, bx = self.rotate(opoints, size, ra)

                                if tx+bx[1, 0] < self.map_size[0] and tx+bx[0, 0] >= 0 and ty+bx[1, 1] < self.map_size[1] and ty+bx[0, 1] >= 0:
                                    aa = sum_[tx+bx[1,0], ty+bx[1,1]]
                                    ab = 0
                                    ba = 0
                                    bb = 0

                                    if tx+bx[0,0] != 0:
                                        ba = sum_[tx+bx[0, 0] - 1, ty+bx[1, 1]]
                                    
                                    if ty+bx[0,1] != 0:
                                        ab = sum_[tx+bx[1, 0], ty+bx[0, 1] - 1]
                                    
                                    if tx+bx[0,0] != 0 and ty+bx[0, 1] != 0:
                                        bb = sum_[tx+bx[0, 0]-1, ty+bx[0, 1]-1]

                                    res = aa - ab - ba + bb

                                    push_flag = False
                                    if res == 0: # If no collision
                                        push_flag = True
                                    else:
                                        push_flag = True
                                        for p in ps:
                                            xx, yy = p
                                            xx += tx
                                            yy += ty
                                            if self.map[xx,yy] != 0:
                                                push_flag = False
                                                break
                        
                                    if push_flag:
                                        dis[tx,ty,s] = f_
                                        Q.put((f_, d + 1, (tx, ty), s))
                                        fa[(tx, ty, s)] = (x, y, state)
                                    else:
                                        break

                                else:
                                    break

                            for iii in range(1, self.bin):
                                s = (state - iii + self.bin) % self.bin
                                if dis[tx,ty,s] != -1:
                                    if dis[tx,ty,s] > f_:
                                        dis[tx,ty,s] = f_
                                        Q.put((f_, d + 1, (tx, ty), s))
                                        fa[(tx, ty, s)] = (x, y, state)
                                    continue

                                ra = pi * s / self.bin
                                ps, bx = self.rotate(opoints, size, ra)

                                if tx+bx[1, 0] < self.map_size[0] and tx+bx[0, 0] >= 0 and ty+bx[1, 1] < self.map_size[1] and ty+bx[0, 1] >= 0:
                                    aa = sum_[tx+bx[1,0], ty+bx[1,1]]
                                    ab = 0
                                    ba = 0
                                    bb = 0

                                    if tx+bx[0,0] != 0:
                                        ba = sum_[tx+bx[0, 0] - 1, ty+bx[1, 1]]
                                    
                                    if ty+bx[0,1] != 0:
                                        ab = sum_[tx+bx[1, 0], ty+bx[0, 1] - 1]
                                    
                                    if tx+bx[0,0] != 0 and ty+bx[0, 1] != 0:
                                        bb = sum_[tx+bx[0, 0]-1, ty+bx[0, 1]-1]

                                    res = aa - ab - ba + bb

                                    push_flag = False
                                    if res == 0: # If no collision
                                        push_flag = True
                                    else:
                                        push_flag = True
                                        for p in ps:
                                            xx, yy = p
                                            xx += tx
                                            yy += ty
                                            if self.map[xx,yy] != 0:
                                                push_flag = False
                                                break
                        
                                    if push_flag:
                                        dis[tx,ty,s] = f_
                                        Q.put((f_, d + 1, (tx, ty), s))
                                        fa[(tx, ty, s)] = (x, y, state)
                                    else:
                                        break

                                else:
                                    break  

        for p in points:
            x, y = p
            x += start[0]
            y += start[1]
            self.map[x, y] = index + 2
            
        # self.map[start[0]:start[0]+size[0],start[1]:start[1]+size[1]] = index + 2
        route = []
        # print('=============================\n\n')
        
        if dis[target[0],target[1],tstate] == -1:
            flag = False

        if flag:
            # print('start',start,'target',target)

            # for r_ in dis:
            #     print(r_)
            # print()
            start = (start[0], start[0], cstate)
            cur_p = (target[0], target[1], tstate)
            # route.append(cur_p)
            print(start, target)
            while True:
                print(cur_p)
                route.append(cur_p)
                if cur_p == fa[cur_p] or cur_p == start:
                    break
                cur_p = fa[cur_p]
                
                
            route.reverse()
            return 1, route
                        
        return 0, route
        # return to_check

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
                if _x[direction] == 1:
                    for i in range(size[1]):
                        if self.map[x + size[0] - 1, y+i] != 0:
                            flag = False
                
                if _x[direction] == -1:
                    for i in range(size[1]):
                        if self.map[x, y+i] != 0:
                            flag = False
                
                if _y[direction] == 1:
                    for i in range(size[0]):
                        if self.map[x+i, y + size[1] - 1] != 0:
                            flag = False

                if _y[direction] == -1:
                    for i in range(size[0]):
                        if self.map[x+i, y] != 0:
                            flag = False

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1]:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        self.route = route
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            self.map[x:x+size[0], y:y+size[1]] = index + 2
            self.pos[index] = target
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        # self.finished = np.zeros(num)

        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break


        self.setmap(pos_list,target_list,size_list)
            
    def getstate_1(self):
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.size,self.cstate,self.tstate)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):
        num = len(self.finished)
        res = np.zeros([num,num,2])
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res


    def getitem(self, index, state):
        size = self.size[index]
        opoints = []
        for i in range(size[0]):
            for j in range(size[1]):
                opoints.append((i,j))
        
        radian = pi * state / self.bin
        points, bbox = self.rotate(opoints, size, radian)
        return points, bbox

    def getcleanmap(self, index):
        start = self.pos[index]
        size = self.size[index]
        cstate = self.cstate[index]
        res = np.array(self.map)
        opoints = []
        for i in range(size[0]):
            for j in range(size[1]):
                opoints.append((i,j))
        
        radian = pi * cstate / self.bin
        points, bbox = self.rotate(opoints, size, radian)

        for p in points:
            x, y = p
            x += start[0]
            y += start[1]
            res[x, y] = 0

        return res

class ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_wall:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        self.bin = 24
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, size, cstate, tstate, wall_list=[]):
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(pos)
        self.size = deepcopy(size)
        self.target = deepcopy(target)
        self.cstate = deepcopy(cstate).astype(np.int)
        self.tstate = deepcopy(tstate).astype(np.int)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            x_, y_ = target[i]
            self.map[x:x+dx, y:y+dy] = i + 2
            self.target_map[x_:x_+dx, y_:y_+dy] = i + 2
        
        for w in wall_list:
            x,y = w
            self.map[x,y] = 1
            self.target_map[x,y] = 1

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos) + 1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod
        
        return total
    

    def rotate(self, points, size, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        points[:,0] += -0.5 * size[0] + 0.5
        points[:,1] += -0.5 * size[1] + 0.5
        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += 0.5 * size[0] - 0.5 + eps
        points[1,:] += 0.5 * size[1] - 0.5 + eps
        points = np.round(points).astype(np.int)
        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        return points.transpose(), np.array([[xmin, ymin], [xmax, ymax]])

    '''
        check the state
        0 represent not accessible
        1 represent accessible
    '''

    def check(self, index):
        flag = True
        target = self.target[index]
        start = self.pos[index]
        size = self.size[index]
        cstate = self.cstate[index]
        # print(cstate)
        tstate = self.tstate[index]
        dis = np.ones([*self.map.shape, self.bin])
        dis *= -1                   # when it starts, all distance was initialized as -1 
        # Q = queue.Queue()
        Q = PQ()
        fn = {}
        fa = {}

        opoints = []
        for i in range(size[0]):
            for j in range(size[1]):
                opoints.append((i,j))
        
        radian = pi * cstate / self.bin
        points, bbox = self.rotate(opoints, size, radian)


        # to confirm the starting point is legal

        # self.map[start[0]:start[0]+size[0],start[1]:start[1]+size[1]] = 0
        for p in points:
            x, y = p
            x += start[0]
            y += start[1]
            self.map[x, y] = 0

        sum_ = np.zeros_like(self.map)
        for i in range(1,sum_.shape[0]):
            sum_[i,0] = sum_[i-1, 0] + self.map[i,0]

        for i in range(1,sum_.shape[1]):
            sum_[0,i] = sum_[0, i-1] + self.map[0,i]

        for i in range(1,sum_.shape[0]):
            for j in range(1,sum_.shape[1]):
                sum_[i,j] = sum_[i,j-1] + sum_[i-1,j] - sum_[i-1,j-1] + self.map[i,j]
        
        
        # for i in range(sum_.shape[0]):
        #     print(sum_[i])

        l = True
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         if self.map[start[0]+i, start[1]+j] != 0:
        #             l = False
        #             break
        #     if not l:
        #         break

        if l:
            f = 0 + abs(start[0] - target[0]) + abs(start[1] - target[1])
            Q.put((f, 0, start, cstate))
            dis[start[0], start[1], cstate] = f
            fa[(start[0], start[1], cstate)] = (start[0], start[1], cstate)
            x, y = start

            for iii in range(1, self.bin):
                s = (cstate + iii) % self.bin
                if dis[x,y,s] != -1:
                    continue

                ra = pi * s / self.bin
                ps, bx = self.rotate(opoints, size, ra)

                if x+bx[1, 0] < self.map_size[0] and x+bx[0, 0] >= 0 and y+bx[1, 1] < self.map_size[1] and y+bx[0, 1] >= 0:
                    aa = sum_[x+bx[1,0], y+bx[1,1]]
                    ab = 0
                    ba = 0
                    bb = 0

                    if x+bx[0,0] != 0:
                        ba = sum_[x+bx[0, 0] - 1, y+bx[1, 1]]
                    
                    if y+bx[0,1] != 0:
                        ab = sum_[x+bx[1, 0], y+bx[0, 1] - 1]
                    
                    if x+bx[0,0] != 0 and y+bx[0, 1] != 0:
                        bb = sum_[x+bx[0, 0]-1, y+bx[0, 1]-1]

                    res = aa - ab - ba + bb

                    push_flag = False
                    if res == 0: # If no collision
                        push_flag = True
                    else:
                        push_flag = True
                        for p in ps:
                            xx, yy = p
                            xx += x
                            yy += y
                            if self.map[xx,yy] != 0:
                                push_flag = False
                                break

                    if push_flag:
                        dis[x,y,s] = f
                        Q.put((f, 0, (x, y), s))
                        fa[(x, y, s)] = (x, y, cstate)
                    else:
                        break

                else:
                    break

            for iii in range(1, self.bin):
                s = (cstate - iii) % self.bin
                if dis[x,y,s] != -1:
                    continue

                ra = pi * s / self.bin
                ps, bx = self.rotate(opoints, size, ra)

                if x+bx[1, 0] < self.map_size[0] and x+bx[0, 0] >= 0 and y+bx[1, 1] < self.map_size[1] and y+bx[0, 1] >= 0:
                    aa = sum_[x+bx[1,0], y+bx[1,1]]
                    ab = 0
                    ba = 0
                    bb = 0

                    if x+bx[0,0] != 0:
                        ba = sum_[x+bx[0, 0] - 1, y+bx[1, 1]]
                    
                    if y+bx[0,1] != 0:
                        ab = sum_[x+bx[1, 0], y+bx[0, 1] - 1]
                    
                    if x+bx[0,0] != 0 and y+bx[0, 1] != 0:
                        bb = sum_[x+bx[0, 0]-1, y+bx[0, 1]-1]

                    res = aa - ab - ba + bb

                    push_flag = False
                    if res == 0: # If no collision
                        push_flag = True
                    else:
                        push_flag = True
                        for p in ps:
                            xx, yy = p
                            xx += x
                            yy += y
                            if self.map[xx,yy] != 0:
                                push_flag = False
                                break

                    if push_flag:
                        dis[x,y,s] = f
                        Q.put((f, 0, (x, y), s))
                        fa[(x, y, s)] = (x, y, cstate)
                    else:
                        break

                else:
                    break


            
        
        # print('====================================================')

        # Heuristic
        while not Q.empty():
            _, d, a, state = Q.get()
            x, y = a
            # print('x',x,'y',y,'dis',d,'f',_)
            if x == target[0] and y == target[1] and state == tstate:
                break
            
            ra = pi * state / self.bin
            ps, bx = self.rotate(opoints, size, ra)

            for i,dx in enumerate(_x):
                dy = _y[i]
                tx = x + dx
                ty = y + dy
                # print('tx,ty',tx,ty,tx+size[0]-1,ty+size[1]-1)
                if tx+bx[1, 0] < self.map_size[0] and tx+bx[0, 0] >= 0 and ty+bx[1, 1] < self.map_size[1] and ty+bx[0, 1] >= 0:
                    aa = sum_[tx+bx[1,0], ty+bx[1,1]]
                    ab = 0
                    ba = 0
                    bb = 0

                    if tx+bx[0,0] != 0:
                        ba = sum_[tx+bx[0, 0] - 1, ty+bx[1, 1]]
                    
                    if ty+bx[0,1] != 0:
                        ab = sum_[tx+bx[1, 0], ty+bx[0, 1] - 1]
                    
                    if tx+bx[0,0] != 0 and ty+bx[0, 1] != 0:
                        bb = sum_[tx+bx[0, 0]-1, ty+bx[0, 1]-1]
                    
                    res = aa - ab - ba + bb
                    # print('aa,ab,ba,bb,res',aa,ab,ba,bb,res)
                    push_flag = False
                    if res == 0: # If no collision
                        push_flag = True
                    else:
                        push_flag = True
                        for p in ps:
                            xx, yy = p
                            xx += tx
                            yy += ty
                            if self.map[xx,yy] != 0:
                                push_flag = False
                                break
                        
                    if push_flag:
                        f_ = d + 1 + abs(tx - target[0]) + abs(ty - target[1])
                        # print('dis',dis[tx,ty],'f_',f_)
                        if dis[tx, ty, state] == -1 or dis[tx, ty, state] > f_:
                            dis[tx, ty, state] = f_
                            Q.put((f_, d + 1, (tx, ty), state))
                            fa[(tx, ty, state)] = (x, y, state)

                            for iii in range(1, self.bin):
                                s = (state + iii) % self.bin
                                if dis[tx,ty,s] != -1:
                                    if dis[tx,ty,s] > f_:
                                        dis[tx,ty,s] = f_
                                        Q.put((f_, d + 1, (tx, ty), s))
                                        fa[(tx, ty, s)] = (x, y, state)
                                    continue

                                ra = pi * s / self.bin
                                ps, bx = self.rotate(opoints, size, ra)

                                if tx+bx[1, 0] < self.map_size[0] and tx+bx[0, 0] >= 0 and ty+bx[1, 1] < self.map_size[1] and ty+bx[0, 1] >= 0:
                                    aa = sum_[tx+bx[1,0], ty+bx[1,1]]
                                    ab = 0
                                    ba = 0
                                    bb = 0

                                    if tx+bx[0,0] != 0:
                                        ba = sum_[tx+bx[0, 0] - 1, ty+bx[1, 1]]
                                    
                                    if ty+bx[0,1] != 0:
                                        ab = sum_[tx+bx[1, 0], ty+bx[0, 1] - 1]
                                    
                                    if tx+bx[0,0] != 0 and ty+bx[0, 1] != 0:
                                        bb = sum_[tx+bx[0, 0]-1, ty+bx[0, 1]-1]

                                    res = aa - ab - ba + bb

                                    push_flag = False
                                    if res == 0: # If no collision
                                        push_flag = True
                                    else:
                                        push_flag = True
                                        for p in ps:
                                            xx, yy = p
                                            xx += tx
                                            yy += ty
                                            if self.map[xx,yy] != 0:
                                                push_flag = False
                                                break
                        
                                    if push_flag:
                                        dis[tx,ty,s] = f_
                                        Q.put((f_, d + 1, (tx, ty), s))
                                        fa[(tx, ty, s)] = (x, y, state)
                                    else:
                                        break

                                else:
                                    break

                            for iii in range(1, self.bin):
                                s = (state - iii + self.bin) % self.bin
                                if dis[tx,ty,s] != -1:
                                    if dis[tx,ty,s] > f_:
                                        dis[tx,ty,s] = f_
                                        Q.put((f_, d + 1, (tx, ty), s))
                                        fa[(tx, ty, s)] = (x, y, state)
                                    continue

                                ra = pi * s / self.bin
                                ps, bx = self.rotate(opoints, size, ra)

                                if tx+bx[1, 0] < self.map_size[0] and tx+bx[0, 0] >= 0 and ty+bx[1, 1] < self.map_size[1] and ty+bx[0, 1] >= 0:
                                    aa = sum_[tx+bx[1,0], ty+bx[1,1]]
                                    ab = 0
                                    ba = 0
                                    bb = 0

                                    if tx+bx[0,0] != 0:
                                        ba = sum_[tx+bx[0, 0] - 1, ty+bx[1, 1]]
                                    
                                    if ty+bx[0,1] != 0:
                                        ab = sum_[tx+bx[1, 0], ty+bx[0, 1] - 1]
                                    
                                    if tx+bx[0,0] != 0 and ty+bx[0, 1] != 0:
                                        bb = sum_[tx+bx[0, 0]-1, ty+bx[0, 1]-1]

                                    res = aa - ab - ba + bb

                                    push_flag = False
                                    if res == 0: # If no collision
                                        push_flag = True
                                    else:
                                        push_flag = True
                                        for p in ps:
                                            xx, yy = p
                                            xx += tx
                                            yy += ty
                                            if self.map[xx,yy] != 0:
                                                push_flag = False
                                                break
                        
                                    if push_flag:
                                        dis[tx,ty,s] = f_
                                        Q.put((f_, d + 1, (tx, ty), s))
                                        fa[(tx, ty, s)] = (x, y, state)
                                    else:
                                        break

                                else:
                                    break  

        for p in points:
            x, y = p
            x += start[0]
            y += start[1]
            self.map[x, y] = index + 2
            
        # self.map[start[0]:start[0]+size[0],start[1]:start[1]+size[1]] = index + 2
        route = []
        # print('=============================\n\n')
        
        if dis[target[0],target[1],tstate] == -1:
            flag = False

        if flag:
            # print('start',start,'target',target)

            # for r_ in dis:
            #     print(r_)
            # print()
            start = (start[0], start[0], cstate)
            cur_p = (target[0], target[1], tstate)
            # route.append(cur_p)
            print(start, target)
            while True:
                print(cur_p)
                route.append(cur_p)
                if cur_p == fa[cur_p] or cur_p == start:
                    break
                cur_p = fa[cur_p]
                
                
            route.reverse()
            return 1, route
                        
        return 0, route
        # return to_check

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]

        if pos == target: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] or x + size[0] > map_size[0] or y + size[1] > map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
                if _x[direction] == 1:
                    for i in range(size[1]):
                        if self.map[x + size[0] - 1, y+i] != 0:
                            flag = False
                
                if _x[direction] == -1:
                    for i in range(size[1]):
                        if self.map[x, y+i] != 0:
                            flag = False
                
                if _y[direction] == 1:
                    for i in range(size[0]):
                        if self.map[x+i, y + size[1] - 1] != 0:
                            flag = False

                if _y[direction] == -1:
                    for i in range(size[0]):
                        if self.map[x+i, y] != 0:
                            flag = False

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                return -500, -1
            
            else:
                self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1]:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        self.route = route
        if flag == 1:
            xx, yy = pos
            x,y = target
            self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            self.map[x:x+size[0], y:y+size[1]] = index + 2
            self.pos[index] = target
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def build_wall(self):
        size = self.map_size
        p = 0.005
        q = 0.05
        dix = [1,0,-1,0]
        diy = [0,1,0,-1]
        while True:
            px = np.random.randint(size[0])
            py = np.random.randint(size[1])
            if fabs(px-size[0]/2) + fabs(py-size[1]/2) > (size[0]+size[1])/4:
                break
        d = int(px < size[0]/2) + int(py < size[1]/2)*2
        if d == 3:
            d = 2
        elif d == 2:
            d = 3

        l = 0
        wall_list = []
        while True:
            cnt = 0
            while True: 
                tx = px + dix[d]
                ty = py + diy[d]
                cnt += 1
                if cnt > 10:
                    break
                if tx == size[0] or tx == -1 or ty == size[1] or ty == -1:
                    d = np.random.randint(4)
                    continue
            
                break

            px = tx
            py = ty
            if not (px,py) in wall_list:
                wall_list.append((px,py))

            r = np.random.rand()
            
            if r < q:
                t = np.random.rand()
                if t < 0.9:
                    d = (d+1)%4
                else:
                    d = np.random.randint(4)
            
            r = np.random.rand()
            if r < p and l > 100:
                break
            
            l += 1

        return wall_list

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        # self.finished = np.zeros(num)
        
        tmp = self.build_wall()
        # tmp += self.build_wall()
        # tmp += self.build_wall()
        sorted(tmp)
        wall_list = []
        l = None
        for t in tmp:
            if t != l:
                l = t
                wall_list.append(t)

        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                hh = [-1,0,1]
                for _ in pos_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx - px) + abs(dy - py) < 5:
                        flag = False
                    
                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break


                for _ in wall_list:
                    if flag == False:
                        break
                    dx,dy = _
                    if dx == px and dy == py:
                        flag = False
                        break
                    
                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break

                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                
                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break

                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break
                
                for _ in wall_list:
                    if flag == False:
                        break
                    dx,dy = _
                    if dx == px and dy == py:
                        flag = False
                        break
                    
                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break

                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(2,31)
                dy = np.random.randint(2,31)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 2, 2
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break


                for wall in wall_list:
                    px_, py_ = wall
                    tx_, ty_ = wall
                    
                    dx_, dy_ = 1, 1
                    

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break


        self.setmap(pos_list,target_list,size_list,np.zeros(len(pos_list)),np.zeros(len(pos_list)),wall_list)
            
    def getstate_1(self):
        state = []
        state.append((self.map == 1).astype(np.float32))
        for i in range(len(self.pos)):
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.size,self.cstate,self.tstate)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):
        num = len(self.finished)
        res = np.zeros([num,num,2])
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res


    def getitem(self, index, state):
        size = self.size[index]
        opoints = []
        for i in range(size[0]):
            for j in range(size[1]):
                opoints.append((i,j))
        
        radian = pi * state / self.bin
        points, bbox = self.rotate(opoints, size, radian)
        return points, bbox

    def getcleanmap(self, index):
        start = self.pos[index]
        size = self.size[index]
        cstate = self.cstate[index]
        res = np.array(self.map)
        opoints = []
        for i in range(size[0]):
            for j in range(size[1]):
                opoints.append((i,j))
        
        radian = pi * cstate / self.bin
        points, bbox = self.rotate(opoints, size, radian)

        for p in points:
            x, y = p
            x += start[0]
            y += start[1]
            res[x, y] = 0

        return res


class ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5)):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        self.bin = 12
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, shape, cstate, tstate, wall=[]):
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(pos)
        self.shape = deepcopy(shape)
        self.target = deepcopy(target)
        self.cstate = deepcopy(np.array(cstate)).astype(np.int)
        self.tstate = deepcopy(np.array(tstate)).astype(np.int)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        
        for p in wall:
            x,y = p
            self.map[x,y] = 1
            self.target_map[x,y] = 1

        for i, _ in enumerate(pos):
            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            # xmax,ymax,xmin,ymin = [-1000,-1000,1000,1000]
            # for p in points:
            #     xx,yy = p
            #     xmax = max(xmax,xx)
            #     xmin = min(xmin,xx)
            #     ymax = max(ymax,yy)
            #     ymin = min(ymin,yy)
            
            # x = max(0,x - 0.5*(xmax-xmin+1))
            # y = max(0,y - 0.5*(ymax-ymin+1))
            
            for p in points:
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0]-1))
                ty = max(0,min(y+yy,self.map_size[1]-1))
                self.map[int(tx),int(ty)] = i + 2
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)

            # for p in points:
            #     xx, yy = p
            #     xmax = max(xmax,xx)
            #     xmin = min(xmin,xx)
            #     ymax = max(ymax,yy)
            #     ymin = min(ymin,yy)
            
            # x = max(0,x - 0.5*(xmax-xmin+1))
            # y = max(0,y - 0.5*(ymax-ymin+1))
            
            
            for p in points:
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0]-1))
                ty = max(0,min(y+yy,self.map_size[1]-1))
                self.target_map[int(tx),int(ty)] = i + 2
                
            # dx, dy = size[i]
            # x, y = _
            # x_, y_ = target[i]
            # self.map[x:x+dx, y:y+dy] = i + 2
            # self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos) + 1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod
        
        return total
    

    def rotate(self, points, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        points[:,0] -= points[:,0].min()
        points[:,1] -= points[:,1].min()
        mx = points[:,0].max()
        my = points[:,1].max()

        points[:,0] -= 0.5 * mx + 0.5
        points[:,1] -= 0.5 * my + 0.5
        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += eps
        points[1,:] += eps
        # points[0,:] += 0.5 * mx - 0.5 + eps
        # points[1,:] += 0.5 * my - 0.5 + eps
        # points = np.round(points).astype(np.int)
        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        # if xmin < 0:
        #     points[0] += 1
        # if ymin < 0:
        #     points[1] += 1

        return points.transpose(), np.array([[xmin, ymin], [xmax, ymax]])

    '''
        check the state
        0 represent not accessible
        1 represent accessible
    '''

    def check(self, index):
        flag = True
        tx,ty = self.target[index]
        sx,sy = self.pos[index]
        shape = self.shape[index]
        cstate = self.cstate[index]
        # print(cstate)
        tstate = self.tstate[index]

        route = []
        ps = np.array(shape)
        px = np.array(ps[:,0],dtype=np.int32)
        py = np.array(ps[:,1],dtype=np.int32)
        num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size
        
        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route

    def equal(self, a, b):
        return fabs(a[0]-b[0])+fabs(a[1]-b[1]) < 1e-6
            

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape,bbox = self.getitem(index, cstate)

        if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            steps = 1
            
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] :
                    steps -= 1
                    break
                    # return -500, -1
                
                flag = True
            
                for p in shape:
                    if x+p[0] < 0 or x+p[0] >= map_size[0] or y+p[1] < 0 or y+p[1] >= map_size[1]:
                        flag = False
                        break

                    if self.map[int(x+p[0]),int(y+p[1])] != 0 and self.map[int(x+p[0]),int(y+p[1])] != index + 2:
                        flag = False
                        break

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]
            
            # print('o steps',steps)
            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[int(xx+p[0]),int(yy+p[1])] = 0

                for p in shape:
                    self.map[int(x+p[0]),int(y+p[1])] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]))
                ty = max(0,min(yy+p[1],self.map_size[1]))
                self.map[int(tx),int(ty)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]))
                ty = max(0,min(y+p[1],self.map_size[1]))
                self.map[int(tx), int(ty)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def move_debug(self, index, direction):
        base = -10
        if index >= len(self.pos):
            print('index error')
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape,bbox = self.getitem(index, cstate)

        if pos == target and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            print('xx',xx,'yy',yy)
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] :
                    steps -= 1
                    print('early done')
                    break
                    # return -500, -1
                
                flag = True
            
                for p in shape:
                    if x+p[0] < 0 or x+p[0] >= map_size[0] or y+p[1] < 0 or y+p[1] >= map_size[1]:
                        flag = False
                        print('over bound',x+p[0],)
                        break
                    if self.map[x+p[0],y+p[1]] != 0:
                        print('collide!!',self.map[x+p[0],y+p[1]],index+2)
                        flag = False
                        break

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                print('cannot move error')
                return -500, -1
            
            else:
                for p in shape:
                    self.map[xx+p[0],yy+p[1]] = 0
                
                for p in shape:
                    self.map[x+p[0],y+p[1]] = index + 2

                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1] and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target and cstate == tstate:
            print('position error')
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            tshape,bbox = self.getitem(index,tstate)
            for p in shape:
                self.map[xx+p[0],yy+p[1]] = 0
            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                self.map[x+p[0], y+p[1]] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            print('path find error')
            return -500, -1


    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        # self.finished = np.zeros(num)

        for i in range(num):
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0]-3)
                py = np.random.randint(size[1]-3)
                flag = True

                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break


        self.setmap(pos_list,target_list,size_list)
            
    def getstate_1(self,shape=[35,35]):
        state = []
        temp = np.zeros(shape).astype(np.float32)
        tmap = self.map == 1
        oshape = self.map.shape
        for i in range(shape[0]):
            x = int(1.0 * i / shape[0] * oshape[0])
            for j in range(shape[1]):
                y = int(1.0 * j / shape[1] * oshape[1])
                temp[i,j] = tmap[x,y]
        # for i in range(oshape[0]):
        #     x = 1.0*i/oshape[0]*shape[0]
        #     for j in range(oshape[1]):
        #         y = 1.0*j/oshape[1]*shape[1]
        #         temp[x,y] = tmap[i,j]

        # state.append((self.map == 1).astype(np.float32))
        state.append(temp)
        
        for i in range(len(self.pos)):
            temp = np.zeros(shape).astype(np.float32)
            tmap = self.map == (i+2)
            for i in range(shape[0]):
                x = int(1.0 * i / shape[0] * oshape[0])
                for j in range(shape[1]):
                    y = int(1.0 * j / shape[1] * oshape[1])
                    temp[i,j] = tmap[x,y]
            
            state.append(temp)
            # state.append((self.map == (i+2)).astype(np.float32))
            # state.append((self.target_map == (i+2)).astype(np.float32))
        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.size,self.cstate,self.tstate)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):
        num = len(self.finished)
        res = np.zeros([num,num,2])
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res


    def getitem(self, index, state):
        shape = self.shape[index]
        opoints = np.array(shape)
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))
        
        radian = 2 * pi * state / self.bin
        points, bbox = self.rotate(opoints, radian)
        return points, bbox

    def getcleanmap(self, index):
        start = self.pos[index]
        # size = self.size[index]
        cstate = self.cstate[index]
        res = np.array(self.map)
        shape = self.shape[index]
        opoints = np.array(shape)
        # opoints = []
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))
        
        radian = 2 * pi * cstate / self.bin
        points, bbox = self.rotate(opoints, radian)
        xx, yy = start
        # xx = max(0, xx - 0.5*(bbox[1,0]-bbox[0,0]+1))
        # yy = max(0, yy - 0.5*(bbox[1,1]-bbox[0,1]+1))

        for p in points:
            x, y = p
            x += xx
            y += yy
            # x = min(x, self.map_size[0]-1)
            # y = min(y, self.map_size[1]-1)
            res[int(x), int(y)] = 0

        return res


class ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5),max_num=5):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        self.bin = 24
        self.max_num = max_num
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, shape, cstate, tstate, wall=[]):
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(np.array(pos))
        self.shape = deepcopy(shape)
        self.target = np.array(target)
        self.cstate = np.array(cstate).astype(np.int)
        self.tstate = np.array(tstate).astype(np.int)
        self.wall = deepcopy(wall)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        self.shapex = []
        self.shapey = []
        # self.boundx = []
        # self.boundy = []
        self.pn = []
        # self.bn = []
        self.edge = []
        self.bound = []
        
        # print(shape)

        
        
        # for bd in self.bound:
        
        # for sh in self.shape:
        #     print(len(sh))
        # for p in self.pos:
        #     print(p)
        
        # print()

        for p in wall:
            x, y = p
            x = int(x)
            y = int(y)
            self.map[x,y] = 1
            self.target_map[x,y] = 1

        # cut_list = []
        for i, _ in enumerate(pos):
            if len(shape[i]) == 0:
                continue
            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))
                
                if self.map[int(tx),int(ty)] >= 1 and self.map[int(tx),int(ty)] != i+2:
                    # if i == 5:
                    #     print('map',self.map[int(tx),int(ty)])

                    id_list.append(i_)
                else:
                    self.map[int(tx),int(ty)] = i + 2

                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            

            # for p in points:
            #     xx, yy = p
            #     xmax = max(xmax,xx)
            #     xmin = min(xmin,xx)
            #     ymax = max(ymax,yy)
            #     ymin = min(ymin,yy)
            
            # x = max(0,x - 0.5*(xmax-xmin+1))
            # y = max(0,y - 0.5*(ymax-ymin+1))
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))

                if self.target_map[int(tx),int(ty)] >= 1 and self.target_map[int(tx),int(ty)] != i + 2:
                    # if i == 5:
                    #     print('tmap',self.map[int(tx),int(ty)])

                    if not i_ in id_list:
                        id_list.append(i_)
                else:
                    self.target_map[int(tx),int(ty)] = i + 2
                
                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
            
            tmp = deepcopy(self.shape[i])
            # print(self.shape[i])
            for i_ in id_list:
                # print(i_,tmp[i_])
                self.shape[i].remove(tmp[i_])
            # dx, dy = size[i]
            # x, y = _
            # x_, y_ = target[i]
            # self.map[x:x+dx, y:y+dy] = i + 2
            # self.target_map[x_:x_+dx, y_:y_+dy] = i + 2
        # print('==========')
        # for sh in self.shape:
        #     print(len(sh))
        self.map = (self.map == 1).astype(np.int32)
        self.target_map = (self.target_map == 1).astype(np.int32)

        for i, _ in enumerate(pos):
            if len(self.shape[i]) == 0:
                continue

            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.map[int(tx),int(ty)] = i + 2
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.target_map[int(tx),int(ty)] = i + 2
                


        for i, sh in enumerate(self.shape):
            

            ed = []
            # bd = []
            map_ = np.zeros(self.map_size)
            mark_ = np.zeros(self.map_size)
            
            for p in sh:
                map_[p[0],p[1]] = 1
            
            # print(i,sh)
            def dfs(p):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                rx = [1,1,1,0,0,-1,-1,-1]
                ry = [-1,0,1,-1,1,-1,0,1]
                stack = []
                stack.append((p,-1))
                last = -2
                
                

                while stack.__len__() != 0:
                    flag = True
                    p,direction = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    # print(p)
                    for i in range(8):
                        x = p[0] + rx[i]
                        y = p[1] + ry[i]

                        if x < 0 or x >= self.map_size[0] or y >= self.map_size[1] or y < 0 or map_[x,y] == 0:
                            # if last == direction:
                            #     ed.pop()

                            ed.append(p)
                            # bd.append(p)
                            last = direction
                            flag = False
                            break

                    vs = []
                    for i in range(8):
                        vs.append([])

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] == 1 and mark_[x,y] == 0:
                            # print(x,y)
                            # dfs((x,y))
                            if flag:
                                # mark_[x,y] = 1
                                stack.append(((x,y),i))
                                break
                            else:
                                cnt = 0
                                for j in range(8):
                                    xx = x + rx[j]
                                    yy = y + ry[j]
                                    if xx < 0 or xx >= self.map_size[0] or yy >= self.map_size[1] or yy < 0 or map_[xx,yy] == 0:
                                        # mark_[x,y] = 1
                                        cnt += 1

                                vs[cnt].append(((x,y),i))
                    
                    for v in vs:
                        for p in v:
                            stack.append(p)
                    
            if len(sh) != 0:
                dfs(sh[0])

            self.edge.append(ed)
            # self.bound.append(bd)
        # for i in range(len(self.pos)):
        #     print(len(self.shape[i]),len(self.edge[i]))

        for i, _ in enumerate(pos):
            if self.equal(self.pos[i],self.target[i]) and self.cstate[i] == self.tstate[i]:
                self.finished[i] = 1

        for ed in self.edge:
            self.pn.append(len(ed))
            for p in ed:
                self.shapex.append(p[0])
                self.shapey.append(p[1])

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos) + 1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod
        
        return total
    

    def rotate(self, points, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        # print(points.shape)
        points[:,0] -= points[:,0].min()
        points[:,1] -= points[:,1].min()
        mx = points[:,0].max()
        my = points[:,1].max()

        points[:,0] -= 0.5 * mx + 0.5
        points[:,1] -= 0.5 * my + 0.5
        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += eps
        points[1,:] += eps
        # points[0,:] += 0.5 * mx - 0.5 + eps
        # points[1,:] += 0.5 * my - 0.5 + eps
        # points = np.round(points).astype(np.int)
        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        # if xmin < 0:
        #     points[0] += 1
        # if ymin < 0:
        #     points[1] += 1

        return points.transpose(), np.array([[xmin, ymin], [xmax, ymax]])

    '''
        check the state
        0 represent not accessible
        1 represent accessible
    '''

    def check(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route
    
    def check_s(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly_rot(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route

    def equal(self, a, b):
        return fabs(a[0]-b[0])+fabs(a[1]-b[1]) < 1e-3
            

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -1
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]
            
            self.last_steps = steps

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 1
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 2, 0
                    else:
                        self.finished[index] = 1
                        return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 1
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 2, 0
                else:
                    self.finished[index] = 1
                    return base + 4, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_s(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            # base += -60
            base += -600
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        # return base + 50, 0
                        return base + 500, 0

                    else:
                        self.finished[index] = 1
                        return base + 2000, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                # return -40, 0
                return -600, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 100000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    # return base + 50, 0
                    return base + 500, 0
                else:
                    self.finished[index] = 1
                    return base + 2000, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_p(self, index, direction):
        base = -1
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 1
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 2, 0
                    else:
                        self.finished[index] = 1
                        return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 1
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 2, 0
                else:
                    self.finished[index] = 1
                    return base + 4, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    
    def move_debug(self, index, direction):
        base = -10
        if index >= len(self.pos):
            print('index error')
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape,bbox = self.getitem(index, cstate)

        if pos == target and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            print('xx',xx,'yy',yy)
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] :
                    steps -= 1
                    print('early done')
                    break
                    # return -500, -1
                
                flag = True
            
                for p in shape:
                    if x+p[0] < 0 or x+p[0] >= map_size[0] or y+p[1] < 0 or y+p[1] >= map_size[1]:
                        flag = False
                        print('over bound',x+p[0],)
                        break
                    if self.map[x+p[0],y+p[1]] != 0:
                        print('collide!!',self.map[x+p[0],y+p[1]],index+2)
                        flag = False
                        break

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                print('cannot move error')
                return -500, -1
            
            else:
                for p in shape:
                    self.map[xx+p[0],yy+p[1]] = 0
                
                for p in shape:
                    self.map[x+p[0],y+p[1]] = index + 2

                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if int(x) == int(target[0]) and int(y) == int(target[1]) and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target and cstate == tstate:
            print('position error')
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            tshape,bbox = self.getitem(index,tstate)
            for p in shape:
                self.map[xx+p[0],yy+p[1]] = 0
            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                self.map[x+p[0], y+p[1]] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            print('path find error')
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')
    
    def build_wall(self):
        size = self.map_size
        p = 0.005
        q = 0.05
        dix = [1,0,-1,0]
        diy = [0,1,0,-1]
        while True:
            px = np.random.randint(size[0])
            py = np.random.randint(size[1])
            if fabs(px-size[0]/2) + fabs(py-size[1]/2) > (size[0]+size[1])/4:
                break
        d = int(px < size[0]/2) + int(py < size[1]/2)*2
        if d == 3:
            d = 2
        elif d == 2:
            d = 3

        l = 0
        wall_list = []
        while True:
            cnt = 0
            while True: 
                tx = px + dix[d]
                ty = py + diy[d]
                cnt += 1
                if cnt > 10:
                    break
                if tx == size[0] or tx == -1 or ty == size[1] or ty == -1:
                    d = np.random.randint(4)
                    continue
            
                break

            if tx >= 0 and tx < size[0] and ty >= 0 and ty < size[1]:
                px = tx
                py = ty
                if not (px,py) in wall_list :
                    wall_list.append((px,py))

            r = np.random.rand()
            
            if r < q:
                t = np.random.rand()
                if t < 0.9:
                    d = (d+1)%4
                else:
                    d = np.random.randint(4)
            
            r = np.random.rand()
            if r < p and l > 100:
                break
            
            l += 1

        return wall_list

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        while True:
            reboot = False
            pos_list = []
            size_list = []
            target_list = []
            # self.finished = np.zeros(num)
            
            tmp = self.build_wall()
            # tmp += self.build_wall()
            # tmp += self.build_wall()
            sorted(tmp)
            wall_list = []
            l = None

            map_ = np.zeros(self.map_size)
            for t in tmp:
                if t != l:
                    l = t
                    map_[l[0],l[1]] = 1
                    wall_list.append(t)

            

            def dfs(p,tp):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]

                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
                        

            for i in range(num):
                while True:
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    flag = True

                    hh = [-1,0,1]
                    for _ in pos_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx - px) + abs(dy - py) < 5:
                            flag = False
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break


                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    if flag:
                        pos_list.append((px,py))
                        break
                
                reboot_cnt = 0
                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    # flag = True
                    
                    flag = dfs((px,py),pos_list[-1])
                    # print(flag)
                    for _ in target_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                            flag = False
                            break

                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break
                    
                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    
                    if flag:
                        target_list.append((px,py))
                        break
                
                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue
            # random.shuffle(pos_list)
            # random.shuffle(target_list)

            sub_num = self.max_num - num
            # print(self.max_num, num)
            pos_s = [ (_, i) for i, _ in enumerate(pos_list)]
            

            for _ in range(sub_num):
                pos_s.append(((0,0),-1))
            # print(len(pos_list))
            random.shuffle(pos_list)
            pos_list = np.array([_[0] for _ in pos_s])
            tmp_target_list = np.zeros_like(pos_list)
            # random.shuffle(target_list)
            _ = 0

            # for target in target_list:
            #     while pos_list[_][0] == pos_list[_][1] and pos_list[_][0] == 0:
            #         _ += 1
            #     tmp_target_list[_] = target
            #     _ += 1
            
            for i,_ in enumerate(pos_s):
                p, id_ = _
                if p[0] == p[1] and p[0] == 0:
                    continue

                tmp_target_list[i] = target_list[id_]

            target_list = tmp_target_list
            
            def dfs(p,tp,size):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                
                dx, dy = size
                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x + dx - 1 < self.map_size[0] and y >= 0 and y + dy - 1 < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            flag = True
                            for j in range(dx):
                                for k in range(dy):
                                    if map_[x+j,y+k] == 1:
                                        flag = False
                                        break
                                if flag == False:
                                    break

                            if flag:
                                stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
            for i in range(self.max_num):
                px, py = pos_list[i]
                tx, ty = target_list[i]

                if px == py and py == 0:
                    size_list.append((0,0))
                    continue

                reboot_cnt = 0

                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    dx = np.random.randint(2,31)
                    dy = np.random.randint(2,31)
                    if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                        # print(px,py,dx,dy)
                        continue

                    flag = True
                    
                    for j in range(self.max_num):
                        if j == i:
                            continue

                        px_, py_ = pos_list[j]
                        tx_, ty_ = target_list[j]

                        if px_ == py_ and py_ == 0:
                            # size_list.append((0,0))
                            continue

                        if j > len(size_list) - 1:
                            dx_, dy_ = 2, 2
                        else:
                            dx_, dy_ = size_list[j]

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    # if flag:
                    #     flag = dfs(pos_list[i],target_list[i],(dx,dy))

                    for wall in wall_list:
                        if not flag:
                            break

                        px_, py_ = wall
                        tx_, ty_ = wall
                        
                        dx_, dy_ = 1, 1
                        

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    if flag:
                        size_list.append((dx,dy))
                        break

                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue

            shapes = []
            for i in range(self.max_num):
                x,y = pos_list[i]
                w,d = size_list[i]
                pos_list[i] = (x+0.5*w,y+0.5*d)
                x,y = target_list[i]
                target_list[i] = (x+0.5*w,y+0.5*d)
                shape = []
                for a in range(w):
                    for b in range(d):
                        shape.append((a,b))

                shapes.append(shape)


            self.setmap(pos_list,target_list,shapes,np.zeros(len(pos_list)),np.zeros(len(pos_list)),wall_list)

            break
    
    def getstate_1(self,shape=[64,64]):
        state = []
        # temp = np.zeros(shape).astype(np.float32)
        # tmap = self.map == 1
        # oshape = self.map.shape
        # for i in range(shape[0]):
        #     x = int(1.0 * i * oshape[0] / shape[0])
        #     for j in range(shape[1]):
        #         y = int(1.0 * j * oshape[1] / shape[1])
        #         temp[i,j] = tmap[x,y]
        # for i in range(oshape[0]):
        #     x = 1.0*i/oshape[0]*shape[0]
        #     for j in range(oshape[1]):
        #         y = 1.0*j/oshape[1]*shape[1]
        #         temp[x,y] = tmap[i,j]

        state.append((self.map == 1).astype(np.float32))
        # state.append(temp)

        for i in range(len(self.pos)):
        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)

        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.target_map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getstate_3(self,shape=[64,64]):
        state = []

        state.append(np.array(self.map).astype(np.float32))
        state.append(np.array(self.target_map).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.shape,self.cstate,self.tstate,self.wall)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):

        num = len(self.finished)
        res = np.zeros([num,num,2])
        # size = np.zeros([num,2])
        # for i,shape in enumerate(self.shape):
        #     size[i]
        return res  
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res

    def getitem(self, index, state):
        shape = self.shape[index]
        opoints = np.array(shape)
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))
        
        radian = 2 * pi * state / self.bin
        points, bbox = self.rotate(opoints, radian)
        return points, bbox

    def getcleanmap(self, index):
        start = self.pos[index]
        # size = self.size[index]
        cstate = self.cstate[index]
        res = np.array(self.map)
        # shape = self.shape[index]
        # opoints = np.array(shape)
        # opoints = []
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))

        points, bbox = self.getitem(index, cstate)
        # radian = 2 * pi * cstate / self.bin
        # points, bbox = self.rotate(opoints, radian)
        xx, yy = start
        # xx = max(0, xx - 0.5*(bbox[1,0]-bbox[0,0]+1))
        # yy = max(0, yy - 0.5*(bbox[1,1]-bbox[0,1]+1))

        for p in points:
            x, y = p
            x += xx
            y += yy
            if x >= self.map_size[0] or y >= self.map_size[0]:
                print('wrong cords',x,y)

            tx = max(0,min(x,self.map_size[0])) + EPS
            ty = max(0,min(y,self.map_size[1])) + EPS
            res[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0
            # x = min(x, self.map_size[0]-1)
            # y = min(y, self.map_size[1]-1)
            # res[int(x), int(y)] = 0

        return res

    def __deepcopy__(self,memodict={}):
        res = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(self.map_size,self.max_num)
        res.map = np.array(self.map)
        res.target_map = np.array(self.target_map)
        res.pos = np.array(self.pos)
        res.target = self.target
        res.shape = self.shape
        res.cstate = np.array(self.cstate)
        res.tstate = self.tstate
        res.wall = self.wall
        res.state_dict = deepcopy(self.state_dict)
        res.finished = np.array(self.finished)
        res.shapex = self.shapex
        res.shapey = self.shapey
        res.pn = self.pn
        res.edge = self.edge
        res.bound = self.bound

        return res
        

class ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly_length:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5),max_num=5):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        self.bin = 24
        self.max_num = max_num
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, shape, cstate, tstate, wall=[]):
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(np.array(pos))
        self.shape = deepcopy(shape)
        self.target = np.array(target)
        self.cstate = np.array(cstate).astype(np.int)
        self.tstate = np.array(tstate).astype(np.int)
        self.wall = deepcopy(wall)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        self.shapex = []
        self.shapey = []
        # self.boundx = []
        # self.boundy = []
        self.pn = []
        # self.bn = []
        self.edge = []
        self.bound = []
        
        # print(shape)

        
        
        # for bd in self.bound:
        
        # for sh in self.shape:
        #     print(len(sh))
        # for p in self.pos:
        #     print(p)
        
        # print()

        for p in wall:
            x, y = p
            x = int(x)
            y = int(y)
            self.map[x,y] = 1
            self.target_map[x,y] = 1

        # cut_list = []
        for i, _ in enumerate(pos):
            if len(shape[i]) == 0:
                continue
            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))
                
                if self.map[int(tx),int(ty)] >= 1 and self.map[int(tx),int(ty)] != i+2:
                    # if i == 5:
                    #     print('map',self.map[int(tx),int(ty)])

                    id_list.append(i_)
                else:
                    self.map[int(tx),int(ty)] = i + 2

                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            

            # for p in points:
            #     xx, yy = p
            #     xmax = max(xmax,xx)
            #     xmin = min(xmin,xx)
            #     ymax = max(ymax,yy)
            #     ymin = min(ymin,yy)
            
            # x = max(0,x - 0.5*(xmax-xmin+1))
            # y = max(0,y - 0.5*(ymax-ymin+1))
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))

                if self.target_map[int(tx),int(ty)] >= 1 and self.target_map[int(tx),int(ty)] != i + 2:
                    # if i == 5:
                    #     print('tmap',self.map[int(tx),int(ty)])

                    if not i_ in id_list:
                        id_list.append(i_)
                else:
                    self.target_map[int(tx),int(ty)] = i + 2
                
                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
            
            tmp = deepcopy(self.shape[i])
            # print(self.shape[i])
            for i_ in id_list:
                # print(i_,tmp[i_])
                self.shape[i].remove(tmp[i_])
            # dx, dy = size[i]
            # x, y = _
            # x_, y_ = target[i]
            # self.map[x:x+dx, y:y+dy] = i + 2
            # self.target_map[x_:x_+dx, y_:y_+dy] = i + 2
        # print('==========')
        # for sh in self.shape:
        #     print(len(sh))
        self.map = (self.map == 1).astype(np.int32)
        self.target_map = (self.target_map == 1).astype(np.int32)

        for i, _ in enumerate(pos):
            if len(self.shape[i]) == 0:
                continue

            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.map[int(tx),int(ty)] = i + 2
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.target_map[int(tx),int(ty)] = i + 2
                


        for i, sh in enumerate(self.shape):
            

            ed = []
            # bd = []
            map_ = np.zeros(self.map_size)
            mark_ = np.zeros(self.map_size)
            
            for p in sh:
                map_[p[0],p[1]] = 1
            
            # print(i,sh)
            def dfs(p):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                rx = [1,1,1,0,0,-1,-1,-1]
                ry = [-1,0,1,-1,1,-1,0,1]
                stack = []
                stack.append((p,-1))
                last = -2
                
                

                while stack.__len__() != 0:
                    flag = True
                    p,direction = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    # print(p)
                    for i in range(8):
                        x = p[0] + rx[i]
                        y = p[1] + ry[i]

                        if x < 0 or x >= self.map_size[0] or y >= self.map_size[1] or y < 0 or map_[x,y] == 0:
                            # if last == direction:
                            #     ed.pop()

                            ed.append(p)
                            # bd.append(p)
                            last = direction
                            flag = False
                            break

                    vs = []
                    for i in range(8):
                        vs.append([])

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] == 1 and mark_[x,y] == 0:
                            # print(x,y)
                            # dfs((x,y))
                            if flag:
                                # mark_[x,y] = 1
                                stack.append(((x,y),i))
                                break
                            else:
                                cnt = 0
                                for j in range(8):
                                    xx = x + rx[j]
                                    yy = y + ry[j]
                                    if xx < 0 or xx >= self.map_size[0] or yy >= self.map_size[1] or yy < 0 or map_[xx,yy] == 0:
                                        # mark_[x,y] = 1
                                        cnt += 1

                                vs[cnt].append(((x,y),i))
                    
                    for v in vs:
                        for p in v:
                            stack.append(p)
                    
            if len(sh) != 0:
                dfs(sh[0])

            self.edge.append(ed)
            # self.bound.append(bd)
        # for i in range(len(self.pos)):
        #     print(len(self.shape[i]),len(self.edge[i]))

        for i, _ in enumerate(pos):
            if self.equal(self.pos[i],self.target[i]) and self.cstate[i] == self.tstate[i]:
                self.finished[i] = 1

        for ed in self.edge:
            self.pn.append(len(ed))
            for p in ed:
                self.shapex.append(p[0])
                self.shapey.append(p[1])

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos) + 1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod
        
        return total
    

    def rotate(self, points, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        # print(points.shape)
        # points[:,0] -= points[:,0].min()
        # points[:,1] -= points[:,1].min()
        # mx = points[:,0].max()
        # my = points[:,1].max()

        # points[:,0] -= 0.5 * mx
        # points[:,1] -= 0.5 * my

        points[:,0] -= 0.5*(points[:,0].max()+points[:,0].min())
        points[:,1] -= 0.5*(points[:,1].max()+points[:,1].min())

        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += eps
        points[1,:] += eps
        # points[0,:] += 0.5 * mx - 0.5 + eps
        # points[1,:] += 0.5 * my - 0.5 + eps
        # points = np.round(points).astype(np.int)
        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        # if xmin < 0:
        #     points[0] += 1
        # if ymin < 0:
        #     points[1] += 1

        return points.transpose(), np.array([[xmin, ymin], [xmax, ymax]])

    '''
        check the state
        0 represent not accessible
        1 represent accessible
    '''

    def check(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route
    
    def check_s(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly_rot(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route

    def equal(self, a, b):
        return fabs(a[0]-b[0])+fabs(a[1]-b[1]) < 1e-3
            
    def route_length(self, route):
        res = 0
        lx = None
        ly = None
        for _ in route:
            x,y,_,_ = _
            if lx is not None:
                res += abs(lx-x) + abs(ly-y)
            lx = x 
            ly = y
        return res
    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -1
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]
            
            self.last_steps = steps

            if steps == 0:
                return -500, -1
            
            else:
                base += -0.5 * log(steps) / log(10)
                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 1
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 2, 0
                    else:
                        self.finished[index] = 1
                        return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route
            length = self.route_length(self.route)
            if length != 0:
                base += -0.5 * log(length) / log(10)
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 1
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 2, 0
                else:
                    self.finished[index] = 1
                    return base + 4, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_s(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            # base += -60
            base += -600
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        # return base + 50, 0
                        return base + 500, 0

                    else:
                        self.finished[index] = 1
                        return base + 2000, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                # return -40, 0
                return -600, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 100000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    # return base + 50, 0
                    return base + 500, 0
                else:
                    self.finished[index] = 1
                    return base + 2000, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_p(self, index, direction):
        base = -1
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 1
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 2, 0
                    else:
                        self.finished[index] = 1
                        return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 1
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 2, 0
                else:
                    self.finished[index] = 1
                    return base + 4, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    
    def move_debug(self, index, direction):
        base = -10
        if index >= len(self.pos):
            print('index error')
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape,bbox = self.getitem(index, cstate)

        if pos == target and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            print('xx',xx,'yy',yy)
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] :
                    steps -= 1
                    print('early done')
                    break
                    # return -500, -1
                
                flag = True
            
                for p in shape:
                    if x+p[0] < 0 or x+p[0] >= map_size[0] or y+p[1] < 0 or y+p[1] >= map_size[1]:
                        flag = False
                        print('over bound',x+p[0],)
                        break
                    if self.map[x+p[0],y+p[1]] != 0:
                        print('collide!!',self.map[x+p[0],y+p[1]],index+2)
                        flag = False
                        break

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                print('cannot move error')
                return -500, -1
            
            else:
                for p in shape:
                    self.map[xx+p[0],yy+p[1]] = 0
                
                for p in shape:
                    self.map[x+p[0],y+p[1]] = index + 2

                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if int(x) == int(target[0]) and int(y) == int(target[1]) and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target and cstate == tstate:
            print('position error')
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            tshape,bbox = self.getitem(index,tstate)
            for p in shape:
                self.map[xx+p[0],yy+p[1]] = 0
            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                self.map[x+p[0], y+p[1]] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            print('path find error')
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')
    
    def build_wall(self):
        size = self.map_size
        p = 0.005
        q = 0.05
        dix = [1,0,-1,0]
        diy = [0,1,0,-1]
        while True:
            px = np.random.randint(size[0])
            py = np.random.randint(size[1])
            if fabs(px-size[0]/2) + fabs(py-size[1]/2) > (size[0]+size[1])/4:
                break
        d = int(px < size[0]/2) + int(py < size[1]/2)*2
        if d == 3:
            d = 2
        elif d == 2:
            d = 3

        l = 0
        wall_list = []
        while True:
            cnt = 0
            while True: 
                tx = px + dix[d]
                ty = py + diy[d]
                cnt += 1
                if cnt > 10:
                    break
                if tx == size[0] or tx == -1 or ty == size[1] or ty == -1:
                    d = np.random.randint(4)
                    continue
            
                break

            if tx >= 0 and tx < size[0] and ty >= 0 and ty < size[1]:
                px = tx
                py = ty
                if not (px,py) in wall_list :
                    wall_list.append((px,py))

            r = np.random.rand()
            
            if r < q:
                t = np.random.rand()
                if t < 0.9:
                    d = (d+1)%4
                else:
                    d = np.random.randint(4)
            
            r = np.random.rand()
            if r < p and l > 100:
                break
            
            l += 1

        return wall_list

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        while True:
            reboot = False
            pos_list = []
            size_list = []
            target_list = []
            # self.finished = np.zeros(num)
            
            tmp = self.build_wall()
            # tmp += self.build_wall()
            # tmp += self.build_wall()
            sorted(tmp)
            wall_list = []
            l = None

            map_ = np.zeros(self.map_size)
            for t in tmp:
                if t != l:
                    l = t
                    map_[l[0],l[1]] = 1
                    wall_list.append(t)

            

            def dfs(p,tp):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]

                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
                        

            for i in range(num):
                while True:
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    flag = True

                    hh = [-1,0,1]
                    for _ in pos_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx - px) + abs(dy - py) < 5:
                            flag = False
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break


                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    if flag:
                        pos_list.append((px,py))
                        break
                
                reboot_cnt = 0
                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    # flag = True
                    
                    flag = dfs((px,py),pos_list[-1])
                    # print(flag)
                    for _ in target_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                            flag = False
                            break

                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break
                    
                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    
                    if flag:
                        target_list.append((px,py))
                        break
                
                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue
            # random.shuffle(pos_list)
            # random.shuffle(target_list)

            sub_num = self.max_num - num
            # print(self.max_num, num)
            pos_s = [ (_, i) for i, _ in enumerate(pos_list)]
            

            for _ in range(sub_num):
                pos_s.append(((0,0),-1))
            # print(len(pos_list))
            random.shuffle(pos_list)
            pos_list = np.array([_[0] for _ in pos_s])
            tmp_target_list = np.zeros_like(pos_list)
            # random.shuffle(target_list)
            _ = 0

            # for target in target_list:
            #     while pos_list[_][0] == pos_list[_][1] and pos_list[_][0] == 0:
            #         _ += 1
            #     tmp_target_list[_] = target
            #     _ += 1
            
            for i,_ in enumerate(pos_s):
                p, id_ = _
                if p[0] == p[1] and p[0] == 0:
                    continue

                tmp_target_list[i] = target_list[id_]

            target_list = tmp_target_list
            
            def dfs(p,tp,size):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                
                dx, dy = size
                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x + dx - 1 < self.map_size[0] and y >= 0 and y + dy - 1 < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            flag = True
                            for j in range(dx):
                                for k in range(dy):
                                    if map_[x+j,y+k] == 1:
                                        flag = False
                                        break
                                if flag == False:
                                    break

                            if flag:
                                stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
            for i in range(self.max_num):
                px, py = pos_list[i]
                tx, ty = target_list[i]

                if px == py and py == 0:
                    size_list.append((0,0))
                    continue

                reboot_cnt = 0

                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    dx = np.random.randint(2,31)
                    dy = np.random.randint(2,31)
                    if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                        # print(px,py,dx,dy)
                        continue

                    flag = True
                    
                    for j in range(self.max_num):
                        if j == i:
                            continue

                        px_, py_ = pos_list[j]
                        tx_, ty_ = target_list[j]

                        if px_ == py_ and py_ == 0:
                            # size_list.append((0,0))
                            continue

                        if j > len(size_list) - 1:
                            dx_, dy_ = 2, 2
                        else:
                            dx_, dy_ = size_list[j]

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    # if flag:
                    #     flag = dfs(pos_list[i],target_list[i],(dx,dy))

                    for wall in wall_list:
                        if not flag:
                            break

                        px_, py_ = wall
                        tx_, ty_ = wall
                        
                        dx_, dy_ = 1, 1
                        

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    if flag:
                        size_list.append((dx,dy))
                        break

                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue

            shapes = []
            for i in range(self.max_num):
                x,y = pos_list[i]
                w,d = size_list[i]
                pos_list[i] = (x+0.5*w,y+0.5*d)
                x,y = target_list[i]
                target_list[i] = (x+0.5*w,y+0.5*d)
                shape = []
                for a in range(w):
                    for b in range(d):
                        shape.append((a,b))

                shapes.append(shape)


            self.setmap(pos_list,target_list,shapes,np.zeros(len(pos_list)),np.zeros(len(pos_list)),wall_list)

            break
    
    def getstate_1(self,shape=[64,64]):
        state = []
        # temp = np.zeros(shape).astype(np.float32)
        # tmap = self.map == 1
        # oshape = self.map.shape
        # for i in range(shape[0]):
        #     x = int(1.0 * i * oshape[0] / shape[0])
        #     for j in range(shape[1]):
        #         y = int(1.0 * j * oshape[1] / shape[1])
        #         temp[i,j] = tmap[x,y]
        # for i in range(oshape[0]):
        #     x = 1.0*i/oshape[0]*shape[0]
        #     for j in range(oshape[1]):
        #         y = 1.0*j/oshape[1]*shape[1]
        #         temp[x,y] = tmap[i,j]

        state.append((self.map == 1).astype(np.float32))
        # state.append(temp)

        for i in range(len(self.pos)):
        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)

        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.target_map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getstate_3(self,shape=[64,64]):
        state = []

        state.append(np.array(self.map).astype(np.float32))
        state.append(np.array(self.target_map).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.shape,self.cstate,self.tstate,self.wall)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):

        num = len(self.finished)
        res = np.zeros([num,num,2])
        # size = np.zeros([num,2])
        # for i,shape in enumerate(self.shape):
        #     size[i]
        return res  
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res

    def getitem(self, index, state):
        shape = self.shape[index]
        opoints = np.array(shape)
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))
        
        radian = 2 * pi * state / self.bin
        points, bbox = self.rotate(opoints, radian)
        return points, bbox

    def getcleanmap(self, index):
        start = self.pos[index]
        # size = self.size[index]
        cstate = self.cstate[index]
        res = np.array(self.map)
        # shape = self.shape[index]
        # opoints = np.array(shape)
        # opoints = []
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))

        points, bbox = self.getitem(index, cstate)
        # radian = 2 * pi * cstate / self.bin
        # points, bbox = self.rotate(opoints, radian)
        xx, yy = start
        # xx = max(0, xx - 0.5*(bbox[1,0]-bbox[0,0]+1))
        # yy = max(0, yy - 0.5*(bbox[1,1]-bbox[0,1]+1))

        for p in points:
            x, y = p
            x += xx
            y += yy
            if x >= self.map_size[0] or y >= self.map_size[0]:
                print('wrong cords',x,y)

            tx = max(0,min(x,self.map_size[0])) + EPS
            ty = max(0,min(y,self.map_size[1])) + EPS
            res[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0
            # x = min(x, self.map_size[0]-1)
            # y = min(y, self.map_size[1]-1)
            # res[int(x), int(y)] = 0

        return res

    def __deepcopy__(self,memodict={}):
        res = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly_length(self.map_size,self.max_num)
        res.map = np.array(self.map)
        res.target_map = np.array(self.target_map)
        res.pos = np.array(self.pos)
        res.target = self.target
        res.shape = self.shape
        res.cstate = np.array(self.cstate)
        res.tstate = self.tstate
        res.wall = self.wall
        res.state_dict = deepcopy(self.state_dict)
        res.finished = np.array(self.finished)
        res.shapex = self.shapex
        res.shapey = self.shapey
        res.pn = self.pn
        res.edge = self.edge
        res.bound = self.bound

        return res
        
class ENV_ablation_wo_base:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5),max_num=5):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        self.bin = 24
        self.max_num = max_num
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, shape, cstate, tstate, wall=[]):
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(np.array(pos))
        self.shape = deepcopy(shape)
        self.target = np.array(target)
        self.cstate = np.array(cstate).astype(np.int)
        self.tstate = np.array(tstate).astype(np.int)
        self.wall = deepcopy(wall)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        self.shapex = []
        self.shapey = []
        # self.boundx = []
        # self.boundy = []
        self.pn = []
        # self.bn = []
        self.edge = []
        self.bound = []
        
        # print(shape)

        
        
        # for bd in self.bound:
        
        # for sh in self.shape:
        #     print(len(sh))
        # for p in self.pos:
        #     print(p)
        
        # print()

        for p in wall:
            x, y = p
            x = int(x)
            y = int(y)
            self.map[x,y] = 1
            self.target_map[x,y] = 1

        # cut_list = []
        for i, _ in enumerate(pos):
            if len(shape[i]) == 0:
                continue
            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))
                
                if self.map[int(tx),int(ty)] >= 1 and self.map[int(tx),int(ty)] != i+2:
                    # if i == 5:
                    #     print('map',self.map[int(tx),int(ty)])

                    id_list.append(i_)
                else:
                    self.map[int(tx),int(ty)] = i + 2

                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            

            # for p in points:
            #     xx, yy = p
            #     xmax = max(xmax,xx)
            #     xmin = min(xmin,xx)
            #     ymax = max(ymax,yy)
            #     ymin = min(ymin,yy)
            
            # x = max(0,x - 0.5*(xmax-xmin+1))
            # y = max(0,y - 0.5*(ymax-ymin+1))
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))

                if self.target_map[int(tx),int(ty)] >= 1 and self.target_map[int(tx),int(ty)] != i + 2:
                    # if i == 5:
                    #     print('tmap',self.map[int(tx),int(ty)])

                    if not i_ in id_list:
                        id_list.append(i_)
                else:
                    self.target_map[int(tx),int(ty)] = i + 2
                
                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
            
            tmp = deepcopy(self.shape[i])
            # print(self.shape[i])
            for i_ in id_list:
                # print(i_,tmp[i_])
                self.shape[i].remove(tmp[i_])
            # dx, dy = size[i]
            # x, y = _
            # x_, y_ = target[i]
            # self.map[x:x+dx, y:y+dy] = i + 2
            # self.target_map[x_:x_+dx, y_:y_+dy] = i + 2
        # print('==========')
        # for sh in self.shape:
        #     print(len(sh))
        self.map = (self.map == 1).astype(np.int32)
        self.target_map = (self.target_map == 1).astype(np.int32)

        for i, _ in enumerate(pos):
            if len(self.shape[i]) == 0:
                continue

            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.map[int(tx),int(ty)] = i + 2
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.target_map[int(tx),int(ty)] = i + 2
                


        for i, sh in enumerate(self.shape):
            

            ed = []
            # bd = []
            map_ = np.zeros(self.map_size)
            mark_ = np.zeros(self.map_size)
            
            for p in sh:
                map_[p[0],p[1]] = 1
            
            # print(i,sh)
            def dfs(p):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                rx = [1,1,1,0,0,-1,-1,-1]
                ry = [-1,0,1,-1,1,-1,0,1]
                stack = []
                stack.append((p,-1))
                last = -2
                
                

                while stack.__len__() != 0:
                    flag = True
                    p,direction = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    # print(p)
                    for i in range(8):
                        x = p[0] + rx[i]
                        y = p[1] + ry[i]

                        if x < 0 or x >= self.map_size[0] or y >= self.map_size[1] or y < 0 or map_[x,y] == 0:
                            # if last == direction:
                            #     ed.pop()

                            ed.append(p)
                            # bd.append(p)
                            last = direction
                            flag = False
                            break

                    vs = []
                    for i in range(8):
                        vs.append([])

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] == 1 and mark_[x,y] == 0:
                            # print(x,y)
                            # dfs((x,y))
                            if flag:
                                # mark_[x,y] = 1
                                stack.append(((x,y),i))
                                break
                            else:
                                cnt = 0
                                for j in range(8):
                                    xx = x + rx[j]
                                    yy = y + ry[j]
                                    if xx < 0 or xx >= self.map_size[0] or yy >= self.map_size[1] or yy < 0 or map_[xx,yy] == 0:
                                        # mark_[x,y] = 1
                                        cnt += 1

                                vs[cnt].append(((x,y),i))
                    
                    for v in vs:
                        for p in v:
                            stack.append(p)
                    
            if len(sh) != 0:
                dfs(sh[0])

            self.edge.append(ed)
            # self.bound.append(bd)
        # for i in range(len(self.pos)):
        #     print(len(self.shape[i]),len(self.edge[i]))

        for i, _ in enumerate(pos):
            if self.equal(self.pos[i],self.target[i]) and self.cstate[i] == self.tstate[i]:
                self.finished[i] = 1

        for ed in self.edge:
            self.pn.append(len(ed))
            for p in ed:
                self.shapex.append(p[0])
                self.shapey.append(p[1])

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos) + 1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod
        
        return total
    

    def rotate(self, points, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        # print(points.shape)
        points[:,0] -= points[:,0].min()
        points[:,1] -= points[:,1].min()
        mx = points[:,0].max()
        my = points[:,1].max()

        points[:,0] -= 0.5 * mx + 0.5
        points[:,1] -= 0.5 * my + 0.5
        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += eps
        points[1,:] += eps
        # points[0,:] += 0.5 * mx - 0.5 + eps
        # points[1,:] += 0.5 * my - 0.5 + eps
        # points = np.round(points).astype(np.int)
        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        # if xmin < 0:
        #     points[0] += 1
        # if ymin < 0:
        #     points[1] += 1

        return points.transpose(), np.array([[xmin, ymin], [xmax, ymax]])

    '''
        check the state
        0 represent not accessible
        1 represent accessible
    '''

    def check(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route
    
    def check_s(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly_rot(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route

    def equal(self, a, b):
        return fabs(a[0]-b[0])+fabs(a[1]-b[1]) < 1e-3
            
    def route_length(self, route):
        res = 0
        lx = None
        ly = None
        for _ in route:
            x,y,_,_ = _
            if lx is not None:
                res += abs(lx-x) + abs(ly-y)
            lx = x 
            ly = y
        return res
    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = 0
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]
            
            self.last_steps = steps

            if steps == 0:
                return -500, -1
            
            else:
                base += -0.5 * log(steps) / log(10)
                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 1
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 2, 0
                    else:
                        self.finished[index] = 1
                        return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route
            length = self.route_length(self.route)
            if length != 0:
                base += -0.5 * log(length) / log(10)
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 1
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 2, 0
                else:
                    self.finished[index] = 1
                    return base + 4, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_s(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            # base += -60
            base += -600
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        # return base + 50, 0
                        return base + 500, 0

                    else:
                        self.finished[index] = 1
                        return base + 2000, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                # return -40, 0
                return -600, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 100000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    # return base + 50, 0
                    return base + 500, 0
                else:
                    self.finished[index] = 1
                    return base + 2000, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_p(self, index, direction):
        base = -1
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 1
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 2, 0
                    else:
                        self.finished[index] = 1
                        return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 1
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 2, 0
                else:
                    self.finished[index] = 1
                    return base + 4, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    
    def move_debug(self, index, direction):
        base = -10
        if index >= len(self.pos):
            print('index error')
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape,bbox = self.getitem(index, cstate)

        if pos == target and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            print('xx',xx,'yy',yy)
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] :
                    steps -= 1
                    print('early done')
                    break
                    # return -500, -1
                
                flag = True
            
                for p in shape:
                    if x+p[0] < 0 or x+p[0] >= map_size[0] or y+p[1] < 0 or y+p[1] >= map_size[1]:
                        flag = False
                        print('over bound',x+p[0],)
                        break
                    if self.map[x+p[0],y+p[1]] != 0:
                        print('collide!!',self.map[x+p[0],y+p[1]],index+2)
                        flag = False
                        break

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                print('cannot move error')
                return -500, -1
            
            else:
                for p in shape:
                    self.map[xx+p[0],yy+p[1]] = 0
                
                for p in shape:
                    self.map[x+p[0],y+p[1]] = index + 2

                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if int(x) == int(target[0]) and int(y) == int(target[1]) and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target and cstate == tstate:
            print('position error')
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            tshape,bbox = self.getitem(index,tstate)
            for p in shape:
                self.map[xx+p[0],yy+p[1]] = 0
            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                self.map[x+p[0], y+p[1]] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            print('path find error')
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')
    
    def build_wall(self):
        size = self.map_size
        p = 0.005
        q = 0.05
        dix = [1,0,-1,0]
        diy = [0,1,0,-1]
        while True:
            px = np.random.randint(size[0])
            py = np.random.randint(size[1])
            if fabs(px-size[0]/2) + fabs(py-size[1]/2) > (size[0]+size[1])/4:
                break
        d = int(px < size[0]/2) + int(py < size[1]/2)*2
        if d == 3:
            d = 2
        elif d == 2:
            d = 3

        l = 0
        wall_list = []
        while True:
            cnt = 0
            while True: 
                tx = px + dix[d]
                ty = py + diy[d]
                cnt += 1
                if cnt > 10:
                    break
                if tx == size[0] or tx == -1 or ty == size[1] or ty == -1:
                    d = np.random.randint(4)
                    continue
            
                break

            if tx >= 0 and tx < size[0] and ty >= 0 and ty < size[1]:
                px = tx
                py = ty
                if not (px,py) in wall_list :
                    wall_list.append((px,py))

            r = np.random.rand()
            
            if r < q:
                t = np.random.rand()
                if t < 0.9:
                    d = (d+1)%4
                else:
                    d = np.random.randint(4)
            
            r = np.random.rand()
            if r < p and l > 100:
                break
            
            l += 1

        return wall_list

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        while True:
            reboot = False
            pos_list = []
            size_list = []
            target_list = []
            # self.finished = np.zeros(num)
            
            tmp = self.build_wall()
            # tmp += self.build_wall()
            # tmp += self.build_wall()
            sorted(tmp)
            wall_list = []
            l = None

            map_ = np.zeros(self.map_size)
            for t in tmp:
                if t != l:
                    l = t
                    map_[l[0],l[1]] = 1
                    wall_list.append(t)

            

            def dfs(p,tp):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]

                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
                        

            for i in range(num):
                while True:
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    flag = True

                    hh = [-1,0,1]
                    for _ in pos_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx - px) + abs(dy - py) < 5:
                            flag = False
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break


                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    if flag:
                        pos_list.append((px,py))
                        break
                
                reboot_cnt = 0
                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    # flag = True
                    
                    flag = dfs((px,py),pos_list[-1])
                    # print(flag)
                    for _ in target_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                            flag = False
                            break

                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break
                    
                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    
                    if flag:
                        target_list.append((px,py))
                        break
                
                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue
            # random.shuffle(pos_list)
            # random.shuffle(target_list)

            sub_num = self.max_num - num
            # print(self.max_num, num)
            pos_s = [ (_, i) for i, _ in enumerate(pos_list)]
            

            for _ in range(sub_num):
                pos_s.append(((0,0),-1))
            # print(len(pos_list))
            random.shuffle(pos_list)
            pos_list = np.array([_[0] for _ in pos_s])
            tmp_target_list = np.zeros_like(pos_list)
            # random.shuffle(target_list)
            _ = 0

            # for target in target_list:
            #     while pos_list[_][0] == pos_list[_][1] and pos_list[_][0] == 0:
            #         _ += 1
            #     tmp_target_list[_] = target
            #     _ += 1
            
            for i,_ in enumerate(pos_s):
                p, id_ = _
                if p[0] == p[1] and p[0] == 0:
                    continue

                tmp_target_list[i] = target_list[id_]

            target_list = tmp_target_list
            
            def dfs(p,tp,size):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                
                dx, dy = size
                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x + dx - 1 < self.map_size[0] and y >= 0 and y + dy - 1 < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            flag = True
                            for j in range(dx):
                                for k in range(dy):
                                    if map_[x+j,y+k] == 1:
                                        flag = False
                                        break
                                if flag == False:
                                    break

                            if flag:
                                stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
            for i in range(self.max_num):
                px, py = pos_list[i]
                tx, ty = target_list[i]

                if px == py and py == 0:
                    size_list.append((0,0))
                    continue

                reboot_cnt = 0

                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    dx = np.random.randint(2,31)
                    dy = np.random.randint(2,31)
                    if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                        # print(px,py,dx,dy)
                        continue

                    flag = True
                    
                    for j in range(self.max_num):
                        if j == i:
                            continue

                        px_, py_ = pos_list[j]
                        tx_, ty_ = target_list[j]

                        if px_ == py_ and py_ == 0:
                            # size_list.append((0,0))
                            continue

                        if j > len(size_list) - 1:
                            dx_, dy_ = 2, 2
                        else:
                            dx_, dy_ = size_list[j]

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    # if flag:
                    #     flag = dfs(pos_list[i],target_list[i],(dx,dy))

                    for wall in wall_list:
                        if not flag:
                            break

                        px_, py_ = wall
                        tx_, ty_ = wall
                        
                        dx_, dy_ = 1, 1
                        

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    if flag:
                        size_list.append((dx,dy))
                        break

                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue

            shapes = []
            for i in range(self.max_num):
                x,y = pos_list[i]
                w,d = size_list[i]
                pos_list[i] = (x+0.5*w,y+0.5*d)
                x,y = target_list[i]
                target_list[i] = (x+0.5*w,y+0.5*d)
                shape = []
                for a in range(w):
                    for b in range(d):
                        shape.append((a,b))

                shapes.append(shape)


            self.setmap(pos_list,target_list,shapes,np.zeros(len(pos_list)),np.zeros(len(pos_list)),wall_list)

            break
    
    def getstate_1(self,shape=[64,64]):
        state = []
        # temp = np.zeros(shape).astype(np.float32)
        # tmap = self.map == 1
        # oshape = self.map.shape
        # for i in range(shape[0]):
        #     x = int(1.0 * i * oshape[0] / shape[0])
        #     for j in range(shape[1]):
        #         y = int(1.0 * j * oshape[1] / shape[1])
        #         temp[i,j] = tmap[x,y]
        # for i in range(oshape[0]):
        #     x = 1.0*i/oshape[0]*shape[0]
        #     for j in range(oshape[1]):
        #         y = 1.0*j/oshape[1]*shape[1]
        #         temp[x,y] = tmap[i,j]

        state.append((self.map == 1).astype(np.float32))
        # state.append(temp)

        for i in range(len(self.pos)):
        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)

        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.target_map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getstate_3(self,shape=[64,64]):
        state = []

        state.append(np.array(self.map).astype(np.float32))
        state.append(np.array(self.target_map).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.shape,self.cstate,self.tstate,self.wall)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):

        num = len(self.finished)
        res = np.zeros([num,num,2])
        # size = np.zeros([num,2])
        # for i,shape in enumerate(self.shape):
        #     size[i]
        return res  
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res

    def getitem(self, index, state):
        shape = self.shape[index]
        opoints = np.array(shape)
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))
        
        radian = 2 * pi * state / self.bin
        points, bbox = self.rotate(opoints, radian)
        return points, bbox

    def getcleanmap(self, index):
        start = self.pos[index]
        # size = self.size[index]
        cstate = self.cstate[index]
        res = np.array(self.map)
        # shape = self.shape[index]
        # opoints = np.array(shape)
        # opoints = []
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))

        points, bbox = self.getitem(index, cstate)
        # radian = 2 * pi * cstate / self.bin
        # points, bbox = self.rotate(opoints, radian)
        xx, yy = start
        # xx = max(0, xx - 0.5*(bbox[1,0]-bbox[0,0]+1))
        # yy = max(0, yy - 0.5*(bbox[1,1]-bbox[0,1]+1))

        for p in points:
            x, y = p
            x += xx
            y += yy
            if x >= self.map_size[0] or y >= self.map_size[0]:
                print('wrong cords',x,y)

            tx = max(0,min(x,self.map_size[0])) + EPS
            ty = max(0,min(y,self.map_size[1])) + EPS
            res[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0
            # x = min(x, self.map_size[0]-1)
            # y = min(y, self.map_size[1]-1)
            # res[int(x), int(y)] = 0

        return res

    def __deepcopy__(self,memodict={}):
        res = ENV_ablation_wo_base(self.map_size,self.max_num)
        res.map = np.array(self.map)
        res.target_map = np.array(self.target_map)
        res.pos = np.array(self.pos)
        res.target = self.target
        res.shape = self.shape
        res.cstate = np.array(self.cstate)
        res.tstate = self.tstate
        res.wall = self.wall
        res.state_dict = deepcopy(self.state_dict)
        res.finished = np.array(self.finished)
        res.shapex = self.shapex
        res.shapey = self.shapey
        res.pn = self.pn
        res.edge = self.edge
        res.bound = self.bound

        return res
        
class ENV_ablation_wo_repetition:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5),max_num=5):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        self.bin = 24
        self.max_num = max_num
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, shape, cstate, tstate, wall=[]):
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(np.array(pos))
        self.shape = deepcopy(shape)
        self.target = np.array(target)
        self.cstate = np.array(cstate).astype(np.int)
        self.tstate = np.array(tstate).astype(np.int)
        self.wall = deepcopy(wall)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        self.shapex = []
        self.shapey = []
        # self.boundx = []
        # self.boundy = []
        self.pn = []
        # self.bn = []
        self.edge = []
        self.bound = []
        
        # print(shape)

        
        
        # for bd in self.bound:
        
        # for sh in self.shape:
        #     print(len(sh))
        # for p in self.pos:
        #     print(p)
        
        # print()

        for p in wall:
            x, y = p
            x = int(x)
            y = int(y)
            self.map[x,y] = 1
            self.target_map[x,y] = 1

        # cut_list = []
        for i, _ in enumerate(pos):
            if len(shape[i]) == 0:
                continue
            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))
                
                if self.map[int(tx),int(ty)] >= 1 and self.map[int(tx),int(ty)] != i+2:
                    # if i == 5:
                    #     print('map',self.map[int(tx),int(ty)])

                    id_list.append(i_)
                else:
                    self.map[int(tx),int(ty)] = i + 2

                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            

            # for p in points:
            #     xx, yy = p
            #     xmax = max(xmax,xx)
            #     xmin = min(xmin,xx)
            #     ymax = max(ymax,yy)
            #     ymin = min(ymin,yy)
            
            # x = max(0,x - 0.5*(xmax-xmin+1))
            # y = max(0,y - 0.5*(ymax-ymin+1))
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))

                if self.target_map[int(tx),int(ty)] >= 1 and self.target_map[int(tx),int(ty)] != i + 2:
                    # if i == 5:
                    #     print('tmap',self.map[int(tx),int(ty)])

                    if not i_ in id_list:
                        id_list.append(i_)
                else:
                    self.target_map[int(tx),int(ty)] = i + 2
                
                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
            
            tmp = deepcopy(self.shape[i])
            # print(self.shape[i])
            for i_ in id_list:
                # print(i_,tmp[i_])
                self.shape[i].remove(tmp[i_])
            # dx, dy = size[i]
            # x, y = _
            # x_, y_ = target[i]
            # self.map[x:x+dx, y:y+dy] = i + 2
            # self.target_map[x_:x_+dx, y_:y_+dy] = i + 2
        # print('==========')
        # for sh in self.shape:
        #     print(len(sh))
        self.map = (self.map == 1).astype(np.int32)
        self.target_map = (self.target_map == 1).astype(np.int32)

        for i, _ in enumerate(pos):
            if len(self.shape[i]) == 0:
                continue

            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.map[int(tx),int(ty)] = i + 2
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.target_map[int(tx),int(ty)] = i + 2
                


        for i, sh in enumerate(self.shape):
            

            ed = []
            # bd = []
            map_ = np.zeros(self.map_size)
            mark_ = np.zeros(self.map_size)
            
            for p in sh:
                map_[p[0],p[1]] = 1
            
            # print(i,sh)
            def dfs(p):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                rx = [1,1,1,0,0,-1,-1,-1]
                ry = [-1,0,1,-1,1,-1,0,1]
                stack = []
                stack.append((p,-1))
                last = -2
                
                

                while stack.__len__() != 0:
                    flag = True
                    p,direction = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    # print(p)
                    for i in range(8):
                        x = p[0] + rx[i]
                        y = p[1] + ry[i]

                        if x < 0 or x >= self.map_size[0] or y >= self.map_size[1] or y < 0 or map_[x,y] == 0:
                            # if last == direction:
                            #     ed.pop()

                            ed.append(p)
                            # bd.append(p)
                            last = direction
                            flag = False
                            break

                    vs = []
                    for i in range(8):
                        vs.append([])

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] == 1 and mark_[x,y] == 0:
                            # print(x,y)
                            # dfs((x,y))
                            if flag:
                                # mark_[x,y] = 1
                                stack.append(((x,y),i))
                                break
                            else:
                                cnt = 0
                                for j in range(8):
                                    xx = x + rx[j]
                                    yy = y + ry[j]
                                    if xx < 0 or xx >= self.map_size[0] or yy >= self.map_size[1] or yy < 0 or map_[xx,yy] == 0:
                                        # mark_[x,y] = 1
                                        cnt += 1

                                vs[cnt].append(((x,y),i))
                    
                    for v in vs:
                        for p in v:
                            stack.append(p)
                    
            if len(sh) != 0:
                dfs(sh[0])

            self.edge.append(ed)
            # self.bound.append(bd)
        # for i in range(len(self.pos)):
        #     print(len(self.shape[i]),len(self.edge[i]))

        for i, _ in enumerate(pos):
            if self.equal(self.pos[i],self.target[i]) and self.cstate[i] == self.tstate[i]:
                self.finished[i] = 1

        for ed in self.edge:
            self.pn.append(len(ed))
            for p in ed:
                self.shapex.append(p[0])
                self.shapey.append(p[1])

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos) + 1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod
        
        return total
    

    def rotate(self, points, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        # print(points.shape)
        points[:,0] -= points[:,0].min()
        points[:,1] -= points[:,1].min()
        mx = points[:,0].max()
        my = points[:,1].max()

        points[:,0] -= 0.5 * mx + 0.5
        points[:,1] -= 0.5 * my + 0.5
        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += eps
        points[1,:] += eps
        # points[0,:] += 0.5 * mx - 0.5 + eps
        # points[1,:] += 0.5 * my - 0.5 + eps
        # points = np.round(points).astype(np.int)
        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        # if xmin < 0:
        #     points[0] += 1
        # if ymin < 0:
        #     points[1] += 1

        return points.transpose(), np.array([[xmin, ymin], [xmax, ymax]])

    '''
        check the state
        0 represent not accessible
        1 represent accessible
    '''

    def check(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route
    
    def check_s(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly_rot(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route

    def equal(self, a, b):
        return fabs(a[0]-b[0])+fabs(a[1]-b[1]) < 1e-3
            
    def route_length(self, route):
        res = 0
        lx = None
        ly = None
        for _ in route:
            x,y,_,_ = _
            if lx is not None:
                res += abs(lx-x) + abs(ly-y)
            lx = x 
            ly = y
        return res
    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -1
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]
            
            self.last_steps = steps

            if steps == 0:
                return -500, -1
            
            else:
                base += -0.5 * log(steps) / log(10)
                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 0
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 2, 0
                    else:
                        self.finished[index] = 1
                        return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route
            length = self.route_length(self.route)
            if length != 0:
                base += -0.5 * log(length) / log(10)
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 0
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 2, 0
                else:
                    self.finished[index] = 1
                    return base + 4, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_s(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            # base += -60
            base += -600
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        # return base + 50, 0
                        return base + 500, 0

                    else:
                        self.finished[index] = 1
                        return base + 2000, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                # return -40, 0
                return -600, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 100000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    # return base + 50, 0
                    return base + 500, 0
                else:
                    self.finished[index] = 1
                    return base + 2000, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_p(self, index, direction):
        base = -1
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 1
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 2, 0
                    else:
                        self.finished[index] = 1
                        return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 1
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 2, 0
                else:
                    self.finished[index] = 1
                    return base + 4, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    
    def move_debug(self, index, direction):
        base = -10
        if index >= len(self.pos):
            print('index error')
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape,bbox = self.getitem(index, cstate)

        if pos == target and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            print('xx',xx,'yy',yy)
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] :
                    steps -= 1
                    print('early done')
                    break
                    # return -500, -1
                
                flag = True
            
                for p in shape:
                    if x+p[0] < 0 or x+p[0] >= map_size[0] or y+p[1] < 0 or y+p[1] >= map_size[1]:
                        flag = False
                        print('over bound',x+p[0],)
                        break
                    if self.map[x+p[0],y+p[1]] != 0:
                        print('collide!!',self.map[x+p[0],y+p[1]],index+2)
                        flag = False
                        break

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                print('cannot move error')
                return -500, -1
            
            else:
                for p in shape:
                    self.map[xx+p[0],yy+p[1]] = 0
                
                for p in shape:
                    self.map[x+p[0],y+p[1]] = index + 2

                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if int(x) == int(target[0]) and int(y) == int(target[1]) and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target and cstate == tstate:
            print('position error')
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            tshape,bbox = self.getitem(index,tstate)
            for p in shape:
                self.map[xx+p[0],yy+p[1]] = 0
            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                self.map[x+p[0], y+p[1]] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            print('path find error')
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')
    
    def build_wall(self):
        size = self.map_size
        p = 0.005
        q = 0.05
        dix = [1,0,-1,0]
        diy = [0,1,0,-1]
        while True:
            px = np.random.randint(size[0])
            py = np.random.randint(size[1])
            if fabs(px-size[0]/2) + fabs(py-size[1]/2) > (size[0]+size[1])/4:
                break
        d = int(px < size[0]/2) + int(py < size[1]/2)*2
        if d == 3:
            d = 2
        elif d == 2:
            d = 3

        l = 0
        wall_list = []
        while True:
            cnt = 0
            while True: 
                tx = px + dix[d]
                ty = py + diy[d]
                cnt += 1
                if cnt > 10:
                    break
                if tx == size[0] or tx == -1 or ty == size[1] or ty == -1:
                    d = np.random.randint(4)
                    continue
            
                break

            if tx >= 0 and tx < size[0] and ty >= 0 and ty < size[1]:
                px = tx
                py = ty
                if not (px,py) in wall_list :
                    wall_list.append((px,py))

            r = np.random.rand()
            
            if r < q:
                t = np.random.rand()
                if t < 0.9:
                    d = (d+1)%4
                else:
                    d = np.random.randint(4)
            
            r = np.random.rand()
            if r < p and l > 100:
                break
            
            l += 1

        return wall_list

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        while True:
            reboot = False
            pos_list = []
            size_list = []
            target_list = []
            # self.finished = np.zeros(num)
            
            tmp = self.build_wall()
            # tmp += self.build_wall()
            # tmp += self.build_wall()
            sorted(tmp)
            wall_list = []
            l = None

            map_ = np.zeros(self.map_size)
            for t in tmp:
                if t != l:
                    l = t
                    map_[l[0],l[1]] = 1
                    wall_list.append(t)

            

            def dfs(p,tp):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]

                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
                        

            for i in range(num):
                while True:
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    flag = True

                    hh = [-1,0,1]
                    for _ in pos_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx - px) + abs(dy - py) < 5:
                            flag = False
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break


                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    if flag:
                        pos_list.append((px,py))
                        break
                
                reboot_cnt = 0
                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    # flag = True
                    
                    flag = dfs((px,py),pos_list[-1])
                    # print(flag)
                    for _ in target_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                            flag = False
                            break

                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break
                    
                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    
                    if flag:
                        target_list.append((px,py))
                        break
                
                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue
            # random.shuffle(pos_list)
            # random.shuffle(target_list)

            sub_num = self.max_num - num
            # print(self.max_num, num)
            pos_s = [ (_, i) for i, _ in enumerate(pos_list)]
            

            for _ in range(sub_num):
                pos_s.append(((0,0),-1))
            # print(len(pos_list))
            random.shuffle(pos_list)
            pos_list = np.array([_[0] for _ in pos_s])
            tmp_target_list = np.zeros_like(pos_list)
            # random.shuffle(target_list)
            _ = 0

            # for target in target_list:
            #     while pos_list[_][0] == pos_list[_][1] and pos_list[_][0] == 0:
            #         _ += 1
            #     tmp_target_list[_] = target
            #     _ += 1
            
            for i,_ in enumerate(pos_s):
                p, id_ = _
                if p[0] == p[1] and p[0] == 0:
                    continue

                tmp_target_list[i] = target_list[id_]

            target_list = tmp_target_list
            
            def dfs(p,tp,size):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                
                dx, dy = size
                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x + dx - 1 < self.map_size[0] and y >= 0 and y + dy - 1 < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            flag = True
                            for j in range(dx):
                                for k in range(dy):
                                    if map_[x+j,y+k] == 1:
                                        flag = False
                                        break
                                if flag == False:
                                    break

                            if flag:
                                stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
            for i in range(self.max_num):
                px, py = pos_list[i]
                tx, ty = target_list[i]

                if px == py and py == 0:
                    size_list.append((0,0))
                    continue

                reboot_cnt = 0

                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    dx = np.random.randint(2,31)
                    dy = np.random.randint(2,31)
                    if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                        # print(px,py,dx,dy)
                        continue

                    flag = True
                    
                    for j in range(self.max_num):
                        if j == i:
                            continue

                        px_, py_ = pos_list[j]
                        tx_, ty_ = target_list[j]

                        if px_ == py_ and py_ == 0:
                            # size_list.append((0,0))
                            continue

                        if j > len(size_list) - 1:
                            dx_, dy_ = 2, 2
                        else:
                            dx_, dy_ = size_list[j]

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    # if flag:
                    #     flag = dfs(pos_list[i],target_list[i],(dx,dy))

                    for wall in wall_list:
                        if not flag:
                            break

                        px_, py_ = wall
                        tx_, ty_ = wall
                        
                        dx_, dy_ = 1, 1
                        

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    if flag:
                        size_list.append((dx,dy))
                        break

                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue

            shapes = []
            for i in range(self.max_num):
                x,y = pos_list[i]
                w,d = size_list[i]
                pos_list[i] = (x+0.5*w,y+0.5*d)
                x,y = target_list[i]
                target_list[i] = (x+0.5*w,y+0.5*d)
                shape = []
                for a in range(w):
                    for b in range(d):
                        shape.append((a,b))

                shapes.append(shape)


            self.setmap(pos_list,target_list,shapes,np.zeros(len(pos_list)),np.zeros(len(pos_list)),wall_list)

            break
    
    def getstate_1(self,shape=[64,64]):
        state = []
        # temp = np.zeros(shape).astype(np.float32)
        # tmap = self.map == 1
        # oshape = self.map.shape
        # for i in range(shape[0]):
        #     x = int(1.0 * i * oshape[0] / shape[0])
        #     for j in range(shape[1]):
        #         y = int(1.0 * j * oshape[1] / shape[1])
        #         temp[i,j] = tmap[x,y]
        # for i in range(oshape[0]):
        #     x = 1.0*i/oshape[0]*shape[0]
        #     for j in range(oshape[1]):
        #         y = 1.0*j/oshape[1]*shape[1]
        #         temp[x,y] = tmap[i,j]

        state.append((self.map == 1).astype(np.float32))
        # state.append(temp)

        for i in range(len(self.pos)):
        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)

        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.target_map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getstate_3(self,shape=[64,64]):
        state = []

        state.append(np.array(self.map).astype(np.float32))
        state.append(np.array(self.target_map).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.shape,self.cstate,self.tstate,self.wall)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):

        num = len(self.finished)
        res = np.zeros([num,num,2])
        # size = np.zeros([num,2])
        # for i,shape in enumerate(self.shape):
        #     size[i]
        return res  
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res

    def getitem(self, index, state):
        shape = self.shape[index]
        opoints = np.array(shape)
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))
        
        radian = 2 * pi * state / self.bin
        points, bbox = self.rotate(opoints, radian)
        return points, bbox

    def getcleanmap(self, index):
        start = self.pos[index]
        # size = self.size[index]
        cstate = self.cstate[index]
        res = np.array(self.map)
        # shape = self.shape[index]
        # opoints = np.array(shape)
        # opoints = []
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))

        points, bbox = self.getitem(index, cstate)
        # radian = 2 * pi * cstate / self.bin
        # points, bbox = self.rotate(opoints, radian)
        xx, yy = start
        # xx = max(0, xx - 0.5*(bbox[1,0]-bbox[0,0]+1))
        # yy = max(0, yy - 0.5*(bbox[1,1]-bbox[0,1]+1))

        for p in points:
            x, y = p
            x += xx
            y += yy
            if x >= self.map_size[0] or y >= self.map_size[0]:
                print('wrong cords',x,y)

            tx = max(0,min(x,self.map_size[0])) + EPS
            ty = max(0,min(y,self.map_size[1])) + EPS
            res[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0
            # x = min(x, self.map_size[0]-1)
            # y = min(y, self.map_size[1]-1)
            # res[int(x), int(y)] = 0

        return res

    def __deepcopy__(self,memodict={}):
        res = ENV_ablation_wo_repetition(self.map_size,self.max_num)
        res.map = np.array(self.map)
        res.target_map = np.array(self.target_map)
        res.pos = np.array(self.pos)
        res.target = self.target
        res.shape = self.shape
        res.cstate = np.array(self.cstate)
        res.tstate = self.tstate
        res.wall = self.wall
        res.state_dict = deepcopy(self.state_dict)
        res.finished = np.array(self.finished)
        res.shapex = self.shapex
        res.shapey = self.shapey
        res.pn = self.pn
        res.edge = self.edge
        res.bound = self.bound

        return res
   
class ENV_ablation_wo_multi:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5),max_num=5):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        self.bin = 24
        self.max_num = max_num
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, shape, cstate, tstate, wall=[]):
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(np.array(pos))
        self.shape = deepcopy(shape)
        self.target = np.array(target)
        self.cstate = np.array(cstate).astype(np.int)
        self.tstate = np.array(tstate).astype(np.int)
        self.wall = deepcopy(wall)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        self.shapex = []
        self.shapey = []
        # self.boundx = []
        # self.boundy = []
        self.pn = []
        # self.bn = []
        self.edge = []
        self.bound = []
        
        # print(shape)

        
        
        # for bd in self.bound:
        
        # for sh in self.shape:
        #     print(len(sh))
        # for p in self.pos:
        #     print(p)
        
        # print()

        for p in wall:
            x, y = p
            x = int(x)
            y = int(y)
            self.map[x,y] = 1
            self.target_map[x,y] = 1

        # cut_list = []
        for i, _ in enumerate(pos):
            if len(shape[i]) == 0:
                continue
            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))
                
                if self.map[int(tx),int(ty)] >= 1 and self.map[int(tx),int(ty)] != i+2:
                    # if i == 5:
                    #     print('map',self.map[int(tx),int(ty)])

                    id_list.append(i_)
                else:
                    self.map[int(tx),int(ty)] = i + 2

                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            

            # for p in points:
            #     xx, yy = p
            #     xmax = max(xmax,xx)
            #     xmin = min(xmin,xx)
            #     ymax = max(ymax,yy)
            #     ymin = min(ymin,yy)
            
            # x = max(0,x - 0.5*(xmax-xmin+1))
            # y = max(0,y - 0.5*(ymax-ymin+1))
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                # cut_list.append((int(tx),int(ty)))

                if self.target_map[int(tx),int(ty)] >= 1 and self.target_map[int(tx),int(ty)] != i + 2:
                    # if i == 5:
                    #     print('tmap',self.map[int(tx),int(ty)])

                    if not i_ in id_list:
                        id_list.append(i_)
                else:
                    self.target_map[int(tx),int(ty)] = i + 2
                
                if self.map[int(tx),int(ty)] == 1:
                    self.map[int(tx),int(ty)] = 0
                if self.target_map[int(tx),int(ty)] == 1:
                    self.target_map[int(tx),int(ty)] = 0
            
            tmp = deepcopy(self.shape[i])
            # print(self.shape[i])
            for i_ in id_list:
                # print(i_,tmp[i_])
                self.shape[i].remove(tmp[i_])
            # dx, dy = size[i]
            # x, y = _
            # x_, y_ = target[i]
            # self.map[x:x+dx, y:y+dy] = i + 2
            # self.target_map[x_:x_+dx, y_:y_+dy] = i + 2
        # print('==========')
        # for sh in self.shape:
        #     print(len(sh))
        self.map = (self.map == 1).astype(np.int32)
        self.target_map = (self.target_map == 1).astype(np.int32)

        for i, _ in enumerate(pos):
            if len(self.shape[i]) == 0:
                continue

            x, y = _
            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            id_list = []
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.map[int(tx),int(ty)] = i + 2
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            
            for i_,p in enumerate(points):
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0])) + EPS
                ty = max(0,min(y+yy,self.map_size[1])) + EPS
                
                self.target_map[int(tx),int(ty)] = i + 2
                


        for i, sh in enumerate(self.shape):
            

            ed = []
            # bd = []
            map_ = np.zeros(self.map_size)
            mark_ = np.zeros(self.map_size)
            
            for p in sh:
                map_[p[0],p[1]] = 1
            
            # print(i,sh)
            def dfs(p):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                rx = [1,1,1,0,0,-1,-1,-1]
                ry = [-1,0,1,-1,1,-1,0,1]
                stack = []
                stack.append((p,-1))
                last = -2
                
                

                while stack.__len__() != 0:
                    flag = True
                    p,direction = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    # print(p)
                    for i in range(8):
                        x = p[0] + rx[i]
                        y = p[1] + ry[i]

                        if x < 0 or x >= self.map_size[0] or y >= self.map_size[1] or y < 0 or map_[x,y] == 0:
                            # if last == direction:
                            #     ed.pop()

                            ed.append(p)
                            # bd.append(p)
                            last = direction
                            flag = False
                            break

                    vs = []
                    for i in range(8):
                        vs.append([])

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] == 1 and mark_[x,y] == 0:
                            # print(x,y)
                            # dfs((x,y))
                            if flag:
                                # mark_[x,y] = 1
                                stack.append(((x,y),i))
                                break
                            else:
                                cnt = 0
                                for j in range(8):
                                    xx = x + rx[j]
                                    yy = y + ry[j]
                                    if xx < 0 or xx >= self.map_size[0] or yy >= self.map_size[1] or yy < 0 or map_[xx,yy] == 0:
                                        # mark_[x,y] = 1
                                        cnt += 1

                                vs[cnt].append(((x,y),i))
                    
                    for v in vs:
                        for p in v:
                            stack.append(p)
                    
            if len(sh) != 0:
                dfs(sh[0])

            self.edge.append(ed)
            # self.bound.append(bd)
        # for i in range(len(self.pos)):
        #     print(len(self.shape[i]),len(self.edge[i]))

        for i, _ in enumerate(pos):
            if self.equal(self.pos[i],self.target[i]) and self.cstate[i] == self.tstate[i]:
                self.finished[i] = 1

        for ed in self.edge:
            self.pn.append(len(ed))
            for p in ed:
                self.shapex.append(p[0])
                self.shapey.append(p[1])

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos) + 1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod
        
        return total
    

    def rotate(self, points, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        # print(points.shape)
        points[:,0] -= points[:,0].min()
        points[:,1] -= points[:,1].min()
        mx = points[:,0].max()
        my = points[:,1].max()

        points[:,0] -= 0.5 * mx + 0.5
        points[:,1] -= 0.5 * my + 0.5
        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += eps
        points[1,:] += eps
        # points[0,:] += 0.5 * mx - 0.5 + eps
        # points[1,:] += 0.5 * my - 0.5 + eps
        # points = np.round(points).astype(np.int)
        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        # if xmin < 0:
        #     points[0] += 1
        # if ymin < 0:
        #     points[1] += 1

        return points.transpose(), np.array([[xmin, ymin], [xmax, ymax]])

    '''
        check the state
        0 represent not accessible
        1 represent accessible
    '''

    def check(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route
    
    def check_s(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly_rot(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route

    def equal(self, a, b):
        return fabs(a[0]-b[0])+fabs(a[1]-b[1]) < 1e-3
            
    def route_length(self, route):
        res = 0
        lx = None
        ly = None
        for _ in route:
            x,y,_,_ = _
            if lx is not None:
                res += abs(lx-x) + abs(ly-y)
            lx = x 
            ly = y
        return res
    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -1
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]
            
            self.last_steps = steps

            if steps == 0:
                return -500, -1
            
            else:
                base += -0.5 * log(steps) / log(10)
                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 1
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    # if self.finished[index] == 1:
                    #     return base + 2, 0
                    # else:
                    self.finished[index] = 1
                    return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route
            length = self.route_length(self.route)
            if length != 0:
                base += -0.5 * log(length) / log(10)
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 1
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                # if self.finished[index] == 1: # if the object was placed before
                #     return base + 2, 0
                # else:
                self.finished[index] = 1
                return base + 4, 0
                # return base + 500, 0
                # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_s(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            # base += -60
            base += -600
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        # return base + 50, 0
                        return base + 500, 0

                    else:
                        self.finished[index] = 1
                        return base + 2000, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                # return -40, 0
                return -600, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 100000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    # return base + 50, 0
                    return base + 500, 0
                else:
                    self.finished[index] = 1
                    return base + 2000, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    def move_p(self, index, direction):
        base = -1
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
                base += -4
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    penlty = self.state_dict[hash_]
                    self.state_dict[hash_] += 1
                    penlty = 1
                    return base-2 * penlty, 0
                
                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 2, 0
                    else:
                        self.finished[index] = 1
                        return base + 4, 0
                        # return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check_s(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0]-1)) + EPS
                ty = max(0,min(y+p[1],self.map_size[1]-1)) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                penlty = self.state_dict[hash_]
                self.state_dict[hash_] += 1
                penlty = 1
                return base -2 * penlty, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                # return base + 10000, 1
                return base + 50, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 2, 0
                else:
                    self.finished[index] = 1
                    return base + 4, 0
                    # return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            # return -500, -1
            return base, -1

    
    def move_debug(self, index, direction):
        base = -10
        if index >= len(self.pos):
            print('index error')
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape,bbox = self.getitem(index, cstate)

        if pos == target and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            print('xx',xx,'yy',yy)
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] :
                    steps -= 1
                    print('early done')
                    break
                    # return -500, -1
                
                flag = True
            
                for p in shape:
                    if x+p[0] < 0 or x+p[0] >= map_size[0] or y+p[1] < 0 or y+p[1] >= map_size[1]:
                        flag = False
                        print('over bound',x+p[0],)
                        break
                    if self.map[x+p[0],y+p[1]] != 0:
                        print('collide!!',self.map[x+p[0],y+p[1]],index+2)
                        flag = False
                        break

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                print('cannot move error')
                return -500, -1
            
            else:
                for p in shape:
                    self.map[xx+p[0],yy+p[1]] = 0
                
                for p in shape:
                    self.map[x+p[0],y+p[1]] = index + 2

                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if int(x) == int(target[0]) and int(y) == int(target[1]) and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target and cstate == tstate:
            print('position error')
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            tshape,bbox = self.getitem(index,tstate)
            for p in shape:
                self.map[xx+p[0],yy+p[1]] = 0
            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                self.map[x+p[0], y+p[1]] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            print('path find error')
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')
    
    def build_wall(self):
        size = self.map_size
        p = 0.005
        q = 0.05
        dix = [1,0,-1,0]
        diy = [0,1,0,-1]
        while True:
            px = np.random.randint(size[0])
            py = np.random.randint(size[1])
            if fabs(px-size[0]/2) + fabs(py-size[1]/2) > (size[0]+size[1])/4:
                break
        d = int(px < size[0]/2) + int(py < size[1]/2)*2
        if d == 3:
            d = 2
        elif d == 2:
            d = 3

        l = 0
        wall_list = []
        while True:
            cnt = 0
            while True: 
                tx = px + dix[d]
                ty = py + diy[d]
                cnt += 1
                if cnt > 10:
                    break
                if tx == size[0] or tx == -1 or ty == size[1] or ty == -1:
                    d = np.random.randint(4)
                    continue
            
                break

            if tx >= 0 and tx < size[0] and ty >= 0 and ty < size[1]:
                px = tx
                py = ty
                if not (px,py) in wall_list :
                    wall_list.append((px,py))

            r = np.random.rand()
            
            if r < q:
                t = np.random.rand()
                if t < 0.9:
                    d = (d+1)%4
                else:
                    d = np.random.randint(4)
            
            r = np.random.rand()
            if r < p and l > 100:
                break
            
            l += 1

        return wall_list

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        while True:
            reboot = False
            pos_list = []
            size_list = []
            target_list = []
            # self.finished = np.zeros(num)
            
            tmp = self.build_wall()
            # tmp += self.build_wall()
            # tmp += self.build_wall()
            sorted(tmp)
            wall_list = []
            l = None

            map_ = np.zeros(self.map_size)
            for t in tmp:
                if t != l:
                    l = t
                    map_[l[0],l[1]] = 1
                    wall_list.append(t)

            

            def dfs(p,tp):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]

                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
                        

            for i in range(num):
                while True:
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    flag = True

                    hh = [-1,0,1]
                    for _ in pos_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx - px) + abs(dy - py) < 5:
                            flag = False
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break


                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    if flag:
                        pos_list.append((px,py))
                        break
                
                reboot_cnt = 0
                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    px = np.random.randint(1,size[0]-3)
                    py = np.random.randint(1,size[1]-3)
                    # flag = True
                    
                    flag = dfs((px,py),pos_list[-1])
                    # print(flag)
                    for _ in target_list:
                        dx, dy = _
                        if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                            flag = False
                            break

                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break
                    
                    for _ in wall_list:
                        if flag == False:
                            break
                        dx,dy = _
                        if dx == px and dy == py:
                            flag = False
                            break
                        
                        for k in hh:
                            for l in hh:
                                xx = px + k
                                yy = py + l
                                if dx == xx and dy == yy:
                                    flag = False
                                    break
                                    
                            if flag == False:
                                break

                        if flag == False:
                            break

                    
                    if flag:
                        target_list.append((px,py))
                        break
                
                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue
            # random.shuffle(pos_list)
            # random.shuffle(target_list)

            sub_num = self.max_num - num
            # print(self.max_num, num)
            pos_s = [ (_, i) for i, _ in enumerate(pos_list)]
            

            for _ in range(sub_num):
                pos_s.append(((0,0),-1))
            # print(len(pos_list))
            random.shuffle(pos_list)
            pos_list = np.array([_[0] for _ in pos_s])
            tmp_target_list = np.zeros_like(pos_list)
            # random.shuffle(target_list)
            _ = 0

            # for target in target_list:
            #     while pos_list[_][0] == pos_list[_][1] and pos_list[_][0] == 0:
            #         _ += 1
            #     tmp_target_list[_] = target
            #     _ += 1
            
            for i,_ in enumerate(pos_s):
                p, id_ = _
                if p[0] == p[1] and p[0] == 0:
                    continue

                tmp_target_list[i] = target_list[id_]

            target_list = tmp_target_list
            
            def dfs(p,tp,size):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                
                dx, dy = size
                stack = []
                stack.append(p)
                
                mark_ = np.zeros(self.map_size)

                while stack.__len__() != 0:
                    flag = True
                    p = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    if p[0] == tp[0] and p[1] == tp[1]:
                        break

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x + dx - 1 < self.map_size[0] and y >= 0 and y + dy - 1 < self.map_size[1] and map_[x,y] != 1 and mark_[x,y] == 0:
                            flag = True
                            for j in range(dx):
                                for k in range(dy):
                                    if map_[x+j,y+k] == 1:
                                        flag = False
                                        break
                                if flag == False:
                                    break

                            if flag:
                                stack.append((x,y))
                            

                return mark_[tp[0],tp[1]] == 1
            
            for i in range(self.max_num):
                px, py = pos_list[i]
                tx, ty = target_list[i]

                if px == py and py == 0:
                    size_list.append((0,0))
                    continue

                reboot_cnt = 0

                while reboot_cnt < 10000:
                    reboot_cnt += 1
                    dx = np.random.randint(2,31)
                    dy = np.random.randint(2,31)
                    if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                        # print(px,py,dx,dy)
                        continue

                    flag = True
                    
                    for j in range(self.max_num):
                        if j == i:
                            continue

                        px_, py_ = pos_list[j]
                        tx_, ty_ = target_list[j]

                        if px_ == py_ and py_ == 0:
                            # size_list.append((0,0))
                            continue

                        if j > len(size_list) - 1:
                            dx_, dy_ = 2, 2
                        else:
                            dx_, dy_ = size_list[j]

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    # if flag:
                    #     flag = dfs(pos_list[i],target_list[i],(dx,dy))

                    for wall in wall_list:
                        if not flag:
                            break

                        px_, py_ = wall
                        tx_, ty_ = wall
                        
                        dx_, dy_ = 1, 1
                        

                        lx = max(px, px_)
                        ly = max(py, py_)
                        rx = min(px + dx, px_ + dx_) - 1
                        ry = min(py + dy, py_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                        lx = max(tx, tx_)
                        ly = max(ty, ty_)
                        rx = min(tx + dx, tx_ + dx_) - 1
                        ry = min(ty + dy, ty_ + dy_) - 1
                        if lx <= rx and ly <= ry:
                            flag = False
                            break

                    if flag:
                        size_list.append((dx,dy))
                        break

                if reboot_cnt >= 10000:
                    reboot = True
                    break
            
            if reboot:
                continue

            shapes = []
            for i in range(self.max_num):
                x,y = pos_list[i]
                w,d = size_list[i]
                pos_list[i] = (x+0.5*w,y+0.5*d)
                x,y = target_list[i]
                target_list[i] = (x+0.5*w,y+0.5*d)
                shape = []
                for a in range(w):
                    for b in range(d):
                        shape.append((a,b))

                shapes.append(shape)


            self.setmap(pos_list,target_list,shapes,np.zeros(len(pos_list)),np.zeros(len(pos_list)),wall_list)

            break
    
    def getstate_1(self,shape=[64,64]):
        state = []
        # temp = np.zeros(shape).astype(np.float32)
        # tmap = self.map == 1
        # oshape = self.map.shape
        # for i in range(shape[0]):
        #     x = int(1.0 * i * oshape[0] / shape[0])
        #     for j in range(shape[1]):
        #         y = int(1.0 * j * oshape[1] / shape[1])
        #         temp[i,j] = tmap[x,y]
        # for i in range(oshape[0]):
        #     x = 1.0*i/oshape[0]*shape[0]
        #     for j in range(oshape[1]):
        #         y = 1.0*j/oshape[1]*shape[1]
        #         temp[x,y] = tmap[i,j]

        state.append((self.map == 1).astype(np.float32))
        # state.append(temp)

        for i in range(len(self.pos)):
        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)

        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.target_map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getstate_3(self,shape=[64,64]):
        state = []

        state.append(np.array(self.map).astype(np.float32))
        state.append(np.array(self.target_map).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.shape,self.cstate,self.tstate,self.wall)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):

        num = len(self.finished)
        res = np.zeros([num,num,2])
        # size = np.zeros([num,2])
        # for i,shape in enumerate(self.shape):
        #     size[i]
        return res  
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res

    def getitem(self, index, state):
        shape = self.shape[index]
        opoints = np.array(shape)
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))
        
        radian = 2 * pi * state / self.bin
        points, bbox = self.rotate(opoints, radian)
        return points, bbox

    def getcleanmap(self, index):
        start = self.pos[index]
        # size = self.size[index]
        cstate = self.cstate[index]
        res = np.array(self.map)
        # shape = self.shape[index]
        # opoints = np.array(shape)
        # opoints = []
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))

        points, bbox = self.getitem(index, cstate)
        # radian = 2 * pi * cstate / self.bin
        # points, bbox = self.rotate(opoints, radian)
        xx, yy = start
        # xx = max(0, xx - 0.5*(bbox[1,0]-bbox[0,0]+1))
        # yy = max(0, yy - 0.5*(bbox[1,1]-bbox[0,1]+1))

        for p in points:
            x, y = p
            x += xx
            y += yy
            if x >= self.map_size[0] or y >= self.map_size[0]:
                print('wrong cords',x,y)

            tx = max(0,min(x,self.map_size[0])) + EPS
            ty = max(0,min(y,self.map_size[1])) + EPS
            res[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0
            # x = min(x, self.map_size[0]-1)
            # y = min(y, self.map_size[1]-1)
            # res[int(x), int(y)] = 0

        return res

    def __deepcopy__(self,memodict={}):
        res = ENV_ablation_wo_multi(self.map_size,self.max_num)
        res.map = np.array(self.map)
        res.target_map = np.array(self.target_map)
        res.pos = np.array(self.pos)
        res.target = self.target
        res.shape = self.shape
        res.cstate = np.array(self.cstate)
        res.tstate = self.tstate
        res.wall = self.wall
        res.state_dict = deepcopy(self.state_dict)
        res.finished = np.array(self.finished)
        res.shapex = self.shapex
        res.shapey = self.shapey
        res.pn = self.pn
        res.edge = self.edge
        res.bound = self.bound

        return res
   


class ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly_norepeat_reward:  # if current state happened before, give a penalty.
    def __init__(self, size=(5,5),max_num=5):
        self.map_size = size
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.route = []
        self.bin = 24
        self.max_num = max_num
        
    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, pos, target, shape, cstate, tstate, wall=[]):
        self.map = np.zeros(self.map_size)
        self.target_map = np.zeros(self.map_size)
        self.pos = deepcopy(pos)
        self.shape = deepcopy(shape)
        self.target = deepcopy(target)
        self.cstate = deepcopy(np.array(cstate)).astype(np.int)
        self.tstate = deepcopy(np.array(tstate)).astype(np.int)
        self.wall = deepcopy(wall)
        self.dis = np.zeros(len(pos))
        self.route = []
        self.state_dict = {}
        self.finished = np.zeros(len(pos))
        self.shapex = []
        self.shapey = []
        # self.boundx = []
        # self.boundy = []
        self.pn = []
        # self.bn = []
        self.edge = []
        self.bound = []
        
        

        for sh in self.shape:
            ed = []
            # bd = []
            map_ = np.zeros(self.map_size)
            mark_ = np.zeros(self.map_size)
            
            for p in sh:
                map_[p[0],p[1]] = 1
            
            def dfs(p):
                gx = [1,0,-1,0]
                gy = [0,1,0,-1]
                rx = [1,1,1,0,0,-1,-1,-1]
                ry = [-1,0,1,-1,1,-1,0,1]
                stack = []
                stack.append((p,-1))
                last = -2
                
                

                while stack.__len__() != 0:
                    flag = True
                    p,direction = stack.pop()
                    if mark_[p[0],p[1]] == 1:
                        continue

                    mark_[p[0],p[1]] = 1
                    
                    # print(p)
                    for i in range(8):
                        x = p[0] + rx[i]
                        y = p[1] + ry[i]

                        if x < 0 or x >= self.map_size[0] or y >= self.map_size[1] or y < 0 or map_[x,y] == 0:
                            if last == direction:
                                ed.pop()

                            ed.append(p)
                            # bd.append(p)
                            last = direction
                            flag = False
                            break

                    vs = []
                    for i in range(8):
                        vs.append([])

                    for i in range(4):
                        x = p[0] + gx[i]
                        y = p[1] + gy[i]
                        
                        if x >= 0 and x < self.map_size[0] and y >= 0 and y < self.map_size[1] and map_[x,y] == 1 and mark_[x,y] == 0:
                            # print(x,y)
                            # dfs((x,y))
                            if flag:
                                # mark_[x,y] = 1
                                stack.append(((x,y),i))
                                break
                            else:
                                cnt = 0
                                for j in range(8):
                                    xx = x + rx[j]
                                    yy = y + ry[j]
                                    if xx < 0 or xx >= self.map_size[0] or yy >= self.map_size[1] or yy < 0 or map_[xx,yy] == 0:
                                        # mark_[x,y] = 1
                                        cnt += 1

                                vs[cnt].append(((x,y),i))
                    
                    for v in vs:
                        for p in v:
                            stack.append(p)
                    
                    
            if len(sh) > 0:
                dfs(sh[0])

            self.edge.append(ed)
            # self.bound.append(bd)
        # for i in range(len(self.pos)):
        #     print(len(self.shape[i]),len(self.edge[i]))

        for ed in self.edge:
            self.pn.append(len(ed))
            for p in ed:
                self.shapex.append(p[0])
                self.shapey.append(p[1])
        
        # for bd in self.bound:
            

        for p in wall:
            x, y = p
            if x < 0 or x >= self.map_size[0] or y < 0 or y >= self.map_size[1]:
                print(x,y)
            self.map[x,y] = 1
            self.target_map[x,y] = 1


        for i, _ in enumerate(pos):
            x, y = _
            if len(self.shape[i]) == 0:
                continue

            s = self.cstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            
            for p in points:
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0]-1)) + EPS
                ty = max(0,min(y+yy,self.map_size[1]-1)) + EPS
                self.map[int(tx),int(ty)] = i + 2
           
            x,y = target[i]
            s = self.tstate[i]
            radian = 2 * pi * s / self.bin
            points, bbox = self.rotate(self.shape[i], radian)
            

            # for p in points:
            #     xx, yy = p
            #     xmax = max(xmax,xx)
            #     xmin = min(xmin,xx)
            #     ymax = max(ymax,yy)
            #     ymin = min(ymin,yy)
            
            # x = max(0,x - 0.5*(xmax-xmin+1))
            # y = max(0,y - 0.5*(ymax-ymin+1))
            
            
            for p in points:
                xx,yy = p
                tx = max(0,min(x+xx,self.map_size[0]-1)) + EPS
                ty = max(0,min(y+yy,self.map_size[1]-1)) + EPS
                self.target_map[int(tx),int(ty)] = i + 2
                
            # dx, dy = size[i]
            # x, y = _
            # x_, y_ = target[i]
            # self.map[x:x+dx, y:y+dy] = i + 2
            # self.target_map[x_:x_+dx, y_:y_+dy] = i + 2

        hash_ = self.hash()
        self.state_dict[hash_] = 1
        # self.check()
    
    
    def hash(self):
        mod = 1000000007
        num = len(self.pos) + 1
        total = 0
        for row in self.map:
            for _ in row:
                total *= num
                total += int(_)
                total %= mod
        
        return total
    

    def rotate(self, points, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        points[:,0] -= points[:,0].min()
        points[:,1] -= points[:,1].min()
        mx = points[:,0].max()
        my = points[:,1].max()

        points[:,0] -= 0.5 * mx + 0.5
        points[:,1] -= 0.5 * my + 0.5
        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += eps
        points[1,:] += eps
        # points[0,:] += 0.5 * mx - 0.5 + eps
        # points[1,:] += 0.5 * my - 0.5 + eps
        # points = np.round(points).astype(np.int)
        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        # if xmin < 0:
        #     points[0] += 1
        # if ymin < 0:
        #     points[1] += 1

        return points.transpose(), np.array([[xmin, ymin], [xmax, ymax]])

    '''
        check the state
        0 represent not accessible
        1 represent accessible
    '''

    def check(self, index):
        flag = True
        # tx,ty = self.target[index]
        # sx,sy = self.pos[index]
        # shape = self.shape[index]
        # cstate = self.cstate[index]
        # print(cstate)
        # tstate = self.tstate[index]

        route = []
        # ps = np.array(shape)
        # px = np.array(ps[:,0],dtype=np.int32)
        # py = np.array(ps[:,1],dtype=np.int32)
        # num = len(ps)
        lx = np.zeros([60000],dtype=np.float32)
        ly = np.zeros([60000],dtype=np.float32)
        lr = np.zeros([60000],dtype=np.int32)
        ld = np.zeros([60000],dtype=np.int32)
        length = np.array([0],dtype=np.int32)
        flag = np.array([1],dtype=np.int32)
        n,m = self.map_size

        tx_ar = np.zeros(len(self.target),dtype=np.float32)
        ty_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sx_ar = np.zeros_like(tx_ar,dtype=np.float32)
        sy_ar = np.zeros_like(tx_ar,dtype=np.float32)

        for i,p in enumerate(self.target):
            tx_ar[i] = p[0]
            ty_ar[i] = p[1]

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]
        
        tmp_map = np.array(self.map,dtype=np.int32)
        shape,bbox = self.getitem(index,self.cstate[index])
        sx,sy = self.pos[index]
        for p in shape:
            x,y = p
            x += sx + EPS
            y += sy + EPS
            tmp_map[min(int(x),self.map_size[0]-1),min(int(y),self.map_size[1]-1)] = 0

        # print('in')
        # print(index)
        # tx = int(tx)
        # ty = int(ty)
        # sx = int(sx)
        # sy = int(sy)
        # search_transpose(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy, cstate,tstate,self.bin,px,py,num,lx,ly,lr,ld,length,flag)
        search_transpose_poly(
            tmp_map, np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
            len(self.pos),index,
            tx_ar,ty_ar,
            sx_ar,sy_ar,
            np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
            lx, ly, lr, ld, length,
            flag
        )
        # print('out')
        # search(np.array(self.map,dtype=np.int32),n,m,index,tx,ty,sx,sy,h,w,lx,ly,length,flag)
        for i in range(length[0]):
            route.append((lx[i],ly[i],lr[i],ld[i]))
        route.reverse()
        return flag[0], route

    def equal(self, a, b):
        return fabs(a[0]-b[0])+fabs(a[1]-b[1]) < 1e-6
            

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index, direction):
        base = -10
        if index >= len(self.pos):
            return -500, -1
        if len(self.shape[index]) == 0:
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape, bbox = self.getitem(index, cstate)

        n,m = self.map_size

        sx_ar = np.zeros(len(self.target),dtype=np.float32)
        sy_ar = np.zeros_like(sx_ar,dtype=np.float32)
        steps_ar = np.zeros(1,dtype=np.int32)

        for i,p in enumerate(self.pos):
            sx_ar[i] = p[0]
            sy_ar[i] = p[1]

        if self.equal(pos, target) and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))
            
            
            translate(
                np.array(self.map,dtype=np.int32),np.array(self.shapex,dtype=np.float32),np.array(self.shapey,dtype=np.float32),np.array(self.pn,dtype=np.int32),n,m,
                len(self.pos),index,direction,
                sx_ar,sy_ar,
                np.array(self.cstate, dtype=np.int32), np.array(self.tstate, dtype=np.int32), self.bin,
                steps_ar
            )
            steps = steps_ar[0]

            x = xx + steps * _x[direction]
            y = yy + steps * _y[direction]

            if steps == 0:
                return -500, -1
            
            else:

                for p in shape:
                    self.map[min(int(xx+p[0]+EPS),self.map_size[0]-1),min(int(yy+p[1]+EPS),self.map_size[1]-1)] = 0

                for p in shape:
                    self.map[min(int(x+p[0]+EPS),self.map_size[0]-1),min(int(y+p[1]+EPS),self.map_size[1]-1)] = index + 2
                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                # x = x + 0.5*(bbox[1][0]-bbox[0][0]+1)
                # y = y + 0.5*(bbox[1][1]-bbox[0][1]+1)
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    # return -600, 0
                    return base, 0

                self.state_dict[hash_] = 1

                if fabs(x-target[0]) + fabs(y-target[1]) < 1e-6 and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if self.equal(pos, target) and cstate == tstate:
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            # xx = max(0,xx - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # yy = max(0,yy - 0.5*(bbox[1][1]-bbox[0][1]+1))

            tshape, bbox = self.getitem(index, tstate)

            # x = max(0,x - 0.5*(bbox[1][0]-bbox[0][0]+1))
            # y = max(0,y - 0.5*(bbox[1][1]-bbox[0][1]+1))

            for p in shape:
                tx = max(0,min(xx+p[0],self.map_size[0])) + EPS
                ty = max(0,min(yy+p[1],self.map_size[1])) + EPS
                self.map[min(int(tx),self.map_size[0]-1),min(int(ty),self.map_size[1]-1)] = 0

            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                tx = max(0,min(x+p[0],self.map_size[0])) + EPS
                ty = max(0,min(y+p[1],self.map_size[1])) + EPS
                self.map[min(int(tx),self.map_size[0]-1), min(int(ty),self.map_size[1]-1)] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                # return -40, 0
                return base, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if fabs(v[0]-w[0])+fabs(v[1]-w[1]) > 1e-6 or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            return -500, -1

    def move_debug(self, index, direction):
        base = -10
        if index >= len(self.pos):
            print('index error')
            return -500, -1

        map_size = self.map_size
        # size = self.size[index]
        target = self.target[index]
        pos = self.pos[index]
        cstate = self.cstate[index]
        tstate = self.tstate[index]
        shape,bbox = self.getitem(index, cstate)

        if pos == target and cstate == tstate: # prevent it from being trapped in local minima that moving object around the destination
            base += -60
        
        if direction < 4:
            xx, yy = pos
            print('xx',xx,'yy',yy)
            steps = 1
            while True:
                x = xx + steps*_x[direction]
                y = yy + steps*_y[direction]
                if x < 0 or x >= map_size[0] or y < 0 or y >= map_size[1] :
                    steps -= 1
                    print('early done')
                    break
                    # return -500, -1
                
                flag = True
            
                for p in shape:
                    if x+p[0] < 0 or x+p[0] >= map_size[0] or y+p[1] < 0 or y+p[1] >= map_size[1]:
                        flag = False
                        print('over bound',x+p[0],)
                        break
                    if self.map[x+p[0],y+p[1]] != 0:
                        print('collide!!',self.map[x+p[0],y+p[1]],index+2)
                        flag = False
                        break

                if not flag:
                    steps -= 1
                    break
                steps += 1

            x = xx + steps*_x[direction]
            y = yy + steps*_y[direction]

            if steps == 0:
                print('cannot move error')
                return -500, -1
            
            else:
                for p in shape:
                    self.map[xx+p[0],yy+p[1]] = 0
                
                for p in shape:
                    self.map[x+p[0],y+p[1]] = index + 2

                # self.map[xx:xx+size[0],yy:yy+size[1]] = 0
                # self.map[x:x+size[0],y:y+size[1]] = index + 2
                self.pos[index] = (x, y)
                
                hash_ = self.hash()          # get the hash value of the current state
                if hash_ in self.state_dict: # if the current state happened before
                    return -600, 0

                self.state_dict[hash_] = 1

                if x == target[0] and y == target[1] and cstate == tstate:
                    if self.finished[index] == 1:
                        return base + 50, 0
                    else:
                        self.finished[index] = 1
                        return base + 500, 0
                        # return base + 200, 0
                
                return base, 0

        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         self.map[target[0]+i, target[1]+j] = 1

        # if it has already reached the target place.
        if pos == target and cstate == tstate:
            print('position error')
            return -500, -1

        # check if legal, calc the distance
        flag, route = self.check(index)
        

        if flag == 1:
            xx, yy = pos
            x,y = target
            self.route = route

            tshape,bbox = self.getitem(index,tstate)
            for p in shape:
                self.map[xx+p[0],yy+p[1]] = 0
            # self.map[xx:xx+size[0], yy:yy+size[1]] = 0
            for p in tshape:
                self.map[x+p[0], y+p[1]] = index + 2

            self.pos[index] = target
            self.cstate[index] = tstate
            
            hash_ = self.hash()          # get the hash value of the current state
            if hash_ in self.state_dict: # if the current state happened before
                return -40, 0


            ff = True
            for i,v in enumerate(self.pos): # check if the task is done
                w = self.target[i]
                if v != w or self.cstate[i] != self.tstate[i]:
                    ff = False
                    break

            
            if ff:
                self.finished[index] = 1
                return base + 10000, 1
            else:
                if self.finished[index] == 1: # if the object was placed before
                    return base + 50, 0
                else:
                    self.finished[index] = 1
                    return base + 500, 0
                    # return base + 200, 0
        # elif flag == -1:
        #     return -1000, 1
        else:
            print('path find error')
            return -500, -1

    def getlastroute(self):
        return self.route

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')
    
    def build_wall(self):
        size = self.map_size
        p = 0.005
        q = 0.05
        dix = [1,0,-1,0]
        diy = [0,1,0,-1]
        while True:
            px = np.random.randint(size[0])
            py = np.random.randint(size[1])
            if fabs(px-size[0]/2) + fabs(py-size[1]/2) > (size[0]+size[1])/4:
                break
        d = int(px < size[0]/2) + int(py < size[1]/2)*2
        if d == 3:
            d = 2
        elif d == 2:
            d = 3

        l = 0
        wall_list = []
        while True:
            cnt = 0
            while True: 
                tx = px + dix[d]
                ty = py + diy[d]
                cnt += 1
                if cnt > 10:
                    break
                if tx == size[0] or tx == -1 or ty == size[1] or ty == -1:
                    d = np.random.randint(4)
                    continue
            
                break

            if tx >= 0 and tx < size[0] and ty >= 0 and ty < size[1]:
                px = tx
                py = ty
                if not (px,py) in wall_list :
                    wall_list.append((px,py))

            r = np.random.rand()
            
            if r < q:
                t = np.random.rand()
                if t < 0.9:
                    d = (d+1)%4
                else:
                    d = np.random.randint(4)
            
            r = np.random.rand()
            if r < p and l > 50:
                break
            
            l += 1

        return wall_list

    def randominit(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        

        for i in range(num):
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in pos_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(size[0])
                py = np.random.randint(size[1])
                flag = True

                for _ in target_list:
                    dx,dy = _
                    if dx == px or dy == py:
                        flag = False
                        break
                if flag:
                    target_list.append((px,py))
                    break
            
        random.shuffle(pos_list)
        random.shuffle(target_list)

        for i in range(num):
            px, py = pos_list[i]
            tx, ty = target_list[i]
            while True:
                dx = np.random.randint(1,11)
                dy = np.random.randint(1,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]
                    if j > len(size_list) - 1:
                        dx_, dy_ = 1, 1
                    else:
                        dx_, dy_ = size_list[j] 

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break

            
        # list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))
        # random.shuffle(list_)
        # for i in list_:
        #     pos_list.append(i[0])
        #     size_list.append(i[1])
        
        self.setmap(pos_list,target_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):

    def randominit_crowded(self,num=5):
        import random
        size = self.map_size
        # print(size) 
        pos_list = []
        size_list = []
        target_list = []
        # self.finished = np.zeros(num)
        
        tmp = self.build_wall()
        # tmp += self.build_wall()
        # tmp += self.build_wall()
        sorted(tmp)
        wall_list = []
        l = None
        for t in tmp:
            if t != l:
                l = t
                wall_list.append(t)

        for i in range(num):
            while True:
                px = np.random.randint(1,size[0]-3)
                py = np.random.randint(1,size[1]-3)
                flag = True

                hh = [-1,0,1]
                for _ in pos_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx - px) + abs(dy - py) < 5:
                        flag = False
                    
                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break


                for _ in wall_list:
                    if flag == False:
                        break
                    dx,dy = _
                    if dx == px and dy == py:
                        flag = False
                        break
                    
                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break

                if flag:
                    pos_list.append((px,py))
                    break
            
            while True:
                px = np.random.randint(1,size[0]-3)
                py = np.random.randint(1,size[1]-3)
                flag = True

                
                for _ in target_list:
                    dx, dy = _
                    if dx == px or dy == py or abs(dx-px)+abs(dy-py) < 5:
                        flag = False
                        break

                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break
                
                for _ in wall_list:
                    if flag == False:
                        break
                    dx,dy = _
                    if dx == px and dy == py:
                        flag = False
                        break
                    
                    for k in hh:
                        for l in hh:
                            xx = px + k
                            yy = py + l
                            if dx == xx and dy == yy:
                                flag = False
                                break
                                
                        if flag == False:
                            break

                    if flag == False:
                        break

                if flag:
                    target_list.append((px,py))
                    break
            
        # random.shuffle(pos_list)
        # random.shuffle(target_list)

        sub_num = self.max_num - num
        # print(self.max_num, num)
        for _ in range(sub_num):
            pos_list.append((0,0))
        # print(len(pos_list))
        random.shuffle(pos_list)
        pos_list = np.array(pos_list)
        tmp_target_list = np.zeros_like(pos_list)
        random.shuffle(target_list)
        _ = 0
        for target in target_list:
            while pos_list[_][0] == pos_list[_][1] and pos_list[_][0] == 0:
                _ += 1
            tmp_target_list[_] = target
            _ += 1

        target_list = tmp_target_list


        for i in range(self.max_num):
            px, py = pos_list[i]
            tx, ty = target_list[i]

            if px == py and py == 0:
                size_list.append((0,0))
                continue

            while True:
                # dx = np.random.randint(2,31)
                # dy = np.random.randint(2,31)
                dx = np.random.randint(2,11)
                dy = np.random.randint(2,11)
                if px + dx > size[0] or py + dy > size[1] or tx + dx > size[0] or ty + dy > size[1]:
                    # print(px,py,dx,dy)
                    continue

                flag = True

                for j in range(self.max_num):
                    if j == i:
                        continue

                    px_, py_ = pos_list[j]
                    tx_, ty_ = target_list[j]

                    if px_ == py_ and py_ == 0:
                        # size_list.append((0,0))
                        continue

                    if j > len(size_list) - 1:
                        dx_, dy_ = 2, 2
                    else:
                        dx_, dy_ = size_list[j]

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break


                for wall in wall_list:
                    px_, py_ = wall
                    tx_, ty_ = wall
                    
                    dx_, dy_ = 1, 1
                    

                    lx = max(px, px_)
                    ly = max(py, py_)
                    rx = min(px + dx, px_ + dx_) - 1
                    ry = min(py + dy, py_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                    lx = max(tx, tx_)
                    ly = max(ty, ty_)
                    rx = min(tx + dx, tx_ + dx_) - 1
                    ry = min(ty + dy, ty_ + dy_) - 1
                    if lx <= rx and ly <= ry:
                        flag = False
                        break

                if flag:
                    size_list.append((dx,dy))
                    break
        
        shapes = []
        for i in range(self.max_num):
            x,y = pos_list[i]
            w,d = size_list[i]
            pos_list[i] = (x+0.5*w,y+0.5*d)
            x,y = target_list[i]
            target_list[i] = (x+0.5*w,y+0.5*d)
            shape = []
            for a in range(w):
                for b in range(d):
                    shape.append((a,b))

            shapes.append(shape)


        self.setmap(pos_list,target_list,shapes,np.zeros(len(pos_list)),np.zeros(len(pos_list)),wall_list)
    
    def getstate_1(self,shape=[64,64]):
        state = []
        # temp = np.zeros(shape).astype(np.float32)
        # tmap = self.map == 1
        # oshape = self.map.shape
        # for i in range(shape[0]):
        #     x = int(1.0 * i * oshape[0] / shape[0])
        #     for j in range(shape[1]):
        #         y = int(1.0 * j * oshape[1] / shape[1])
        #         temp[i,j] = tmap[x,y]
        # for i in range(oshape[0]):
        #     x = 1.0*i/oshape[0]*shape[0]
        #     for j in range(oshape[1]):
        #         y = 1.0*j/oshape[1]*shape[1]
        #         temp[x,y] = tmap[i,j]

        state.append((self.map == 1).astype(np.float32))
        # state.append(temp)

        for i in range(len(self.pos)):
        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)

        #     temp = np.zeros(shape).astype(np.float32)
        #     tmap = self.target_map == (i+2)
        #     for i in range(shape[0]):
        #         x = int(1.0 * i * oshape[0] / shape[0])
        #         for j in range(shape[1]):
        #             y = int(1.0 * j * oshape[1] / shape[1])
        #             temp[i,j] = tmap[x,y]
            
        #     state.append(temp)
            state.append((self.map == (i+2)).astype(np.float32))
            state.append((self.target_map == (i+2)).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getstate_2(self, index):
        state = []
        obs = (self.map == 1).astype(np.float32)
        cho = (self.map == (index+2)).astype(np.float32)
        cho_t = (self.target_map == (index+2)).astype(np.float32)
        oth = np.zeros_like(obs).astype(np.bool)
        oth_t = np.zeros_like(obs).astype(np.bool)
        for i in range(len(self.pos)):
            if i != index + 2:
                oth |= self.map == (i+2)
                oth_t |= self.target_map == (i+2)
            # state.append((self.map == (i+1)).astype(np.float32))
        
        state = [obs,cho,cho_t,oth,oth_t]
        return np.transpose(np.array(state),[1,2,0])

    def getstate_3(self,shape=[64,64]):
        state = []

        state.append(np.array(self.map).astype(np.float32))
        state.append(np.array(self.target_map).astype(np.float32))

        return np.transpose(np.array(state),[1,2,0])
    
    def getmap(self):
        return np.array(self.map)
    
    def gettargetmap(self):
        return self.target_map

    def getconfig(self):
        return (self.pos,self.target,self.shape,self.cstate,self.tstate,self.wall)
    
    def getfinished(self):
        return deepcopy(self.finished)

    def getconflict(self):

        num = len(self.finished)
        res = np.zeros([num,num,2])
        # size = np.zeros([num,2])
        # for i,shape in enumerate(self.shape):
        #     size[i]
        return res  
        for axis in range(2):
            for i in range(num):
                l = np.min([self.pos[i][axis],self.target[i][axis]])
                r = np.max([self.pos[i][axis],self.target[i][axis]]) + self.size[i][axis] - 1
                for j in range(num):
                    if (self.pos[j][axis] >= l and self.pos[j][axis] <= r) or (self.pos[j][axis] + self.size[j][axis] >= l and self.pos[j][axis] + self.size[j][axis] <= r):
                        res[i,j,axis] = 1
        return res

    def getitem(self, index, state):
        shape = self.shape[index]
        opoints = np.array(shape)
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))
        
        radian = 2 * pi * state / self.bin
        points, bbox = self.rotate(opoints, radian)
        return points, bbox

    def getcleanmap(self, index):
        start = self.pos[index]
        # size = self.size[index]
        cstate = self.cstate[index]
        res = np.array(self.map)
        shape = self.shape[index]
        opoints = np.array(shape)
        # opoints = []
        # for i in range(size[0]):
        #     for j in range(size[1]):
        #         opoints.append((i,j))
        
        radian = 2 * pi * cstate / self.bin
        points, bbox = self.rotate(opoints, radian)
        xx, yy = start
        # xx = max(0, xx - 0.5*(bbox[1,0]-bbox[0,0]+1))
        # yy = max(0, yy - 0.5*(bbox[1,1]-bbox[0,1]+1))

        for p in points:
            x, y = p
            x += xx
            y += yy
            # x = min(x, self.map_size[0]-1)
            # y = min(y, self.map_size[1]-1)
            res[int(x), int(y)] = 0

        return res



class ENVN:
    def __init__(self, size=(5,5), start=(0,0)):
        self.map_size = size
        self.dest = np.zeros(self.map_size)
        self.map = np.zeros(self.map_size)
        self.start = start

    """
        params
        pos:
            a list of the left bottom point of the furniture
        size:
            a list of the size of the furniture
    """
    def setmap(self, dest_pos, pos, size): 
        self.dest = np.zeros(self.map_size)
        self.map = np.zeros(self.map_size)
        self.pos = pos
        self.dest_pos = dest_pos
        self.size = size
        self.mark = np.zeros(len(pos))
        for i, _ in enumerate(pos):
            dx, dy = size[i]
            x, y = _
            for gox in range(dx):
                for goy in range(dy):
                    self.map[x + gox,y + goy] = i + 2

    '''
        return reward, finish_flag 1 represent the finished state
    '''
    def move(self, index):
        if index >= len(self.mark) or self.mark[index] == 1:
            return -500, 0
        target = self.pos[index]
        size = self.size[index]
        dis = np.ones_like(self.map)
        dis *= -1                   # when it starts, all distance was initialized as -1 
        Q = queue.Queue()
        self.mark[index] = 1

        # to confirm the starting point is legal
        l = True
        for i in range(size[0]):
            for j in range(size[1]):
                if self.map[self.start[0]+i,self.start[1]+j] == 1:
                    l = False
                    break
            if not l:
                break
        if l:
            Q.put(self.start)
            dis[self.start[0], self.start[1]] = 0

        # BFS
        while not Q.empty():
            a = Q.get()
            x,y = a
            d = dis[x,y]
            l = True     # whether it is legal
            
            if l:
                # dis[x, y] = d + 1
                if x == target[0] and y == target[1]:
                    break

                for i, dx in enumerate(_x):
                    for dy in _y:
                        tx = x + dx
                        ty = y + dy
                        if tx >= 0 and tx < self.map_size[0] and ty >= 0 and ty < self.map_size[1] and dis[tx,ty] == -1:
                            Q.put((tx,ty))
                            dis[tx,ty] = d + 1
        
        for i in range(size[0]):
            for j in range(size[1]):
                self.map[target[0]+i, target[1]+j] = 1

        # with open("dis.txt",'a') as fp:
        #     for _ in dis:
        #         for __ in _:
        #             fp.write('%d '%__)
        #         fp.write('\n')
        #     fp.write('\n')
        if dis[target[0],target[1]] == -1:
            return -1000, 1

        if np.sum(self.mark) == len(self.mark):
            return 1000-dis[target[0],target[1]], 1

        return -dis[target[0],target[1]], 0

    def printmap(self):
        with open('op.txt','a') as fp:
            for _ in self.map:
                for __ in _:
                    fp.write('%d '%__)
                fp.write('\n')
            fp.write('\n')

    def randominit(self):
        import random
        size = self.map_size
        while True:
            dx = np.random.randint(2,4)
            dy = np.random.randint(2,4)
            dx_ = np.random.randint(2,4)
            dy_ = np.random.randint(2,4)
            alter = []

            for i in range(size[0]):
                if i + dx - 1 < size[0]:
                    for j in range(size[1]):
                        if j + dy - 1 < size[1]:
                            for k in range(i+1,size[0]):
                                if k + dx_ - 1 < size[0]:
                                    for l in range(j+1,size[1]):
                                        if l + dy_ - 1 < size[1]:
                                            if not (i + dx - 1 >= k and j + dy - 1 >= l):
                                                alter.append([(i,j),(k,l)])
            
            num = len(alter)
            if num > 0:
                choice = np.random.randint(num)
                pair = alter[choice]
                break
        
        pos_list = []
        size_list = []
        list_ =[(pair[0],(dx,dy)),(pair[1],(dx_,dy_))]
        random.shuffle(list_)
        for i in list_:
            pos_list.append(i[0])
            size_list.append(i[1])
        
        self.setmap(pos_list,size_list)

            
            # for i in range(size[0]):
            #     for j in range(size[1]):
                    

            # for i in range(size[0]):
            #     for j in range(size[1]):
        
    def getstate(self):
        return np.array(self.map)

def test():
    env = ENV3(size=(15,15))
    # pos = [[2,1],[7,8],[1,11]]
    # size = [[1,3],[6,2],[10,1]]
    pos = [[7,4],[10,6],[13,3]]
    size = [[3,5],[3,9],[2,7]]
    env.setmap(pos,size)
    reward,done = env.move(2)
    print(reward, done)
    reward,done = env.move(0)
    print(reward, done)
    reward,done = env.move(1)
    print(reward, done)

def test2():
    env = ENV_M_C(size=(15,15))
    pos = [[7,4],[10,6],[13,3]]
    size = [[3,5],[3,9],[2,7]]
    env.setmap(pos,size)
    state = env.getstate()
    print(state.shape)
    # for s in state:
    #     for r in s:
    #         print(r)
    #     print()

def test_scene():
    from visualization import convert
    from PIL import Image
    # env = ENV_scene(size=(15,15))
    env = ENV_scene_new_action_pre_state(size=(15,15))
    # for i in range(50):
    #     env.randominit()
    #     map_ = env.getmap()
    #     map_ = convert(map_)
    #     targetmap = env.gettargetmap()
    #     targetmap = convert(targetmap)
    #     map_ = Image.fromarray(map_)
    #     map_.save('test_scene/%d_pos.png'%i)
    #     targetmap = Image.fromarray(targetmap)
    #     targetmap.save('test_scene/%d_target.png'%i)
        # env.getstate_1()
    task = 0
    step = 0
    while True:
        cmd = input()
        if cmd == 'start':
            step = 0
            task += 1
            env.randominit()
            pos = env.getmap()
            pos = convert(pos)
            img = Image.fromarray(pos)
            img.save('test_scene/%d_step_%d.png'%(task,step))
            target = env.gettargetmap()
            target = convert(target)
            img = Image.fromarray(target)
            img.save('test_scene/%d_ideal.png'%task)
            

            
            step += 1
            continue
        if cmd == 'end':
            break
        
        a,b = [int(x) for x in cmd.split()]
        reward, done = env.move(a, b)
        print('reward',reward,'done',done,'hash',env.hash(),env.hash(),'happened before',env.hash() in env.state_dict)
        pos = env.getmap()
        pos = convert(pos)
        img = Image.fromarray(pos)
        img.save('test_scene/%d_step_%d.png'%(task,step))
        step += 1

def li():
    from visualization import convert
    from PIL import Image
    li = []
    for i in range(38):
        li.append(i)
    a = np.array([li])
    a = convert(a)
    Image.fromarray(a).save('li.png')

def test_L():
    from visualization import convert
    from PIL import Image
    import time
    env = ENV_M_C_L(size=(128,128))
    t = time.time()
    env.randominit()
    t = time.time()-t
    print(t)
    img = env.getmap()
    img = convert(img)
    img = Image.fromarray(img)
    img.save('see.png')

def test_new_action():
    from visualization import convert
    from PIL import Image
    env = ENV_scene_new_action((15,15))
    env.randominit_crowded()
    s = env.getmap()
    s = convert(s)
    Image.fromarray(s).save('what.png')
    s = env.gettargetmap()
    s = convert(s)
    Image.fromarray(s).save('whatt.png')
    
def test_hash():
    env = ENV_scene_new_action_pre_state()
    env.randominit()
    print(env.hash())   

def test_crowd():
    from visualization import convert
    from PIL import Image
    env = ENV_scene_new_action_pre_state(size=(15,15))
    for i in range(100):
        env.randominit_crowded()
        m = env.getmap()
        m = convert(m)
        Image.fromarray(m).save('test_crowd/%d.png'%i)

def test_route():
    env = ENV_scene_new_action_pre_state(size=(15,15))
    for i in range(20):
        env.randominit_crowded()
        for step in range(10):
            choice = np.random.randint(5)
            direction = np.random.randint(5)
            print('choice',choice,'direction',direction)
            env.move(choice,direction)

def gen_testcase():
    import os
    from visualization import convert
    from PIL import Image
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(64,64),max_num=25)
    path = 'data'
    total = 50000
    dir_ = 'imgs'
    cases = []
    for i in range(total):
        num = np.random.randint(5,26)
        img_path = os.path.join(path, dir_, '%d.png'%i)
        env.randominit_crowded(num)
        # img = env.getmap()
        # # print(img.shape)
        # img = convert(img)
        # img = Image.fromarray(img)
        # img.save(img_path)
        print(img_path)
        # img_path = os.path.join(path, dir_, '%d_t.png'%i)
        # img = env.gettargetmap()
        # # print(img.shape)
        # img = convert(img)
        # img = Image.fromarray(img)
        # img.save(img_path)
        cases.append(env.getconfig())

    with open('data/exp64_%d.pkl'%num,'wb') as fp:
        pickle.dump(cases,fp)
    # for i in range(24):
    #     dir_ = '%d'%i
    #     if not os.path.exists(os.path.join(path,dir_)):
    #         os.makedirs(os.path.join(path,dir_))
        
    #     for j in range(100):
    #         num = np.random.randint(5,26)
    #         img_path = os.path.join(path, dir_, '%d.png'%j)
    #         env.randominit_crowded(num)
    #         img = env.getmap()
    #         # print(img.shape)
    #         img = convert(img)
    #         img = Image.fromarray(img)
    #         img.save(img_path)
    #         print(img_path)
    #         img_path = os.path.join(path, dir_, '%d_t.png'%j)
    #         img = env.gettargetmap()
    #         # print(img.shape)
    #         img = convert(img)
    #         img = Image.fromarray(img)
    #         img.save(img_path)
        
    # nums = [5,10,15,20]
    # for num in nums:
    #     cases = []
    #     print('num',num,'finished')
    #     dir_ = '%d'%num
    #     if not os.path.exists(os.path.join(path,dir_)):
    #         os.makedirs(os.path.join(path,dir_))
    #     for i in range(20):
    #         env.randominit_crowded(num)
    #         cases.append(env.getconfig())

    #         img_path = os.path.join(path, dir_, '%d.png'%i)
    #         img = env.getmap()
    #         # print(img.shape)
    #         img = convert(img)
    #         img = Image.fromarray(img)
    #         img.save(img_path)
    #         print(img_path)
    #         img_path = os.path.join(path, dir_, '%d_t.png'%i)
    #         img = env.gettargetmap()
    #         # print(img.shape)
    #         img = convert(img)
    #         img = Image.fromarray(img)
    #         img.save(img_path)

    #     with open('exp64_%d.pkl'%num,'wb') as fp:
    #         pickle.dump(cases,fp)

def test_parallel():
    env_p = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_max(size=(55,55),max_num=20)

    env_o = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(55,55))
    configs = []
    
    for i in range(100):
        env_o.randominit_crowded(10)
        configs.append(env_o.getconfig())
    a = []
    b = []

    t = time.time()
    for j, config in enumerate(configs):
        pos, target, size = config
        env_o.setmap(pos, target, size)
        for i in range(5):
            reward,done = env_o.move(i,4)
            b.append(reward)
        # print(j,'ok')

    print('chuan',time.time()-t)
    
    t = time.time()
    for j, config in enumerate(configs):
        pos, target, size = config
        env_p.setmap(pos, target, size)
        for i in range(5):
            reward,done = env_p.move(i,4)
            a.append(reward)
        # print(j,'ok')

    print('parallel',time.time()-t)

    
    for i,_ in enumerate(a):
        print(a[i])
        if a[i] != b[i]:
            print(i,'error',a[i],b[i])
    
def test_rotate():

    def rotate(points, size, radian):
        eps = 1e-6
        res_list = []
        points = np.array(points).astype(np.float32)
        points[:,0] += -0.5 * size[0] + 0.5
        points[:,1] += -0.5 * size[1] + 0.5
        points = points.transpose()
        rot = np.array([[cos(radian),-sin(radian)],[sin(radian), cos(radian)]])
        points = np.matmul(rot,points)
        points[0,:] += 0.5 * size[0] - 0.5 + eps
        points[1,:] += 0.5 * size[1] - 0.5 + eps
        

        points = np.round(points).astype(np.int)

        xmax = points[0].max()
        xmin = points[0].min()
        ymax = points[1].max()
        ymin = points[1].min()

        return points.transpose(), [[xmin, ymin], [xmax, ymax]]

    from visualization import convert_to_img
    from PIL import Image
    eps = 1e-6
    _map = np.zeros([35,35])
    _t = np.zeros([35,35])
    size = (1,20)
    pos = (15,15)
    pos_list = []
    ori_list = []
    r = pi/6
    for i in range(size[0]):
        for j in range(size[1]):
            ori_list.append((i,j))
    
    pos_list, bbox = rotate(ori_list, size, r)

    # for x in range(size[0]):
    #     _x = x - 0.5 * size[0] + 0.5
    #     for y in range(size[1]):
    #         ori_list.append((x,y))
    #         _y = y - 0.5 * size[1] + 0.5
    #         rot = np.array([[cos(r),-sin(r)],[sin(r), cos(r)]]) 
    #         res = np.matmul(rot,np.array([[_x],[_y]]))
    #         # print(res)
    #         tx = res[0][0]
    #         ty = res[1][0]
    #         print(tx,ty)
    #         tx += 0.5 * size[0] - 0.5 + eps
    #         ty += 0.5 * size[1] - 0.5 + eps
    #         print(tx,ty)
    #         tx = int(np.round(tx))
    #         ty = int(np.round(ty))
    #         print(tx,ty)
    #         print('====================================')
    #         pos_list.append((tx,ty))
    
    bbox = np.array(bbox)
    bbox[:,0] += pos[0]
    bbox[:,1] += pos[1]
    print(bbox)

    for i,p in enumerate(pos_list):
        op = ori_list[i]
        op = (op[0] + pos[0], op[1] + pos[1])
        p_ = (p[0] + pos[0], p[1] + pos[1])
        print(p_)
        _map[p_[0],p_[1]] = 2
        _t[op[0],op[1]] = 3
    
    for i in range(bbox[0][0], bbox[1][0]+1):
        y1 = bbox[0][1]
        y2 = bbox[1][1]
        _t[i,y1] = _t[i,y2] = 4
    
    for i in range(bbox[0][1], bbox[1][1]+1):
        x1 = bbox[0][0]
        x2 = bbox[1][0]
        _t[x1,i] = _t[x2,i] = 4
    
    img = convert_to_img(_map,_t,np.zeros_like(_map))
    img = Image.fromarray(img)
    img.save('rot.png')

def write_case():
    filename = []
    filename.append('exp35_5')
    filename.append('exp35_9')
    filename.append('exp35_13')
    filename.append('exp35_17')

    for f in filename:
        with open('%s.pkl'%f,'rb') as fp:
            configs = pickle.load(fp)
        
        with open('%s.txt'%f,'w') as fp:
            for config in configs:
                pos,target,size = config
                for p in pos:
                    fp.write('(%d,%d) '%p)
                fp.write('\n')
                for t in target:
                    fp.write('(%d,%d) '%t)
                fp.write('\n')
                for s in size:
                    fp.write('(%d,%d) '%s)
                fp.write('\n\n')
        
def show_case():
    import os
    from PIL import Image
    from visualization import convert_to_img
    from visualization import convert
    filename = []
    filename.append('exp35_5')
    filename.append('exp35_9')
    filename.append('exp35_13')
    filename.append('exp35_17')
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic(size=(35,35))
    for f in filename:
        with open('%s.pkl'%f,'rb') as fp:
            configs = pickle.load(fp)
        
        save_path = f
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, config in enumerate(configs):
            p, t, s = config
            env.setmap(p, t, s)
            map_ = env.getmap()
            tmap_ = env.gettargetmap()
            # img = convert_to_img(map_,tmap_,np.zeros_like(map_))
            img = convert(map_)
            img = Image.fromarray(img)
            img.save('%s/%d_i.png'%(f,i))
            img = convert(tmap_)
            img = Image.fromarray(img)
            img.save('%s/%d_t.png'%(f,i))
            print('finished %s/%d.png'%(f,i))

def test_border():
    from visualization import convert
    from PIL import Image
    map_size = (256,256)

    with open('config.pkl','rb') as fp:
        test_configs = pickle.load(fp)
    
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=map_size)

    for idd, test_config in enumerate(test_configs):
        # print('case',idd)
        pos_, target_, shape_, cstate_, tstate_, wall_ = test_config
        env.setmap(pos_, target_, shape_, cstate_, tstate_, wall_)
        map_ = np.array(env.getmap()==1,dtype=np.int32)

        for i,pos in enumerate(env.pos):
            edge = env.edge[i]
            x,y = pos
            edge, bbox = env.rotate(edge, 0.5*pi)
            for p in edge:
                xx = min(max(int(p[0] + x),0),255)
                yy = min(max(int(p[1] + y),0),255)
                # x,y = p
                map_[xx,yy] = i + 2

        img = convert(map_)
        img = Image.fromarray(img)
        img.save('wtf.png')
    
def test_poly():
    from visualization import convert
    from PIL import Image

    # with open('config.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    with open('simple_64_config.pkl','rb') as fp:
        test_configs = pickle.load(fp)
    map_size = (64,64)
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape(size=map_size)
    env_poly = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=map_size)

    for idd, test_config in enumerate(test_configs):
        pos_, target_, shape_, cstate_, tstate_, wall_ = deepcopy(test_config)
        env.setmap(pos_, target_, shape_, cstate_, tstate_, wall_)
        pos_, target_, shape_, cstate_, tstate_, wall_ = deepcopy(test_config)
        env_poly.setmap(pos_, target_, shape_, cstate_, tstate_, wall_)
        
        num = pos_.__len__()
        # pre = 0
        # for j in range(num):
        #     with open('points%d.txt'%j,'w') as fp:
        #         for i in range(env_poly.pn[j]):
        #             fp.write('%f %f\n'%(env_poly.shapex[pre+i],env_poly.shapey[pre+i]))
        #     pre += env_poly.pn[j]

        # reward, done = env_poly.move(0,0)
        # print(reward, done)
        # map_ = np.array(env_poly.getmap() == 1,dtype=np.int32)
        
        # env_poly.move(0,1)

        # for i,pos in enumerate(env_poly.pos):
        #     edge = env_poly.edge[i]
        #     x,y = pos
        #     edge, bbox = env_poly.rotate(edge, 0)
        #     for p in edge:
        #         xx = min(max(int(p[0] + x),0),255)
        #         yy = min(max(int(p[1] + y),0),255)
        #         # x,y = p
        #         map_[xx,yy] = i + 2
       
        # img = convert(map_)
        # img = Image.fromarray(img)
        # img.save('wtf222.png')
        o_time = 0
        n_time = 0
        for i in range(num):
                # direction = 4
            # i = 1
            for direction in range(5):
                # direction = 1
                print('obj',i,'direction',direction)
                t = time.time()
                reward, done = env.move(i,direction)
                o_time += time.time() - t
                t = time.time()
                reward_poly, done_poly = env_poly.move(i,direction)
                n_time += time.time() - t
                print('o_reward',reward,'n_reward',reward_poly)
                if reward != reward_poly or done != done_poly:
                    print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!========================','old: %d %d'%(reward,done),'poly: %d %d'%(reward_poly,done_poly))

                print('o_time %f n_time %f'%(o_time,n_time))


                map_ = np.array(env.getmap())
                img = convert(map_)
                img = Image.fromarray(img).resize((1024,1024))
                img.save('see_test/env_%d_%d.png'%(i,direction))

                map_ = np.array(env_poly.getmap() == 1,dtype=np.int32)

                for j,pos in enumerate(env_poly.pos):
                    edge = env_poly.edge[j]
                    x,y = pos
                    ra = 2 * pi * env_poly.cstate[j] / env_poly.bin 
                    edge, bbox = env_poly.rotate(edge, ra)
                    for idx,p in enumerate(edge):
                        p_ = edge[idx-1]
                        nn = int(max(fabs(p[0]-p_[0]),fabs(p[1]-p_[1])) + 1e-6 + 1)
                        for k in range(nn+1):
                            xx = min(max(int(1.0*k/nn*p[0]+(1.0-1.0*k/nn)*p_[0] + x),0),255)
                            yy = min(max(int(1.0*k/nn*p[1]+(1.0-1.0*k/nn)*p_[1] + y),0),255)
                            # x,y = p
                            map_[xx,yy] = j + 2
            
                img = convert(map_)
                img = Image.fromarray(img).resize((1024,1024))
                
                img.save('see_test/env_poly_%d_%d.png'%(i,direction))
        #         print('===============================')
            
            # break

def test_the_case():
    from visualization import convert
    from PIL import Image

    # with open('config.pkl','rb') as fp:
    #     test_configs = pickle.load(fp)
    config_file = 'simple_64_config.pkl'
    with open(config_file,'rb') as fp:
        test_configs = pickle.load(fp)
    map_size = (64,64)
    # env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape(size=map_size)
    env_poly = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=map_size)

    for idd, test_config in enumerate(test_configs):
        # pos_, target_, shape_, cstate_, tstate_, wall_ = deepcopy(test_config)
        # env.setmap(pos_, target_, shape_, cstate_, tstate_, wall_)
        pos_, target_, shape_, cstate_, tstate_, wall_ = deepcopy(test_config)
        env_poly.setmap(pos_, target_, shape_, cstate_, tstate_, wall_)
        
        num = pos_.__len__()

        o_time = 0
        n_time = 0
        config_cnt = 0
        while True:
            action = input()
            if action == 'quit':
                break
            if action == 'save':
                config = env_poly.getconfig()
                config_cnt += 1
                configs.append(config)
                with open('%s_%d'%(config_file[:-4],config_cnt),'wb') as fp:
                    pickle.dump(configs,fp)

                continue
            
            i,direction = [int(k) for k in action.split()]
        # for i in range(num):
        #         # direction = 4
        #     # i = 1
        #     for direction in range(5):
        #         # direction = 1
            print('obj',i,'direction',direction)
            

            reward_poly, done_poly = env_poly.move(i,direction)


            map_ = np.array(env_poly.getmap() == 1,dtype=np.int32)

            for j,pos in enumerate(env_poly.pos):
                edge = env_poly.edge[j]
                x,y = pos
                ra = 2 * pi * env_poly.cstate[j] / env_poly.bin 
                edge, bbox = env_poly.rotate(edge, ra)
                for idx,p in enumerate(edge):
                    p_ = edge[idx-1]
                    nn = int(max(fabs(p[0]-p_[0]),fabs(p[1]-p_[1])) + 1e-6 + 1)
                    for k in range(nn+1):
                        xx = min(max(int(1.0*k/nn*p[0]+(1.0-1.0*k/nn)*p_[0] + x),0),63)
                        yy = min(max(int(1.0*k/nn*p[1]+(1.0-1.0*k/nn)*p_[1] + y),0),63)
                        # x,y = p
                        map_[xx,yy] = j + 2
        
            img = convert(map_)
            img = Image.fromarray(img).resize((1024,1024))
            
            img.save('see_test/env_poly_%d_%d.png'%(i,direction))
        #         print('===============================')
            
            # break


def test_copy():
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly([64,64],25)
    t_copy = 0
    t_dcopy = 0
    total = time.time()
    for i in range(20):
        
        env.randominit_crowded(20)
        # t = time.time()
        # env_copy = env.__deepcopy__()
        # t_copy += time.time() - t
        t = time.time()
        env_copy = deepcopy(env)
        t_dcopy += time.time() - t
        for j in range(125):
            state = env.getstate_1()
            state_c = env_copy.getstate_1()
            
            item = int(j / 5)
            a = int(j % 5)
            r, f = env.move(item,a)
            r_c,f_c = env_copy.move(item,a)

            if not (state == state_c).all():
                print('error')

            # print(r,f,r_c,f_c)
    total = time.time() - total
    print(total, t_copy, t_dcopy)


def test_wall():
    from visualization import convert
    from PIL import Image
    env = ENV_scene_new_action_pre_state_penalty_conflict_heuristic_transpose_shape_poly(size=(64,64))
    for i in range(30):
        for j in range(1,26):
            env.randominit_crowded(j)
            m = env.getmap()
            m = convert(m)
            Image.fromarray(m).save('walls_poly/%d_%d.png'%(i,j))
            m = env.gettargetmap()
            m = convert(m)
            Image.fromarray(m).save('walls_poly/%d_%d_t.png'%(i,j))
            print(i,j)



if __name__ == '__main__':
    # main()
    # test()
    # test2()
    # test_scene()
    # test_crowd()
    # test_L()
    # test_new_action()
    # li()
    # test_route()
    gen_testcase()
    # write_case()
    # test_wall()
    # test_parallel()
    # test_border()
    # test_poly()
    # test_the_case()
    # show_case()
    # test_rotate()
    # test_hash()
