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


class ENV:  # if current state happened before, give a penalty.
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
        res = ENV(self.map_size,self.max_num)
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
   


if __name__ == '__main__':
    