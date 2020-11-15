import numpy as np
import queue
from math import *
from copy import deepcopy
import time
from queue import PriorityQueue as PQ
import pickle
_x = [-1,1,0,0]
_y = [0,0,-1,1]


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
        return self.map
    
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
