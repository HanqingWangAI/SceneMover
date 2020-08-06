import numpy as np 
from math import sqrt
from math import log
from math import sin

gamma = 0.95
# C = 3
action_size = 25

class TreeNode:
    
    def __init__(self):
        self.id = 0
        self.times = 0
        self.reward = 0
        self.value = 0
        self.faid = 0
        self.childs = {}
        self.state = None
        self.h_state = None
        self.action = 0
        self.best = -1
        self.depth = 0
        self.policy = 0
        self.sims = [] # This list records the historical simiulation frames and the corresponding actions.
        '''
        an entry of sims is like (state, action, reward), 
        where action is the action lead to the state and 
        reward is the reward acquired when the action is executed.
        For the lstm, state include the current map and the hidden state,
        which is (state, h_state).
        '''
    
    def addchild(self, action, child):
        self.childs[action] = child
    
    def printnode(self):
        print('id:',self.id,', value:',self.value,', reward:',self.reward,', socre:',gamma*self.value+self.reward,', faid:',self.faid,', action:',self.action,', times:',self.times,', best:',self.best)
    
 

class MCT:
    def __init__(self):
        self.root = TreeNode()
        self.nodedict = {}
        self.nodedict[self.root.id] = self.root
        self.nnum = 1
        self.explist = []
        self.explist.append(self.root.id)
    
    def setactionsize(self, action_size):
        self.action_size = action_size

    def selection(self,C=3):
        max_value = -np.inf
        min_value = np.inf
        for _id in self.explist:
            node = self.nodedict[_id]
            max_value = max([node.value, max_value])
            min_value = min([node.value, min_value])

        # if max_value == min_value:
        #     print('error!',len(self.explist))

        max_score = -1
        choose = -1

        for _id in self.explist:
            node = self.nodedict[_id]
            fa = self.nodedict[node.faid]
            score = 1.0*(node.value-min_value+1)/(max_value-min_value+1) + C * sqrt(log(fa.times+1)/(node.times+1))

            if score > max_score:
                max_score = score
                choose = _id
        if choose == -1:
            print('error')
            for _id in self.explist:
                node = self.nodedict[_id]
                fa = self.nodedict[node.faid]
                score = 1.0*(node.value-min_value+1)/(max_value-min_value+1) + C * sqrt(log(fa.times+1)/(node.times+1))
                print('scpre')
        # print('max_value',max_value,'min_value',min_value,'max_score',max_score,'value',self.nodedict[choose].value)
        return choose

    def getemptyactions(self, _id):
        node = self.nodedict[_id]
        actions = []
        for i in range(self.action_size):
            if i in node.childs:
                continue
            actions.append(i)
        return actions

    def expansion(self, _id, action, reward, state, done = 0): # if the action is taken, return True, else return false
        node = self.nodedict[_id]
        illegal = (done <= -1)
        if not action in node.childs:
            if illegal:
                node.childs[action] = None
                if len(node.childs) == self.action_size:
                    self.explist.remove(_id)
                return (False, -1)

            if done != 1:
                self.explist.append(self.nnum)
                
            child = TreeNode()
            child.action = action
            child.faid = _id
            child.id = int(self.nnum)
            child.reward = reward
            child.state = state
            child.depth = node.depth + 1
            node.childs[action] = child.id
            self.nodedict[child.id] = child
            self.nnum += 1

            if len(node.childs) == self.action_size:
                self.explist.remove(_id)
            return (True, child.id)

        return (False, -1)
        
    # When execute the simulation, the value need to be updated

    def getstate(self, _id):
        node = self.nodedict[_id]
        return node.state
        
    def haschild(self, _id, action):
        node = self.nodedict[_id]
        return action in node.childs

    def backpropagation(self, _id):
        node = self.nodedict[_id]
        while node.faid != _id:
            node.times += 1
            action = node.action
            fa = self.nodedict[node.faid]

            if fa.best == -1: # if the father doesn't have a son before
                value = gamma * node.value + node.reward
                fa.best = action
                if value >= fa.value:
                    fa.value = value
            else:
                if fa.best == action: # if the best is this son
                    max_value = -np.inf
                    best = -1
                    for _ in fa.childs:
                        if fa.childs[_] == None:
                            continue
                        
                        tn = self.nodedict[fa.childs[_]]
                        value = gamma * tn.value + tn.reward
                        if value > max_value:
                            best = _
                            max_value = value
                    
                    # if best != action:
                    #     print('id',_id,"value",node.value,'best',best,'action',action)
                    if max_value >= fa.value:
                        fa.best = best
                        fa.value = max_value
                else: # if the best is other son
                    value = gamma * node.value + node.reward
                    if value > fa.value:
                        fa.value = value
                        fa.best = action

                        # print('id',_id,"value",node.value,'best',fa.best,'action',action)

            _id = node.faid
            node = self.nodedict[_id]
        
        node.times += 1 # the root node

    def getrootid(self):
        return self.root.id

    def printtree(self):
        nodes = []
        nodes.append(0)
        while len(nodes) != 0:
            node = self.nodedict[nodes[0]]
            node.printnode()
            for key in node.childs:
                if node.childs[key] != None:
                    nodes.append(node.childs[key])
            nodes.remove(nodes[0])

    def print_structure(self):
        nodes = []
        nodes.append(self.root.id)
        rt_depth = self.root.depth
        depths = {}
        while len(nodes) != 0:
            node = self.nodedict[nodes[0]]
            # node.printnode()
            d_ = node.depth - rt_depth
            if not d_ in depths:
                depths[d_] = 0
            depths[d_] += 1
            for key in node.childs:
                if node.childs[key] != None:
                    nodes.append(node.childs[key])
            nodes.remove(nodes[0])
        
        for i,_ in enumerate(depths):
            if _ == 0:
                break
            print('depth',i,'number',_)
    
    def getbestdepth(self):
        node = self.root
        best = node.best
        depth = node.depth
        while best in node.childs:
            node = self.nodedict[node.childs[best]]
            best = node.best
            depth = node.depth
        
        return depth - self.root.depth


    def nextstep(self, action):
        if not action in self.root.childs:
            print('No such son!')
            return False
        
        self.root = self.nodedict[self.root.childs[action]]
        self.root.faid = self.root.id
        tosave = []
        nodes = []
        nodes.append(self.root.id)
        while len(nodes) != 0:
            node = self.nodedict[nodes[0]]
            tosave.append(node.id)
            for key in node.childs:
                if node.childs[key] != None:
                    nodes.append(node.childs[key])
            nodes.remove(nodes[0])
        
        todel = [id_ for id_ in self.nodedict if not id_ in tosave]
        
        for id_ in todel:
            if id_ in self.nodedict:
                self.nodedict.pop(id_)
            if id_ in self.explist:
                self.explist.remove(id_)    

        return True



def main():
    tree = MCT()
    tree.root.state = 0
    for i in range(10000):
        _id = tree.selection()
        action = np.random.randint(action_size)
        state = tree.getstate(_id)
        reward = action*2
        next_state = state + action

        succ, child_id = tree.expansion(_id, action, reward, next_state)
        if succ:
            tree.nodedict[child_id].value = sin(next_state)
            tree.backpropagation(child_id)
    
    # tree.printtree()
    tree.root.printnode()
            
    
    

if __name__ == '__main__':
    main()