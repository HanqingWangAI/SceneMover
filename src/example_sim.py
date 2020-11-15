from utils.env import ENV
from utils.visualization import convert_to_img
import numpy as np
import pickle
import os
from PIL import Image

def main():
    dataset = '../data/train_IL.pkl'
    env = ENV(size=(64,64))

    with open(dataset,'rb') as fp:
        data = pickle.load(fp)[0]


    pos = data['pos']
    target = data['target']
    shape = data['shape']
    cstate = data['cstate']
    tstate = data['tstate']
    wall = data['wall']
    actions = data['actions']

    env.setmap(pos, target, shape, cstate, tstate, wall)

    imgs = []
    input_map = env.getmap()
    target_map = env.gettargetmap()
    imgs.append(convert_to_img(input_map, target_map, np.zeros([64,64])))
    for a in actions:
        item, d = a
        env.move(item, d)

        input_map = env.getmap()
        target_map = env.gettargetmap()
        imgs.append(convert_to_img(input_map, target_map, np.zeros([64,64])))

    if not os.path.exists('../imgs'):
        os.makedirs('../imgs')

    for i,img in enumerate(imgs):
        path = '../imgs/%d.png' % i
        img = Image.fromarray(np.uint8(img))
        img.save(path)


if __name__ == '__main__':
    main()