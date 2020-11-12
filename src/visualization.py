from PIL import Image
import numpy as np
colors = [[225,225,225],[0,0,0],[0,225,0],[0,0,225],[225,0,0],[225,255,0],[225,0,255],[0,255,255]]
route_color = [255, 127, 80]

for i in range(30):
    r = (i+1)*35 % 225
    g = (i+1)*r % 225
    b = (i+1)*g*r % 225
    colors.append([r,g,b])
 
def convert(input_map):
    # colors = [[78,113,190],[184,90,154],[156,33,137],[158,2,109],[184,40,138],[204,47,105],[143,29,120],[186,135,76],[233,174,106],[254,227,136],[255,254,160],[255,233,87],[242,259,63],[242,117,63],[232,126,81],[222,140,104],[1,25,53],[0,52,63],[29,176,184],[55,198,192],[208,233,255],[88,131,52],[74,102,165],[209,168,69],[188,113,61],[44,181,73],[40,136,255],[255,160,192],[155,150,219]]
    colors=[[182,38,61],[236,100,66],[240,147,61],[246,198,68],[234,222,107],[181,211,109],[118,197,139],[83,183,173],[66,121,152],[61,61,104],[75,27,71],[132,30,64]]
    color_num = len(colors)
    margin_width = 0
    grid_width = 5
    map_size = input_map.shape
    img_size = [map_size[0]*(grid_width + margin_width) + margin_width, map_size[1]*(grid_width + margin_width) + margin_width]
    img = np.ones([*img_size, 3])
    img *= 255
    
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            id = int(input_map[i,j])
            if id == 0:
                color = [225,225,225]
            elif id == 1:
                color = [128,128,128]
            else:
                # color = colors[((id-1)*7)%color_num]
                color = colors[id%color_num]
            bx = i * (grid_width + margin_width) + margin_width
            by = j * (grid_width + margin_width) + margin_width
            for k in range(grid_width):
                for l in range(grid_width):
                    img[bx+k,by+l] = color
    
    return np.uint8(img)

def convert_terrain(input_map,terrain):
    colors = [[78,113,190],[184,90,154],[156,33,137],[158,2,109],[184,40,138],[204,47,105],[143,29,120],[186,135,76],[233,174,106],[254,227,136],[255,254,160],[255,233,87],[242,259,63],[242,117,63],[232,126,81],[222,140,104],[1,25,53],[0,52,63],[29,176,184],[55,198,192],[208,233,255],[88,131,52],[74,102,165],[209,168,69],[188,113,61],[44,181,73],[40,136,255],[255,160,192],[155,150,219]]
    color_num = len(colors)
    margin_width = 0
    grid_width = 5
    map_size = input_map.shape
    img_size = [map_size[0]*(grid_width + margin_width) + margin_width, map_size[1]*(grid_width + margin_width) + margin_width]
    img = np.ones([*img_size, 3])
    img *= 255
    
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            id = int(input_map[i,j])
            if id == 0:
                color = terrain[i,j][:3]
            else:
                color = colors[(id-1)%color_num]
            bx = i * (grid_width + margin_width) + margin_width
            by = j * (grid_width + margin_width) + margin_width
            for k in range(grid_width):
                for l in range(grid_width):
                    img[bx+k,by+l] = color
    
    return np.uint8(img)


def convert_to_img(input_map, target_map, route_map):
    colors = [[225,225,225],[0,0,0],[78,113,190],[184,90,154],[156,33,137],[158,2,109],[184,40,138],[204,47,105],[143,29,120],[186,135,76],[233,174,106],[254,227,136],[255,254,160],[255,233,87],[242,259,63],[242,117,63],[232,126,81],[222,140,104],[1,25,53],[0,52,63],[29,176,184],[55,198,192],[208,233,255],[88,131,52],[74,102,165],[209,168,69],[188,113,61],[44,181,73],[40,136,255],[255,160,192],[155,150,219]]
    margin_width = 0
    grid_width = 10
    map_size = input_map.shape
    img_size = [map_size[0]*(grid_width + margin_width) + margin_width, map_size[1]*(grid_width + margin_width) + margin_width]
    img = np.ones([*img_size, 3])
    img *= 255
    
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            id = int(input_map[i,j])
            color = colors[id]
            bx = i * (grid_width + margin_width) + margin_width
            by = j * (grid_width + margin_width) + margin_width
            for k in range(grid_width):
                for l in range(grid_width):
                    img[bx+k,by+l] = color
    
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            id = int(target_map[i,j])
            if id != 0:
                color = colors[id]
                bx = i * (grid_width + margin_width) + margin_width
                by = j * (grid_width + margin_width) + margin_width
                for k in range(grid_width):
                    for l in range(grid_width):
                        if (k + l) % 3 == 0:
                            img[bx+k,by+l] = color
    
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            id = int(route_map[i,j])
            if id != 0:            
                bx = i * (grid_width + margin_width) + margin_width
                by = j * (grid_width + margin_width) + margin_width
                for k in range(grid_width):
                    for l in range(grid_width):
                        if (k + l) % 2 == 0:
                            img[bx+k,by+l] = route_color
    
    return np.uint8(img)

