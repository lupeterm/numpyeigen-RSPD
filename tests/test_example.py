from example import planedetection
import json
import numpy as np
import os
from itertools import accumulate

def read_mesh(path: str):

    data = mesh = normals = None
    with open(path) as json_file:
        data = json.load(json_file)
        mesh = np.empty((len(data['verticies']), 3), dtype=np.float32)
        normals = np.empty((len(data['normals']), 3), dtype=np.float32)
    for i, v in enumerate(data['verticies']):
        mesh[i][0] = v['x']
        mesh[i][1] = v['y']
        mesh[i][2] = v['z']
        normals[i][0] = data['normals'][i]['x']
        normals[i][1] = data['normals'][i]['y']
        normals[i][2] = data['normals'][i]['z']
    return mesh,normals


if __name__ == '__main__':
    meshpath = "/home/pedda/Documents/uni/BA/Thesis/catkin_ws/src/plane-detection/src/EVAL/Stanford3dDataset_v1.2_Aligned_Version/TEST/WC_1/WC_1.txt"
    # meshpath = "/home/pedda/Documents/work/simulator/testcases/BePo/2022-07-26_13-00/mesh_690.10400390625.json"
    # points, normals = read_mesh(meshpath)
    points = np.loadtxt(meshpath, usecols= (0,1,2),dtype=np.float32)
    normals = np.copy(points)
    print(points.shape, normals.shape)
    np.savetxt(os.path.join('output','test.txt'),points, delimiter= " ")    

    sizes, points = planedetection(points, normals)
    print(points.shape, len(sizes))
    for i, plane in enumerate(np.split(points, list(accumulate(sizes)))):
        if len(plane) > 0:
            np.savetxt(os.path.join('output',f'plane-{i}.txt'),plane, delimiter=' ')