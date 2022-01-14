#!/usr/bin/python3
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
from scipy import spatial
import hnswlib
import math
import time

img2vec = Img2Vec(cuda=False, model='densenet')
p = hnswlib.Index(space = 'cosine', dim = 1024) # possible options are l2, cosine or ip
p.init_index(max_elements = 200, ef_construction = 200, M = 16)


def store_olivetti_models_dict():
    global r
    global img2vec
    
    cnt = 1
    for personid in range(1, 41):
        person = "s" + str(personid)
        for face in range(1, 6):
            facepath = '../olivetti-database/' + person + "/" + str(face) + '.bmp'
            print ("Training face: " + facepath)
            img = Image.open(facepath).convert('RGB')
            vec = img2vec.get_vec(img)
            p.add_items(vec, cnt)
            cnt = cnt + 1
            

def test_olivetti_models_dict():
    global p
    success = 0
    p.set_ef(100)
    
    start_time = time.time()
    for personid in range(1, 41):
        person = "s" + str(personid)
        for face in range(6, 11):
            facepath = '../olivetti-database/' + person + "/" + str(face) + '.bmp'
            print ("Testing face: " + facepath)
            queryImage = Image.open(facepath).convert('RGB')
            vec = img2vec.get_vec(queryImage)  
            labels, distances = p.knn_query(vec, k = 1)
            if (math.ceil(labels.astype(int)[0][0]/5) == personid):
                success=success + 1
            print(math.ceil(labels.astype(int)[0][0]/5))

    print(success/200*100)
    print("--- %s seconds ---" % (time.time() - start_time))

store_olivetti_models_dict()
test_olivetti_models_dict()
