#!/usr/bin/python3
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np
from scipy import spatial

img2vec = Img2Vec(cuda=False, model='densenet')
trainDict = {}


def store_olivetti_models_dict():
    global r
    global trainDict
    global img2vec

    for person in range(1, 41):
        person = "s" + str(person)
        for face in range(1, 6):
            facepath = '../olivetti-database/' + person + "/" + str(face) + '.bmp'
            print ("Training face: " + facepath)
            img = Image.open(facepath).convert('RGB')
            vec = img2vec.get_vec(img)
            trainDict[person + "-" + str(face)] = vec


def test_olivetti_models_dict():
    success = 0
    for person in range(1, 41):
        person = "s" + str(person)
        for face in range(6, 11):
            facepath = '../olivetti-database/' + person + "/" + str(face) + '.bmp'
            print ("Testing face: " + facepath)
            found = find_face_dict_cosin(facepath)
            if (person == found):
                success = success +1 

    print(success/200*100)


def find_face_dict(path):
    global img2vec
    global trainDict
    results = {}

    queryImage = Image.open(path).convert('RGB')
    vec = img2vec.get_vec(queryImage)

    for i in trainDict.items():
        tmp = np.absolute(np.subtract(vec,i[1]))
        results[i[0]] = np.sum(tmp)
    found = str(min(results, key=results.get))

    print (found.split("-")[0])
    return (found.split("-")[0])


def find_face_dict_by5(path):
    global img2vec
    global trainDict
    results = []

    queryImage = Image.open(path).convert('RGB')
    vec = img2vec.get_vec(queryImage)

    for i in trainDict.items():
        tmp = np.absolute(np.subtract(vec,i[1]))
        results.append(np.sum(tmp))

    diffs = np.add.reduceat(results, np.arange(0, len(results), 5))
    index_min = np.argmin(diffs)
    print ("s" + str(index_min+1))
    return ("s" + str(index_min+1))


def find_face_dict_cosin(path):
    global img2vec
    global trainDict
    results = {}

    queryImage = Image.open(path).convert('RGB')
    vec = img2vec.get_vec(queryImage)

    for i in trainDict.items():
        results[i[0]] = spatial.distance.cosine(vec, i[1]) 
    found = str(min(results, key=results.get))

    print (found.split("-")[0])
    return (found.split("-")[0])


def find_face_dict_cosin_bestof(path):
    global img2vec
    global trainDict
    results = {}
    od = OrderedDict()

    queryImage = Image.open(path).convert('RGB')
    vec = img2vec.get_vec(queryImage)

    for i in trainDict.items():
        results[i[0]] = spatial.distance.cosine(vec, i[1]) 
        od[i[0]] = spatial.distance.cosine(vec, i[1]) 
    
    sort_orders = sorted(results.items(), key=lambda x: x[1], reverse=False)
    npalist = np.array(sort_orders[:5])
    faces = npalist[:,0]
    tokenized = [i.split('-', 1)[0] for i in faces]
    frequent = max(set(tokenized), key=tokenized.count)
    return frequent


store_olivetti_models_dict()
test_olivetti_models_dict()
