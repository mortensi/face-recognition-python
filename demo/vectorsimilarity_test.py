#!/usr/bin/python3
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.query import Query
from img2vec_pytorch import Img2Vec
from PIL import Image
import numpy as np

NUMBER_SAMPLES = 200
FACE_IMAGE_VECTOR_FIELD='face_image_vector'
IMAGE_VECTOR_DIMENSION=1024

r = redis.Redis(host='127.0.0.1', port=6379, password='')
r.flushdb()

img2vec = Img2Vec(cuda=False, model='densenet')

def store_olivetti_models():
    global r
    global img2vec
    
    for person in range(1, 41):
        person = "s" + str(person)
        for face in range(1, 6):
            facepath = '../olivetti-database/' + person + "/" + str(face) + '.bmp'
            print ("Training face: " + facepath)
            img = Image.open(facepath).convert('RGB')
            vec = img2vec.get_vec(img)
            face_image_vector = vec.astype(np.float32).tobytes()
            #print(format(face_image_vector,'x'))
            face_data_values ={ 'person_id':person,
                                'person_path':facepath,
                                  FACE_IMAGE_VECTOR_FIELD:face_image_vector}
            r.hset('face_'+person+'_'+str(face),mapping=face_data_values)


def test_olivetti_models_vect():
    success = 0
    for person in range(1, 41):
        person = "s" + str(person)
        for face in range(6, 11):
            facepath = '../olivetti-database/' + person + "/" + str(face) + '.bmp'
            print ("Testing face: " + facepath)
            found = find_face(facepath)
            if (person == found):
                success = success +1

    print(success/200*100)


def create_hnsw_index (redis_conn,index_name,vector_field_name,number_of_vectors, vector_dimensions=IMAGE_VECTOR_DIMENSION, distance_metric='L2',M=40,EF=200):
    global r
    schema = (VectorField("face_image_vector", "HNSW", {"TYPE": "FLOAT32", "DIM": IMAGE_VECTOR_DIMENSION, "DISTANCE_METRIC": "L2"}),)
    hnsw_index = r.ft().create_index(schema)
    return hnsw_index


def find_face(path):
    global r
    global img2vec

    img = Image.open(path).convert('RGB')
    vec = img2vec.get_vec(img)
    face_image_vector = vec.astype(np.float32).tobytes()

    q = Query("*=>[KNN 1 @face_image_vector $vec]").return_field("__face_image_vector_score")
    res = r.ft().search(q, query_params={"vec": face_image_vector})

    for face in res.docs:
        print ('*** Face found ***')
        print(face.id.split("_")[1])
        return face.id.split("_")[1]

my_hnsw_index = create_hnsw_index(r,'my_hnsw_index',FACE_IMAGE_VECTOR_FIELD,NUMBER_SAMPLES,IMAGE_VECTOR_DIMENSION,'L2',M=40,EF=200)
store_olivetti_models()
test_olivetti_models_vect()
