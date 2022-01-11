# personDB

In this repository I am sharing demos to test modelling and classification algorithms for face recognition. At the moment the typical pipeline is not implemented in all its components. In particular, I will start with a simple experiment:

- The ["The ORL Database of Faces"](https://cam-orl.co.uk/facedatabase.html) is used to train and test the system. The ORL database is among the simplest face databases, made up of pictures of 40 individuals, 10 pictures each, for a total of 400 pictures, 92x112 black and white bitmaps. The faces are already aligned and normalized and ready to be processed by a feature extraction algorithm.
- For the training, I am using 5 images, and the remaining 5 images are used for testing through the classificating algorithm.
- In the current demo, the model extraction is performed using Dense Convolutional Network in the [Pytorch](https://pytorch.org/hub/pytorch_vision_densenet/) library. 
- One feature vector is extracted for every face using Christian Safka's wrapper library [img2vec](https://github.com/christiansafka/img2vec), the feature is stored in a dictionary in the form `{'person1_face1':'vector', 'person1_face2':'vector', ...,'person2:face1':'vector',...}`
- The classification is performed using different approaches, being the cosin similarity the most effective (98% of the 200 test faces are recognized) to calculate the similarity of a test vector extracted from the face under test to the face from the training set.
- Classification is performed iteratively, which is a quite slow approach, but it is sufficient for the sake of showing how a full system can be coded in a few lines of code

```
┌────────────────┐      ┌────────────────┐     ┌────────────────┐    ┌────────────────┐
│     FACE       ├─────►│  NORMALIZATION ├────►│    FEATURE     ├───►│    FEATURE     │
│  DETECTION     │      │                │     │   EXTRACTION   │    │     STORE      │
└────────────────┘      └────────────────┘     └───────┬────────┘    └───────▲────────┘
                                                       │                     │
                                                       │             ┌───────┴────────┐
                                                       └────────────►│ CLASSIFICATION │
                                                                     └────────────────┘
```

# Usage

Clone the repository, setup a virtual environment and run the script:

```
python3 -m venv myvenv
source myvenv/bin/activate
pip install img2vec_pytorch
pip install scipy
```

Then run the script under the `demo` folder:

```
python3 olivetti_test.py
```

# Future developments

In the future I will test different models, such as multi-vector feature extraction for a single face, or different kind of indexing or even a neural network to speed up the classification of testing images.

## References

- https://github.com/christiansafka/img2vec
- https://cam-orl.co.uk/facedatabase.html
