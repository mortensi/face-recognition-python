# personDB

In this repository I am sharing demos to test modelling and classification algorithms for face recognition. At the moment the typical pipeline is not implemented in all its components. In particular, I will start with a simple experiment:

- The ["The ORL Database of Faces"](https://cam-orl.co.uk/facedatabase.html) is used to train and test the system. ORL database is among the most simple face databases, made up of pictures of 40 individuals, 10 pictures each, for a total of 400 pictures, 92x112 black and white bitmaps. The faces are already normalized and ready to be processed by a feature extraction algorithm
- For the training, I am using 5 images, and the remaining 5 images are used for testing the classificator.
- In the current demo, the model extraction is performed using Dense Convolutional Network in the [Pytorch](https://pytorch.org/hub/pytorch_vision_densenet/) library. 
- One feature vector is extracted for every face using Christian Safka's wrapper library [img2vec](https://github.com/christiansafka/img2vec), the feature is stored in a dictionary in the form {'person1_face1':'vector', 'person1_face2':'vector', ...,'person2:face1':'vector',...}
- The classification is performed using different approaches, being the cosin similarity the most effective (98% of the 200 test faces are recognized)

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

## References

https://github.com/christiansafka/img2vec
https://cam-orl.co.uk/facedatabase.html
