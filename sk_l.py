import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA

pca = PCA(n_components = 200)

components = pca.fit_transform(face_data_standardized)
filtered = pca.inverse_transform(components)
result2 = abs(face_data_standardized - filtered).mean(axis = 1)
