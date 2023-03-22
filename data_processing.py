import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import stumpy
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import datetime as dt

ck_2020 = pd.read_csv(r'data/ck_2020.csv')
ck_2020.columns
cols = [ 'game_date',	'release_speed',	'release_pos_x',	'release_pos_z',	'description',	'zone',	'stand',	'balls',	'strikes',	'pfx_x'	, 'pfx_z',	'vx0',	'vy0',	'vz0', 'ax',	'ay',	'az',	'effective_speed',	'release_spin_rate',	'release_extension',	'at_bat_number',	'pitch_name',	'if_fielding_alignment',	'of_fielding_alignment',	'spin_axis']
cols_pitch_type = ['pitch_type',	'release_speed','release_spin_rate',	'release_pos_x',	'release_pos_z',	'pfx_x'	, 'pfx_z', 'plate_x','plate_z',	'vx0',	'vy0',	'vz0', 'ax',	'ay',	'az',		'spin_axis']

df = ck_2020[cols_pitch_type].dropna()
df.info()
df['pitch_type'].value_counts()
df.describe()
df.hist(bins=50, figsize=(20,15))
#Release Position of the ball measured in feet from the catcher's perspective.
df.plot(kind='scatter', x='release_pos_x', y='release_pos_z',alpha=0.3)
#Position of the ball when it crosses home plate from the catcher's perspective.
df.plot(kind='scatter', x='plate_x', y='plate_z',alpha=0.3)


"""plot release position"""
ordinal_encoder = OrdinalEncoder()
pitch_type_encoded = ordinal_encoder.fit_transform(df[['pitch_type']].values)
df['pitch_type_encoded'] = pitch_type_encoded
ordinal_encoder.categories_
df.plot(kind='scatter', x='release_pos_x', y='release_pos_z', alpha=0.4, s='release_speed', label='pitch_type', figsize=(10, 7),c=pitch_type_encoded,cmap=plt.get_cmap(name='jet'))
plt.legend()

plt.scatter(x=df['release_pos_x'], y=df['release_pos_z'], s=df['release_speed'],c=pitch_type_encoded, alpha = 0.5)
#ax.set_xlabel(r'x', fontsize=14)
#ax.set_ylabel(r'y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

dot_size = 2*(df['release_speed']*0.1)**2
fig, ax = plt.subplots()
scatter = ax.scatter(x=df['release_pos_x'], y=df['release_pos_z'], s=dot_size,c=pitch_type_encoded, alpha=0.5)
# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="pitch_type")
ax.add_artist(legend1)
# produce a legend with a cross section of sizes from the scatter
#handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
#legend2 = ax.legend(handles, labels, loc="upper right", title="Speed")
plt.show()

"""plot correlation coefficient"""
attributes = ['pitch_type', 'release_speed', 'release_spin_rate', 'spin_axis']
scatter_matrix(frame=df, figsize=(20, 15))
plt.show()

"""PCA"""
attributes_PCA = ['release_speed','release_spin_rate',	'release_pos_x',	'release_pos_z',	'pfx_x'	, 'pfx_z', 'plate_x','plate_z',	'vx0',	'vy0',	'vz0', 'ax',	'ay',	'az',		'spin_axis']
pca = PCA(n_components=2)
pca_data = pca.fit_transform(df[attributes_PCA])
pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x=pca_data[:,0], y=pca_data[:,1],c=pitch_type_encoded)
#plt.scatter(x=pca_data[:,0], y=pca_data[:,1], c=pitch_type_encoded)
ax.set_xlabel(r'x', fontsize=14)
ax.set_ylabel(r'y', fontsize=14)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="pitch_type")
ax.add_artist(legend1)
plt.show()

#
ck_2020 = ck_2020[cols].dropna().loc[::-1].reset_index(drop=True)
df_reverseed = df[attributes_PCA].loc[::-1].reset_index(drop=True)
pca = PCA(n_components=1)
pca_data = pca.fit_transform(df_reverseed)
pca.explained_variance_ratio_

plt.figure(figsize=(15,10))
plt.xlabel('Time')
plt.ylabel('PCA 1st Comp')
plt.plot(pca_data)
plt.show()

#MP
plt.style.use('https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/stumpy.mplstyle')
ck_2020['game_date'].value_counts().mean()
m = 89
mp = stumpy.stump(pca_data[:,0], m)
#Discord
discord_idx = np.argsort(mp[:, 0])[-1]
print(f"The discord is located at index {discord_idx}")
nearest_neighbor_distance = mp[discord_idx, 0]
print(f"The nearest neighbor subsequence to this discord is {nearest_neighbor_distance} units away")
fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Discord (Anomaly/Novelty) Discovery', fontsize='30')
axs[0].plot(pca_data[:,0])
axs[0].set_ylabel('pca', fontsize='20')
rect = Rectangle((discord_idx, -500), m, 700, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile', fontsize='20')
axs[1].axvline(x=discord_idx, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()

ck_2020.iloc[discord_idx:discord_idx+m,]
#he has the 3 ER (second highest), 2 BB (tie highest), 3 SO (tie lowest) on 09/09 game

#kmeans
kmeans = KMeans(n_clusters=3)
prediction = kmeans.fit_predict(pca_data)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x=pca_data[:,0], y=pca_data[:,1],c=prediction)
#plt.scatter(x=pca_data[:,0], y=pca_data[:,1], c=pitch_type_encoded)
ax.set_xlabel(r'x', fontsize=14)
ax.set_ylabel(r'y', fontsize=14)
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper right", title="pitch_type")
ax.add_artist(legend1)
plt.show()