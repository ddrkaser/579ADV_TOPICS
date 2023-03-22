import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, mutual_info_regression,RFE, SelectKBest, SelectFromModel
from sklearn.naive_bayes import GaussianNB

ck_2022 = pd.read_csv(r'data/ck_2022.csv')
cols_pitch_type = ['pitch_type',	'release_speed','release_spin_rate',	'release_pos_x',	'release_pos_z',	'pfx_x'	, 'pfx_z', 'plate_x','plate_z',	'vx0',	'vy0',	'vz0', 'ax',	'ay',	'az',		'spin_axis']

df = ck_2022[cols_pitch_type].dropna()
#df = ck_2020[cols_pitch_type].dropna()

#drop outlier
df = df[df['release_pos_x']<2]
df = df[df['pitch_type'].isin(['SL','FF','CU'])]

ordinal_encoder = OrdinalEncoder()
pitch_type_encoded = ordinal_encoder.fit_transform(df[['pitch_type']].values)
df['pitch_type_encoded'] = pitch_type_encoded
ordinal_encoder.categories_

attributes_input = ['release_speed','release_spin_rate',	'release_pos_x',	'release_pos_z',	'pfx_x'	, 'pfx_z', 'plate_x','plate_z',	'vx0',	'vy0',	'vz0', 'ax',	'ay',	'az',		'spin_axis']

df_input = df[attributes_input]


#split train/test
df_input['pitch_type'] = pitch_type_encoded.flatten()
train_set, test_set = train_test_split(df_input, test_size=0.2, random_state=12)
train_X = train_set.iloc[:,:-1]
train_Y = train_set.iloc[:,-1]
test_X = test_set.iloc[:,:-1]
test_Y = test_set.iloc[:,-1]

#standardization
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
train_X = pd.DataFrame(train_X)
test_X = scaler.fit_transform(test_X)
test_X = pd.DataFrame(test_X)

gnb = GaussianNB()
gnb.fit(X=train_X, y=train_Y)
gnb.score(test_X, test_Y)
#socre 0.99726

result = gnb.predict(test_X[:5])
test_Y[:5]
