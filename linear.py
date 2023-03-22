import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, mutual_info_regression,RFE, SelectKBest, SelectFromModel

ck_2022 = pd.read_csv(r'data/ck_2022.csv')
cols_pitch_type = ['pitch_type',	'release_speed','release_spin_rate',	'release_pos_x',	'release_pos_z',	'pfx_x'	, 'pfx_z', 'plate_x','plate_z',	'vx0',	'vy0',	'vz0', 'ax',	'ay',	'az',		'spin_axis']

df = ck_2022[cols_pitch_type].dropna()

#drop outlier
df = df[df['release_pos_x']<2]
df = df[df['pitch_type'].isin(['SL','FF','CU'])]

ordinal_encoder = OrdinalEncoder()
pitch_type_encoded = ordinal_encoder.fit_transform(df[['pitch_type']].values)
df['pitch_type_encoded'] = pitch_type_encoded
ordinal_encoder.categories_

attributes_input = ['release_speed','release_spin_rate',	'release_pos_x',	'release_pos_z',	'pfx_x'	, 'pfx_z', 'plate_x','plate_z',	'vx0',	'vy0',	'vz0', 'ax',	'ay',	'az',		'spin_axis']

df_input = df[attributes_input]

#standardization
scaler = StandardScaler()
df_std = scaler.fit_transform(df_input)
df_std = pd.DataFrame(df_std)
#split train/test
df_std['pitch_type'] = pitch_type_encoded.flatten()
train_set, test_set = train_test_split(df_std, test_size=0.2, random_state=42)
train_X = train_set.iloc[:,:-1]
train_Y = train_set.iloc[:,-1]
test_X = test_set.iloc[:,:-1]
test_Y = test_set.iloc[:,-1]

lin_reg = LinearRegression()
lin_reg.fit(X=train_X, y=train_Y)
lin_reg.score(test_X, test_Y)

f_test, p_values = f_regression(train_X, train_Y)
mi = mutual_info_regression(train_X, train_Y)
#round the result, because we are using linear regression
result = np.around(lin_reg.predict(test_X[:5]))
test_Y[:5]

#RFE
def k_feature_scores(train_X,train_Y,test_X,test_Y, k):
    rfe_selector = RFE(estimator=LinearRegression(),n_features_to_select = k, step = 1)
    rfe_selector.fit(train_X, train_Y)
    train_X.columns[rfe_selector.get_support()]
    train_X_k = train_X.iloc[:,rfe_selector.get_support()]
    lin_reg = LinearRegression()
    lin_reg.fit(X=train_X_k, y=train_Y)
    return lin_reg.score(test_X.iloc[:,rfe_selector.get_support()], test_Y)

def plot_score():
    scores = []
    for i in range(1,train_X.shape[1]):
        scores.append(k_feature_scores(train_X,train_Y,test_X,test_Y, i))
        print('Feature %d: %f' % (i, scores[i-1]))
    # plot the scores
    plt.bar(np.arange(1,train_X.shape[1]), scores, align='center')
    plt.show()



