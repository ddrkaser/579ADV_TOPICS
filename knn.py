from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import neighbors, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay


df_input = pd.DataFrame(pca_data)
df_input['pitch_type'] = pitch_type_encoded.flatten()

train_set, test_set = train_test_split(df_input, test_size=0.2, random_state=42)
train_X = train_set.iloc[:,:2]
train_Y = train_set.iloc[:,2]
test_X = test_set.iloc[:,:2]
test_Y = test_set.iloc[:,2]

neigh = KNeighborsClassifier(n_neighbors=8, weights="distance")
neigh.fit(train_X,train_Y)
neigh.score(test_X, test_Y)

result = neigh.predict(test_X[:5])
test_Y[:5]

#find best k
score = []
for k in range(1,21):
    neigh = KNeighborsClassifier(n_neighbors=k, weights="distance")
    neigh.fit(train_X,train_Y)
    score.append(neigh.score(test_X, test_Y))

plt.plot(np.arange(20) + 1, score)
plt.xticks(np.arange(0, 21, 2.0))
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()

#plot boundary
# Create color maps
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ["darkorange", "c", "darkblue"]


n_neighbors =8
for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(train_X, train_Y)
    print(clf.score(test_X, test_Y))

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        train_X,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        #xlabel='x',
        #ylabel='y',
        shading="auto",
    )

    # Plot also the training points
    sns.scatterplot(
        x=train_X.iloc[:, 0],
        y=train_X.iloc[:, 1],
        hue=ordinal_encoder.categories_[0][train_Y.values.astype(int)],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

plt.show()

