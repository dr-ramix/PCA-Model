from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
boston=load_boston()
x=boston.data
y=boston.target
print(x)
model=PCA(n_components=3)
x=model.fit_transform(x)
print(x)