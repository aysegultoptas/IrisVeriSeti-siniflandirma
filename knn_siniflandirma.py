import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

veri = pd.read_csv('Iris.csv')

tur = veri.iloc[:,-1:].values

x_egitim, x_test, y_egitim, y_test = train_test_split(veri.iloc[:,1:-1],tur,test_size = 0.7,random_state=0)

knn = KNeighborsClassifier(n_neighbors=5 ,metric="euclidean")

knn.fit(x_egitim,y_egitim.ravel())

sonuc = knn.predict(x_test) 

basari_orani = accuracy_score(y_test, sonuc)

print("\n\nK-NN algoritması başarı oranı: ")
print(basari_orani)

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, sonuc)

print("\n\nK-NN algoritması karmaşıklık matrisi: ")
print(matrix)










