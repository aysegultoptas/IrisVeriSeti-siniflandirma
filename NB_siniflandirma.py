import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

veri = pd.read_csv('Iris.csv')

tur = veri.iloc[:,-1:].values

x_egitim, x_test, y_egitim, y_test = train_test_split(veri.iloc[:,1:-1],tur,test_size=0.7,random_state=0)

gnb = GaussianNB()

gnb.fit(x_egitim, y_egitim.ravel())

sonuc = gnb.predict(x_test)

basari_orani = accuracy_score(y_test, sonuc)

print("\n\nNaive Bayes algoritması başarı oranı: ")
print(basari_orani)

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, sonuc) 

print("\n\nNaive Bayes algoritması karmaşıklık matrisi: ")
print(matrix)




