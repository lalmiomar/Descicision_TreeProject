

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import stopword
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

import re

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

from sklearn.model_selection import  train_test_split

# from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from  sklearn.metrics import classification_report,confusion_matrix

data_frame = pd.read_excel('Datasets/fake_new_dataset.xlsx')
# print(data_frame.head())
# print(data_frame.shape)
# print(data_frame.columns)
#
# print(data_frame)
# # print(stopwords.words('english'))
# # print(data_frame.iloc[0,2])
#
# phrase_brute = data_frame.iloc[0,2]
# # print(phrase_brute)
# phrase_brute = re.sub('[^a-zA-Z]', ' ', phrase_brute)
# # print(phrase_brute)
# phrase_brute = phrase_brute.lower()
# # print(phrase_brute)
#
# phrase_brute = phrase_brute.split()
# print(phrase_brute)
# for i in phrase_brute:
#      print(i)
#
# phrase_proper = []
#
# for word in phrase_brute:
#     if word not in set(stopwords.words('english')):
#         phrase_proper.append(word)
#
# print(phrase_proper)
#
# phrase_proper = " ".join(phrase_proper)
# print(phrase_proper)
# print(data_frame.shape)
corpus = []

# print(data_frame.iloc[:,2])
# for i in range(3119):
#     contex = re.sub('[^a-zA-Z]', ' ', data_frame.iloc[i, 2])
#     corpus = corpus.append(contex)
#
# print(corpus)
# print(data_frame.iloc[:,-1])

for i in range(3119):
    contex = re.sub('[^a-zA-Z]', ' ', data_frame.iloc[i,2])
    contex = contex.lower()
    contex = contex.split()
    ps = PorterStemmer()
    contex = [ps.stem(word) for word in contex if word not in set(stopwords.words('english'))]
    contex = " ".join(contex)
    corpus.append(contex)

print(corpus)
# print(corpus.shape())
x = cv.fit_transform(corpus).toarray()
# print(x)

y = data_frame.iloc[:,-1]
# print(y)
# # print(cv.get_feature_names())
#
x_data_frame = pd.DataFrame(data=x, columns=cv.get_feature_names())
y_data_frame = pd.DataFrame(data=y.values, columns=['Target'])
# print(x_data_frame)
# print(y_data_frame)
data_frame1 = pd.concat([x_data_frame,y_data_frame], axis=1)
# print(data_frame1.T)

x__df = data_frame1.iloc[:,0:-1]
y__df = data_frame1.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x__df, y__df, train_size=0.8, random_state=1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = DecisionTreeClassifier()


#Training
model.fit(x_train,y_train)
# prediction
y_pred = model.predict(x_test)
print('y_pred: ',y_pred)

# Test or evaluate
print('Accuracy of train:', model.score(x_train, y_train))
# Test of training model
print('Accuracy of test: ', model.score(x_test, y_test))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#
plt.figure()
sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d')
plt.xlabel('Target')
plt.ylabel('outcome')
plt.show()