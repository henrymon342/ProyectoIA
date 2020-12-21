import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# gender
# race/ethnicity 
# parental_level_of_education 
# lunch 
# test_preparation_course
# math_score
# reading_score
# writing_score 

data = pd.read_csv("StudentsPerformance.csv")
# print(data.tail())



#PREPROCESAMIENTO
#"math score","reading score","writing score"
promath = data['math score'].mean()
proreading = data['reading score'].mean()
prowriting = data['writing score'].mean()
data['math score'].replace(np.nan, promath)
data['reading score'].replace(np.nan, proreading)
data['writing score'].replace(np.nan, prowriting)

header = ['genero', 'raza', 'nivel de educacion padres', 'almuerzo', 'test de preparacion', 'nota matematicas', 'nota lectura', 'nota escritura']
data.columns = header

data = pd.get_dummies(data, columns = ['genero'], drop_first = True)
data = pd.get_dummies(data, columns = ['test de preparacion'], drop_first = True)

print(data)


df = pd.DataFrame(data)
datos = df.iloc[:,:].values
print(datos)
x = datos[:,5:8]
y = datos[:,0:1]
print(x)
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print("X_train")
print(X_train)
print("y_train")
print(y_train)