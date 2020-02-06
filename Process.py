import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

Outputs = {}

Columns = ['age', 'workclass', 'fnlwgt', 'education',
           'education-num', 'marital-stauts', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain',
           'capital-loss', 'hours-per-week', 'native-country',
           'income']

def SaveOutput(Objects : dict) :
    try: os.chdir('Output')
    except: 
        os.mkdir('Output')
        os.chdir('Output')
    SaveFile = open('Save.txt','w')
    for i in Objects:
        SaveFile.write('%s : %s \n' %(i,str(Objects[i])))
    SaveFile.close()
    return 'SAVED'


def Main() :
    df = pd.read_csv('adult.data', names=Columns)
    
    Outputs['Columns'] = Columns

    df.drop(['workclass', 'fnlwgt', 'education', 'marital-stauts', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','native-country'], inplace=True, axis=1)

    df['income >50k'] = pd.get_dummies(df['income'], drop_first=True)
    df.drop('income',inplace=True, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(df[['age','education-num','hours-per-week']], df['income >50k'], test_size=0.3, random_state=42)

    KNN1 = KNeighborsClassifier(n_neighbors=1)
    KNN1.fit(X_train, y_train)
    predictions = KNN1.predict(X_test)
    Outputs['Confusion Matrix 0'] = confusion_matrix(y_test, predictions)
    Outputs['Classification Report 0'] = classification_report(y_test, predictions)


    error_rate = []

    for i in range(1,50):
        KNN = KNeighborsClassifier(n_neighbors=i)
        KNN.fit(X_train, y_train)
        prediction = KNN.predict(X_test)
        error_rate.append(np.mean(prediction != y_test))

        Outputs['Confusion Matrix %d' % i] = confusion_matrix(y_test, prediction)
        Outputs['Classification Report %d' % i] = classification_report(y_test, prediction)

    fig = plt.figure(figsize=(15,5))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(range(1,50),error_rate, marker='o')
    ax.set_xlabel('K Value')
    ax.set_ylabel('Error rate')
    ax.set_title('Choosing the K value')
    plt.savefig('K Value.png', dpi=200)

    print(error_rate.index(min(error_rate)) + 1, min(error_rate))


    KNN2 = KNeighborsClassifier(n_neighbors=45)
    KNN2.fit(X_train, y_train)
    predictions2 = KNN2.predict(X_test)

    Outputs['Confusion Matrix Last'] = confusion_matrix(y_test, predictions2)
    Outputs['Classification Report Last'] = classification_report(y_test, predictions2)
    SaveOutput(Outputs)
    return 'Done'


if __name__ == '__main__':
    Main()