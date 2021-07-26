# IN THIS PYTHON CODE WE WOULD TRAIN A
# KNN MACHINE TO PREDICT WEATHER A PERSON
# MAKES 50K A YEAR OR NOT

# WE USE THESE PRINT TO SEE WHAT STATUS WE HAVE
# AND WHERE WE ARE
print("PHASE 1 : Importing modules")

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# A DICT OBJECT TO SAVE RESULTS OF EACH TRAIN
# IN AN ALGORYTM TO LOWER THE ERROR
Outputs = {}

# THE ALGORYTHM USES THIS AS THE MAX VALUE OF
# K TO TEST
max_k_value = 50

# PANDAS WILL ALWAYS TAKE THE FIRST LINE IN THE
# CSV FILE AS THE COLUMN NAMES BUT IN THIS DATASET
# FIRST LINE CONTAINS DATA NOT THE COLUMN NAMES SO
# WE SET THEM MANNUALLY
Columns = ['age', 'workclass', 'fnlwgt', 'education',
           'education-num', 'marital-stauts', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain',
           'capital-loss', 'hours-per-week', 'native-country',
           'income']

# A FUNCTION TO TAKE THE DICT OUTPUT, SAVE AND
# APPEND TO THE PREVIOUS SAVED DATAS
# ALSO CREATES A SUBDIRECTORY NAMED 'Output'
# SO WE DONT MESS THINGS UP
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
    # READ DATASET
    print("PHASE 2 : Reading dataset")
    df = pd.read_csv('adult.data', names=Columns)
    
    Outputs['Columns'] = Columns

    # AT THIS POINT WE DROP SOME COLUMNS TO BE ABLE
    # TO TRAIN THE SYSTEM BUT WE CAN DO MORE PROCESSING
    # AND USE THESE DATAS TOO
    print("PHASE 3 : Preprocessing dataset")
    df.drop(['workclass', 'fnlwgt', 'education', 'marital-stauts', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','native-country'], inplace=True, axis=1)

    # SINCE INCOME COLUMN IS CATEGORICAL WE SHOULD
    # CONVERT IT TO A COLUMN OF ZEROS AND ONES (BOOLEANS)
    # TO BE ABLE TO TRAIN THE MACHINE
    df['income >50k'] = pd.get_dummies(df['income'], drop_first=True)
    df.drop('income',inplace=True, axis=1)

    # HERE WE TRAIN A BASIC SYSTEM ON THE DATASET
    print("PHASE 4 : Base Training model")

    # SPLITTING TO TEST AND TRAIN SETS
    X_train, X_test, y_train, y_test = train_test_split(df[['age','education-num','hours-per-week']], df['income >50k'], test_size=0.3, random_state=42)

    print("PHASE 5 : Calculating error of base model")
    KNN1 = KNeighborsClassifier(n_neighbors=1)
    KNN1.fit(X_train, y_train)
    predictions = KNN1.predict(X_test)
    Outputs['Confusion Matrix 0'] = confusion_matrix(y_test, predictions)
    Outputs['Classification Report 0'] = classification_report(y_test, predictions)



    print("PHASE 6 : Finding the best value for K")
    error_rate = []
    # THIS IS SIMPLE WE TAKE K VALUE WE TRAIN WE TEST ERROR
    # THE K VALUE WHICH MAKES LEAST ERRROR WILL BE CHOSEN
    for i in range(1,max_k_value):
        KNN = KNeighborsClassifier(n_neighbors=i)
        KNN.fit(X_train, y_train)
        prediction = KNN.predict(X_test)
        error_rate.append(np.mean(prediction != y_test))

        Outputs['Confusion Matrix %d' % i] = confusion_matrix(y_test, prediction)
        Outputs['Classification Report %d' % i] = classification_report(y_test, prediction)

    # PLOTTING K VALUES AND ERRORS
    print("PHASE 7 : Plotting K values and errors")
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(range(1,max_k_value),error_rate, marker='o')
    ax.set_xlabel('K Value')
    ax.set_ylabel('Error rate')
    ax.set_title('Choosing the K value')
    plt.savefig('K Value.png', dpi=100)

    print(error_rate.index(min(error_rate)) + 1, min(error_rate))

    # Retraining with the best K value
    print("PHASE 8 : Retraining with the best K value")
    KNN2 = KNeighborsClassifier(n_neighbors=45)
    KNN2.fit(X_train, y_train)
    predictions2 = KNN2.predict(X_test)

    # SAVING ALL THE RESULTS
    print("PHASE 9 : Saving results")
    Outputs['Confusion Matrix Last'] = confusion_matrix(y_test, predictions2)
    Outputs['Classification Report Last'] = classification_report(y_test, predictions2)
    SaveOutput(Outputs)
    return 'Done'

# RUN MAIN
if __name__ == '__main__':
    Main()