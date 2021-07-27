# IN THIS PYTHON CODE WE WOULD TRAIN A
# KNN MACHINE TO PREDICT WEATHER A PERSON
# MAKES 50K A YEAR OR NOT
#
# CODING POLICY IS FUNCTIONAL PROGRAMMING
# BECAUSE WE GET LESS GLOBAL VARAIABLES AND
# MORE LOCAL, AND IN RESULT LESS RAM USAGE
# DURING EXECUTION

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
import joblib


Main_Dir = 'E:/Archives/Documents/Python/Adults/Adults-dataset/Adults-dataset'
Dataset_Dir = '%s/SRC/DataFiles/Dataset.csv' % Main_Dir
Machine_Save_Dir = '%s/Output/KNN - Machine.sav' % Main_Dir

# THE ALGORYTHM USES THIS AS THE MAX VALUE OF
# K TO TEST
max_k_value = 3
min_k_value = 2

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
    Output_Dir = '%s/Output' % Main_Dir
    try: os.chdir(Output_Dir)
    except: 
        os.mkdir(Output_Dir)
        os.chdir(Output_Dir)
    SaveFile = open('Trained_Machines_Reports.txt','w')
    for i in Objects:
        SaveFile.write('%s : %s \n' %(i,str(Objects[i])))
    SaveFile.close()
    return 'SAVED'

# PLOTTING K VALES AND ERRORS
def PlotKValues(error_rate : list, k_range : range):
    # PLOT COLOR CONFIGURATIONS
    Plot_Color = (0.2, 0.2, 0.2, 0.8)

    # CREATE FIGURE OBJ
    fig = plt.figure(figsize=(15,5))

    # ADD AXES
    ax = fig.add_axes([0.1,0.1,0.8,0.8], facecolor=Plot_Color)

    # PLOT
    ax.plot(k_range, error_rate, marker='o')

    # SETTINGS
    ax.set_xlabel('K Value')
    ax.set_ylabel('Error rate')
    ax.set_title('Choosing the K value')
    ax.set_xlim(min_k_value, max_k_value)

    # SAVE
    plt.savefig('%s/Output/K Value.png' % Main_Dir, dpi=100, facecolor=Plot_Color, edgecolor=Plot_Color)

    return 'PLOTTED'

def Train_Test_Machine(Datas : dict, Outputs : dict, K : int, SAVE : bool) :
    # DEFINE MACHINE WITH K VALUE
    KNN1 = KNeighborsClassifier(n_neighbors=K)

    # TRAIN MACHINE
    KNN1.fit(Datas['X_Train'], Datas['Y_Train'])

    # FEED TEST DATA TO MACHINE AND RECORD PREDICTIONS
    predictions = KNN1.predict(Datas['X_Test'])

    # CALCULATE ERRORS
    Outputs['Confusion Matrix %d' % K] = confusion_matrix(Datas['Y_Test'], predictions)
    Outputs['Classification Report %d' % K] = classification_report(Datas['Y_Test'], predictions)
    error = np.mean(predictions != Datas['Y_Test'])

    # SAVING BEST MACHINE
    if SAVE :
        Machine_Save_File = open(Machine_Save_Dir, 'wb')
        joblib.dump(KNN1, Machine_Save_File)
        Machine_Save_File.close()

    return error

def Main_Process() :
    # DEFINIG VARIABLES LOCAL TO THIS
    # FUNCTION :
    # A DICT OBJECT TO SAVE RESULTS OF EACH TRAIN
    # IN AN ALGORYTM TO LOWER THE ERROR
    Outputs = {}
    Train_Test_Datas = {}
    error_rate = []
    # READ DATASET
    print("PHASE 2 : Reading dataset")
    df = pd.read_csv(Dataset_Dir, names=Columns)
    
    Outputs['Columns'] = Columns

    # AT THIS POINT WE DROP SOME COLUMNS TO BE ABLE
    # TO TRAIN THE SYSTEM BUT WE CAN DO MORE PROCESSING
    # AND USE THESE DATAS TOO
    print("PHASE 3 : Preprocessing dataset")

    # THE LIST BELLOW CONTAINS COLUMNS THAT NEED TO BE
    # CONVERTED TO ZEROS AND ONES
    columns_to_convert = ['workclass', 'education', 'marital-stauts',
                         'occupation', 'relationship', 'race',
                         'sex','native-country', 'income']
    for temp_column in columns_to_convert :
        print('PHASE 3 : Processing Column %s' % temp_column)
        df = df.join(pd.get_dummies(df[temp_column], drop_first=True))
        df.drop(temp_column, axis=1, inplace=True)

    # HERE WE TRAIN A BASIC SYSTEM ON THE DATASET
    print("PHASE 4 : Splitting Dataset")

    # SPLITTING TO TEST AND TRAIN SETS
    x_train, x_test, y_train, y_test = train_test_split(df.drop(' >50K', axis=1), df[' >50K'], test_size=0.3, random_state=42)

    # AT THIS POINT WE ADD THEM ALL TO A DICT OBJ
    # SO THAT WE GET A PRETTIER AND CLEANER FUNCTION 
    # CALL BUT HANDLE THE SAME STUFF AT Train_Test_Machine()
    Train_Test_Datas['X_Train'] = x_train
    Train_Test_Datas['X_Test'] = x_test
    Train_Test_Datas['Y_Train'] = y_train
    Train_Test_Datas['Y_Test'] = y_test

    print("PHASE 5 : Training base model, K=1")
    Train_Test_Machine(Train_Test_Datas, Outputs, 1, False)



    print("PHASE 6 : Finding the best value for K")


    # SETTING K RANGE
    K_range = range(min_k_value, max_k_value + 1)
    # THIS IS SIMPLE WE TAKE K VALUE WE TRAIN WE TEST ERROR
    # THE K VALUE WHICH MAKES LEAST ERRROR WILL BE CHOSEN
    for Temp_K in K_range:
        print('PAHSE 6 : Training K=%d' % Temp_K)
        error = Train_Test_Machine(Train_Test_Datas, Outputs, Temp_K, False)
        error_rate.append(error)
        print('PHASE 6 : k = %d error = %f' % (Temp_K, error_rate[Temp_K - min_k_value] * 100))


    # PLOTTING K VALUES AND ERRORS
    print("PHASE 7 : Plotting K values and errors")
    PlotKValues(error_rate, K_range)

    # THE LINE BELLOW :
    # error_rate.index(min(error_rate)) + min_k_value
    # WILL RETURN THE BEST K VALUE
    # SINCE min WILL RETURN LOWEST VALUE IN LIST
    # WE USE index TO GET THE INDEX OF IT BUT THIS
    # INDEX + min_k_value IS THE ACTUAL K
    Best_K_Value = error_rate.index(min(error_rate)) + min_k_value
    print(Best_K_Value, min(error_rate))
    
    # SAVING ALL THE RESULTS AND MACHINE
    print("PHASE 8 : Saving results")
    SaveOutput(Outputs)

    
    # Retraining with the best K value
    print("PHASE 8 : Retraining with the best K value")
    last_error = Train_Test_Machine(Train_Test_Datas, Outputs, Best_K_Value, True)
    error_rate.append(last_error)



    return 'Done'

# RUN MAIN
if __name__ == '__main__':
    print(Main_Process())
