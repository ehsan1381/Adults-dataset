# IN THIS PYTHON CODE WE WOULD TRAIN A
# KNN MACHINE TO PREDICT WEATHER A PERSON
# MAKES 50K A YEAR OR NOT
#
# CODING POLICY IS FUNCTIONAL PROGRAMMING
# BECAUSE WE GET LESS GLOBAL VARAIABLES AND
# MORE LOCAL, AND IN RESULT LESS RAM USAGE
# DURING EXECUTION

import os
Main_Dir = 'E:/Archives/Documents/Python/Adults/Adults-dataset/Adults-dataset'
Dataset_Dir = '%s/SRC/DataFiles/Dataset.csv' % Main_Dir
Machine_Save_Dir = '%s/Output' % Main_Dir

# THE ALGORYTHM USES THIS AS THE MAX VALUE OF
# K TO TEST
max_k_value = 25
min_k_value = 20
Phase = 1

# DEFINITION OF Print_Log
def Print_Log(Log_String : str, Phase : int) :
    global Algorythm

    Output_Dir = '%s/Output' % Main_Dir
    try: os.chdir(Output_Dir)
    except:
        os.mkdir(Output_Dir)
        os.chdir(Output_Dir)

    if Phase == 0 :
        Log_File = open('DataLog - ( %s ).txt' % Algorythm, 'a')
        Log_File.write('%s\n' % Log_String)
        print('%s' % Log_String)
        Log_File.close()
    
    else :
        Log_File = open('DataLog - ( %s ).txt' % Algorythm, 'a')
        Log_File.write('PHASE %d : %s\n' % (Phase, Log_String))
        print('PHASE %d : %s' % (Phase, Log_String))
        Log_File.close()
    return 0

# DETERMINE AGLORYTM
print('Enter Algorythm to Train : ( K-Neareset-Neighbor Gaussian-Naive-Bayes Logistic-Classifier Support-Vector-Machine )')
print('Enter Capital Letters of Algorythm eg : KNN for K-Nearest-Neighbors')
print('* ALL to Run All Algorythms')
Algorythm = input('Algorythm : ')

# WE USE THESE PRINT TO SEE WHAT STATUS WE HAVE
# AND WHERE WE ARE
Print_Log("Importing modules", Phase)

# NEXT PHASE
Phase += 1

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

# PANDAS WILL ALWAYS TAKE THE FIRST LINE IN THE
# CSV FILE AS THE COLUMN NAMES BUT IN THIS DATASET
# FIRST LINE CONTAINS DATA NOT THE COLUMN NAMES SO
# WE SET THEM MANNUALLY
Columns = ['age', 'workclass', 'fnlwgt', 'education',
           'education-num', 'marital-stauts', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain',
           'capital-loss', 'hours-per-week', 'native-country',
           'income']


# THE LIST BELLOW CONTAINS COLUMNS THAT NEED TO BE ENCODED
# OR SIMPLY CATEGORICAL COLUMNS
Columns_to_Encode = ['workclass', 'education', 'marital-stauts',
                         'occupation', 'relationship', 'race',
                         'sex','native-country', 'income']

# A FUNCTION TO TAKE THE DICT OUTPUT, SAVE AND
# APPEND TO THE PREVIOUS SAVED DATAS
# ALSO CREATES A SUBDIRECTORY NAMED 'Output'
# SO WE DONT MESS THINGS UP
def SaveOutput(Objects : dict) :
    global Algorythm

    Output_Dir = '%s/Output' % Main_Dir
    try: os.chdir(Output_Dir)
    except:
        os.mkdir(Output_Dir)
        os.chdir(Output_Dir)
    SaveFile = open('Reports - %s.txt' % Algorythm,'w')
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


    

def K_Neareset_Neighbor_Machine(Datas : dict, Outputs : dict, K : int, SAVE : bool) :
    # PHASE VARIABLE
    global Phase

    # DEFINE MACHINE WITH K VALUE
    KNN = KNeighborsClassifier(n_neighbors=K)

    # TRAIN MACHINE
    KNN.fit(Datas['X_Train'], Datas['Y_Train'])

    # FEED TEST DATA TO MACHINE AND RECORD PREDICTIONS
    predictions = KNN.predict(Datas['X_Test'])

    # CALCULATE ERRORS
    Outputs['Confusion Matrix %d' % K] = confusion_matrix(Datas['Y_Test'], predictions)
    Outputs['Classification Report %d' % K] = classification_report(Datas['Y_Test'], predictions)
    error = np.mean(predictions != Datas['Y_Test'])

    # SAVING BEST MACHINE
    if SAVE :
        # PLOT CONFUSION MATRIX
        Print_Log('Saving Confusion Matrix Plot', Phase)
        # NEXT PHASE
        Phase += 1
        Confusion_mat_plot = plot_confusion_matrix(KNN, Datas['X_Test'], Datas['Y_Test']) 
        Confusion_mat_plot.figure_.savefig('%s/Output/Confusion Matrix ( KNN %d ).png' % (Main_Dir, K), dpi=150)

        # SAVING MACHINE
        Print_Log('Saving Machine', Phase)
        # NEXT PHASE
        Phase += 1
        joblib.dump(KNN, '%s/Trained Machine ( KNN %d ).sav' % (Machine_Save_Dir, K))

    return error

def Preprocess_Data(DataFrame) :
    # PHASE VARIABLE
    global Phase

    # START ENCODING
    for temp_column in Columns_to_Encode :
        Print_Log('Processing Column %s' % temp_column, Phase)
        # DEFINE LABEL ENCODER
        Print_Log('\t  Defining Encoder', 0)
        Encoder = preprocessing.LabelEncoder()

        # FIT ENCODER
        Print_Log('\t  Fitting data to Encoder', 0)
        Encoder.fit(DataFrame[temp_column])

        # PRINT ENCODER MAPPING
        Print_Log("\t  Label mapping :", 0)
        for encoded_num, item in enumerate(Encoder.classes_):
            Print_Log('\t  %s ---> %s' % (item , encoded_num), 0)


        # TRANSFORM COLUMN VALUES
        Print_Log('Transforming Column %s' % temp_column, Phase)
        DataFrame[temp_column] = Encoder.transform(DataFrame[temp_column])
    Print_Log('Saving processed data', Phase)
    DataFrame.to_csv('%s/SRC/DataFiles/Processed_DataFrame.csv' % Main_Dir, index=False, header=False)
    return DataFrame

def Split_Data(DataFrame) :
    # PHASE VARIABLE
    global Phase

    # SPLITTING TO TEST AND TRAIN SETS
    x_train, x_test, y_train, y_test = train_test_split(DataFrame.drop('income', axis=1), DataFrame['income'], test_size=0.3, random_state=42)

    # AT THIS POINT WE ADD THEM ALL TO A DICT OBJ
    # SO THAT WE GET A PRETTIER AND CLEANER FUNCTION
    # CALL BUT HANDLE THE SAME STUFF AT Train_Test_Machine()
    Train_Test_Datas = dict()
    Train_Test_Datas['X_Train'] = x_train
    Train_Test_Datas['X_Test'] = x_test
    Train_Test_Datas['Y_Train'] = y_train
    Train_Test_Datas['Y_Test'] = y_test

    return Train_Test_Datas

def Determine_Best_K(Train_Test_Datas, Outputs) :
    # PHASE VARIABLE
    global Phase

    error_rate = []

    # SETTING K RANGE
    K_range = range(min_k_value, max_k_value + 1)
    # THIS IS SIMPLE WE TAKE K VALUE WE TRAIN WE TEST ERROR
    # THE K VALUE WHICH MAKES LEAST ERRROR WILL BE CHOSEN
    for Temp_K in K_range:
        Print_Log('Training K=%d' % Temp_K, Phase)
        error = K_Neareset_Neighbor_Machine(Train_Test_Datas, Outputs, Temp_K, False)
        error_rate.append(error)
        Print_Log('\t  Results : k = %d error = %f' % (Temp_K, error_rate[Temp_K - min_k_value] * 100), 0)

    # PLOTTING K VALUES AND ERRORS
    Print_Log("Plotting K values and errors", Phase)
    # NEXT PHASE
    Phase += 1
    PlotKValues(error_rate, K_range)

    # THE LINE BELLOW :
    # error_rate.index(min(error_rate)) + min_k_value
    # WILL RETURN THE BEST K VALUE
    # SINCE min WILL RETURN LOWEST VALUE IN LIST
    # WE USE index TO GET THE INDEX OF IT BUT THIS
    # INDEX + min_k_value IS THE ACTUAL K
    Best_K_Value = error_rate.index(min(error_rate)) + min_k_value

    return Best_K_Value

def Gaussian_Naive_Bayes_Machine(Datas : dict, Outputs : dict, SAVE : bool):
    # PHASE VARIABLE
    global Phase

    # DEFINE MACHINE
    Print_Log('Creating Gaussian Naive Bayes Machine', Phase)
    GNB = GaussianNB()

    # TRAIN MACHINE
    Print_Log('Training Gaussian Naive Bayes Machine', Phase)
    GNB.fit(Datas['X_Train'], Datas['Y_Train'])

    # FEED TEST DATA TO MACHINE AND RECORD PREDICTIONS
    Print_Log('Predicting with Gaussian Naive Bayes Machine', Phase)
    predictions = GNB.predict(Datas['X_Test'])

    # CALCULATE ERRORS
    Outputs['Gaussian Naive Bayes'] = confusion_matrix(Datas['Y_Test'], predictions)
    Outputs['Classification Report'] = classification_report(Datas['Y_Test'], predictions)
    error = np.mean(predictions != Datas['Y_Test'])

    # SAVING BEST MACHINE
    if SAVE :
        # PLOT CONFUSION MATRIX
        Print_Log('Saving Confusion Matrix Plot', Phase)
        # NEXT PHASE
        Phase += 1
        Confusion_mat_plot = plot_confusion_matrix(GNB, Datas['X_Test'], Datas['Y_Test']) 
        Confusion_mat_plot.figure_.savefig('%s/Output/Confusion Matrix ( GNB ).png' % Main_Dir, dpi=150)

        # SAVING MACHINE
        Print_Log('Saving Machine', Phase)
        # NEXT PHASE
        Phase += 1
        joblib.dump(GNB, '%s/Trained Machine ( GNB ).sav' % Machine_Save_Dir)

    return error

def Logistic_Regression_Classifier_Machine(Datas : dict, Outputs : dict, C_Value : int, SAVE : bool) :
    # PHASE VARIABLE
    global Phase

    # CREATE MACHINE
    Print_Log('Creating Logistic Classifier Machine', Phase)
    Logistic_Regression_Classifier = LogisticRegression(solver = 'liblinear', C=C_Value)

    # TRAIN MACHINE
    Print_Log('Training Logistic Classifier Machine', Phase)
    Logistic_Regression_Classifier.fit(Datas['X_Train'], Datas['Y_Train'])

    # PREDICT
    Print_Log('Predicting with Logistic  Classifier Machine', Phase)
    predictions = Logistic_Regression_Classifier.predict(Datas['X_Test'])

    # CALCULATE ERRORS
    Outputs['Confusion Matrix ( Logistic Classifier )'] = confusion_matrix(Datas['Y_Test'], predictions)
    Outputs['Classification Report ( Logistic Classifier )'] = classification_report(Datas['Y_Test'], predictions)
    error = np.mean(predictions != Datas['Y_Test'])

    # SAVING BEST MACHINE
    if SAVE :
        # PLOT CONFUSION MATRIX
        Print_Log('Saving Confusion Matrix Plot', Phase)
        # NEXT PHASE
        Phase += 1
        Confusion_mat_plot = plot_confusion_matrix(Logistic_Regression_Classifier, Datas['X_Test'], Datas['Y_Test']) 
        Confusion_mat_plot.figure_.savefig('%s/Output/Confusion Matrix ( LC ).png' % Main_Dir, dpi=150)

        Print_Log('Saving Machine', Phase)
        # NEXT PHASE
        Phase += 1
        joblib.dump(Logistic_Regression_Classifier, '%s/Trained Machine ( LC ).sav' % Machine_Save_Dir)
    return error

def Support_Vector_Machine(Datas : dict, Outputs : dict, SAVE : bool, Random_State = 0) :
    # PHASE VARIABLE
    global Phase

    # CREATE MACHINE
    Print_Log('Creating Support Vector Machine', Phase)
    Support_Vector_Machine = OneVsOneClassifier(LinearSVC(random_state = Random_State))

    # TRAIN MACHINE
    Print_Log('Training Support Vector Machine', Phase)
    Support_Vector_Machine.fit(Datas['X_Train'], Datas['Y_Train'])

    # PREDICT
    Print_Log('Predicting with Support Vector Machine', Phase)
    predictions = Support_Vector_Machine.predict(Datas['X_Test'])

    # CALCULATE ERRORS
    Outputs['Confusion Matrix ( Support Vector Machine )'] = confusion_matrix(Datas['Y_Test'], predictions)
    Outputs['Classification Report ( Support Vector Machine )'] = classification_report(Datas['Y_Test'], predictions)
    error = np.mean(predictions != Datas['Y_Test'])

    # SAVING BEST MACHINE
    if SAVE :
        # PLOT CONFUSION MATRIX
        Print_Log('Saving Confusion Matrix Plot', Phase)
        # NEXT PHASE
        Phase += 1
        Confusion_mat_plot = plot_confusion_matrix(Support_Vector_Machine, Datas['X_Test'], Datas['Y_Test']) 
        Confusion_mat_plot.figure_.savefig('%s/Output/Confusion Matrix ( SVM ).png' % Main_Dir, dpi=150)

        # SAVING MACHINE
        Print_Log('Saving Machine', Phase)
        # NEXT PHASE
        Phase += 1
        joblib.dump(Support_Vector_Machine,'%s/Trained Machine ( SVM ).sav' % Machine_Save_Dir)

    return error


def Main_Process() :
    # PHASE VARIABLE
    global Phase

    # ALGORYTHM VARIABLE
    global Algorythm
    # DEFINIG VARIABLES LOCAL TO THIS
    # FUNCTION :
    # A DICT OBJECT TO SAVE RESULTS OF EACH TRAIN
    # IN AN ALGORYTM TO LOWER THE ERROR
    Outputs = {}
    Train_Test_Datas = {}



    # READ DATASET
    Print_Log("Reading dataset", Phase)
    # NEXT PHASE
    try :
        df = pd.read_csv(Dataset_Dir, names=Columns)
    except FileNotFoundError:
        Print_Log('Error : DataSet Not Found Check Directory')
    Outputs['Columns'] = Columns

    # PREPROCESSING DATASET
    Print_Log("Preprocessing dataset", Phase)
    # NEXT PHASE
    Phase += 1
    df = Preprocess_Data(df)



    Print_Log("Splitting Dataset", Phase)
    # NEXT PHASE
    Phase += 1
    Train_Test_Datas = Split_Data(df)

    # K-Nearest-Neighbor
    if Algorythm == 'KNN' :
        Print_Log("Finding the best value for K", Phase)
        Best_K = Determine_Best_K(Train_Test_Datas, Outputs)
        # NEXT PHASE
        Phase += 1

        # SAVING ALL THE RESULTS AND MACHINE
        Print_Log("Saving results", Phase)
        # NEXT PHASE
        Phase += 1



        # Retraining with the best K value
        Print_Log("Retraining with the best K value", Phase)
        # NEXT PHASE
        Phase += 1
        K_Neareset_Neighbor_Machine(Train_Test_Datas, Outputs, Best_K, True)

    # Gaussian-Naive-Bayes
    elif Algorythm == 'GNB' :
        Gaussian_Naive_Bayes_Machine(Train_Test_Datas, Outputs, True)
        # NEXT PHASE
        Phase += 1

    # Logistic-Classifier
    elif Algorythm == 'LC' :
        Logistic_Regression_Classifier_Machine(Train_Test_Datas, Outputs, 10, True)
        # NEXT PHASE
        Phase += 1

    # Support-Vector-Machine
    elif Algorythm == 'SVM' :
        Support_Vector_Machine(Train_Test_Datas, Outputs, True)
        # NEXT PHASE
        Phase += 1

    # RUN ALL ALGORYTHMS
    elif Algorythm == 'ALL' :
        # KNN
        print('Running KNN')
        Print_Log("Finding the best value for K", Phase)
        Best_K = Determine_Best_K(Train_Test_Datas, Outputs)
        
        # RETRATINING WITH THE BEST K VALUE
        Print_Log("Retraining with the best K value", Phase)
        K_Neareset_Neighbor_Machine(Train_Test_Datas, Outputs, Best_K, True)

        # GNB
        print('Running GNB')
        Gaussian_Naive_Bayes_Machine(Train_Test_Datas, Outputs, True)

        # LC
        print('Running LC')
        Logistic_Regression_Classifier_Machine(Train_Test_Datas, Outputs, 10, True)
        
        # SVM
        print('Running SVM')
        Support_Vector_Machine(Train_Test_Datas, Outputs, True)


    # SAVE OUTPUT
    Print_Log('Saving Outputs', Phase)
    SaveOutput(Outputs)


    return 'Done'

# RUN MAIN
if __name__ == '__main__':
    Main_Process()
