# VISUALIZATIONS

# IMPORT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import os
import heapq
from operator import itemgetter

DIR = 'DataFiles/Dataset.csv'

# COUNT CATEGORICAL VALUES IN A COLUMN
# RETURN DICTIONARY IN FOLLOWING FORMAT
# {'%CATEGORICAL_VALUE' : NUMBER}
def count_values(Dataframe_column) :
    values = {}
    for val in Dataframe_column :
        try :
            values[val] = values[val] + 1
        except KeyError :
            values[val] = 0


    return values


# TO GET THE TOP 5 CATEGORICAL VALUESS
# IN EACH BAR PLOT WE DO AS BELLOW
def max_n(values : dict, n):
    topitems = heapq.nlargest(n, values.items(), key=itemgetter(1))
    topitemsasdict = dict(topitems)
    return topitemsasdict


# MODIFY CATEGORICAL VALUES TO GET A BETTER RESULT
def modify_race_values(val) :
    li = [' Asian-Pac-Islander', ' Amer-Indian-Eskimo']
    if val in li : return ' Other'
    else: return val


# SINCE THERE ARE MULTIPLE PLOTS WE SIMPLY DEFINE
# A FUNCTION TO DO SO
def count_plot(values : dict, col) :
    font = {'size'   : 5}

    plt.rc('font', **font)

    fig = plt.figure(figsize=(2.5, 2.5))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8], facecolor=(0.2, 0.2, 0.2, 0.8))
    ax.grid(color=(0.1, 0.1, 0.1, 0.01))
    ax.bar(values.keys(), values.values(), color=(1, 0.2, 0.2, 0.5), width=0.3)

    fig.savefig('%s.png' % col, dpi=300, facecolor=(0.2, 0.2, 0.2, 0.8), edgecolor=(0.2, 0.2, 0.2, 0.8))
    return


# TO GET A BETTER UNDRESTANDING OF NUMERICAL VAUES
# WE DEFINE RANGES AND APPLY THEM TO THE AGE COLUMN
def Classifie_Age(Age):
    Result = ''
    if Age<20: Result = 'Teenage'
    elif 20<=Age<=30: Result = 'Young'
    elif Age>30: Result = 'Adult'
    return Result


def Main_Visualization():

    # COLUMN NAMES
    Columns = ['age', 'workclass', 'fnlwgt', 'education',
               'education-num', 'marital-stauts', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain',
               'capital-loss', 'hours-per-week', 'native-country',
               'income']
    
    # READ DATASET
    df = pd.read_csv(DIR, names=Columns)
    
    # CREATE FOLDER FOR OUTPUT PLOTS
    try: os.chdir('Output')
    except:
        os.mkdir('Output')
        os.chdir('Output')
    
    # MODIFY RACE AND AGE COLUMNS
    df['race'] = df['race'].apply(modify_race_values)
    df['Age_Classified'] = df['age'].apply(Classifie_Age)

    # LIST OF COLUMNS TO BE PLOTTED
    categorical_columns = ['education', 'relationship', 'race', 'sex', 'native-country', 'income', 'Age_Classified']
    
    # PLOT
    for col in categorical_columns :
       values_dict = count_values(df[col])
       values_dict = max_n(values_dict, 5)
       count_plot(values_dict, col)

  

    return

if __name__ == "__main__" :
	Main()
