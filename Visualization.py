import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os







Columns = ['age', 'workclass', 'fnlwgt', 'education',
           'education-num', 'marital-stauts', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain',
           'capital-loss', 'hours-per-week', 'native-country',
           'income']

df = pd.read_csv('adult.data', names=Columns)

try: os.chdir('Output')
except:
    os.mkdir('Output')
    os.chdir('Output')
plt.figure(figsize=(10,5))
sns.countplot(df['race'])
plt.savefig('racce.png',dpi=200)


sns.countplot(df['income'])
plt.savefig('income.png',dpi=200)

sns.countplot(df['sex'])
plt.savefig('sex.png',dpi=200)


plt.figure(figsize=(80,5))
sns.countplot(df['native-country'])
plt.savefig('native-country.png',dpi=200)


plt.figure(figsize=(18,5))
sns.countplot(df['education'])
plt.savefig('education.png',dpi=200)



Classified_by_Age = pd.DataFrame()

def Classifie_Age(Age):
    Result = ''
    if Age<20: Result = 'Teenage'
    elif 20<=Age<=30: Result = 'Young'
    elif Age>30: Result = 'Adult'
    return Result

Classified_by_Age['Age_Classified'] = df['age'].apply(Classifie_Age)
sns.countplot(Classified_by_Age['Age_Classified'])


