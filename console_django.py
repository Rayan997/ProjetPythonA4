# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:08:35 2021

@author: Miche
"""
#%% transformer dataset
import pandas as pd
df =pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df["IMC"] = df["Weight"]/(df["Height"]*df["Height"])

df_quantitatif = df.copy() #Creation d'une copie de la dataframe

df_quantitatif = df_quantitatif.drop('Gender', 1) #On supprime les colonnes que l'on veut modifier
df_quantitatif = df_quantitatif.drop('family_history_with_overweight', 1)
df_quantitatif = df_quantitatif.drop('FAVC', 1)
df_quantitatif = df_quantitatif.drop('FCVC', 1)
df_quantitatif = df_quantitatif.drop('CAEC', 1)
df_quantitatif = df_quantitatif.drop('SMOKE', 1)
df_quantitatif = df_quantitatif.drop('SCC', 1)
df_quantitatif = df_quantitatif.drop('CALC', 1)

df_quantitatif['Gender'] = [0 if x == 'Female' else 1 for x in df['Gender']]
df_quantitatif['family_history_with_overweight'] = [0 if x == 'no' else 1 for x in df['family_history_with_overweight']]
df_quantitatif['FAVC'] = [0 if x == 'no' else 1 for x in df['FAVC']]

# Je fais ce changement car je pense que ici 1 correspond à : la personne consomme des légumes à chaque repas
# et 3 : la personne ne consomme jamais de légume, or je veux l'inverse
# Assertion faite en comparant la moyenne de FCVC en fonction du NObeyesdad

df_quantitatif['FCVC'] = [1 if x == 3 else 2 if x == 2 else 3 for x in df['FCVC']]
df_quantitatif['CAEC'] = [0 if x == 'no' else 1 if x == "Sometimes" else 2 if x == "Frequently" else 3 for x in df['CAEC']]
df_quantitatif['SMOKE'] = [0 if x == 'no' else 1 for x in df['SMOKE']]
df_quantitatif['SCC'] = [0 if x == 'no' else 1 for x in df['SCC']]
df_quantitatif['CALC'] = [0 if x == 'no' else 1 if x == "Sometimes" else 2 if x == "Frequently" else 3 for x in df['CALC']]

df_quantitatif_modelisation = df_quantitatif.copy()

df_quantitatif_modelisation = df_quantitatif_modelisation.drop("SMOKE", 1)
df_quantitatif_modelisation = df_quantitatif_modelisation.drop("Gender", 1)
df_quantitatif_modelisation = df_quantitatif_modelisation.drop("Height" , 1)
df_quantitatif_modelisation = df_quantitatif_modelisation.drop("Weight", 1)

from sklearn.preprocessing import StandardScaler

X = df_quantitatif_modelisation.copy()
X = X.drop('MTRANS', 1)
X = X.drop('IMC', 1)
X = X.drop('NObeyesdad', 1)

Y=df['IMC']

X2 = X.copy()
#scaler = StandardScaler()
#scaler.fit(X2)
#X2 = scaler.transform(X2)
Y2 = df['NObeyesdad']


#%% séparation en training et test
from sklearn.model_selection import train_test_split
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2) #datasets pour prédire NObesity (classification)

#%% Random Forest
from sklearn.ensemble import RandomForestClassifier


modelRF2 = RandomForestClassifier(bootstrap = False , max_depth=50, max_features="sqrt", min_samples_leaf=1,min_samples_split=5,n_estimators=1000)# optimisation de l'algo
modelRF2.fit(X2_train, Y2_train)# apprentissage

#variables de base du dataset: Gender,Age,Height,Weight,family_history_with_overweight,
#                              FAVC,FCVC,NCP,CAEC,SMOKE,CH2O,SCC,FAF,TUE,CALC,MTRANS,NObeyesdad
#variables à enlever: gender, height, weight, smoke, mtrans, imc, N0beyesdad
#
#Age,family_history_with_overweight,FAVC,FCVC,NCP,CAEC,CH2O,SCC,FAF,TUE,CALC
#Age,NCP,family_history_with_overweight,FAVC,FCVC,CAEC,CH2O,SCC,FAF,TUE,CALC
#Age,NCP,CH2O,family_history_with_overweight,FAVC,FCVC,CAEC,SCC,FAF,TUE,CALC
#Age,NCP,CH2O,FAF,family_history_with_overweight,FAVC,FCVC,CAEC,SCC,TUE,CALC
#Age,NCP,CH2O,FAF,TUE,family_history_with_overweight,FAVC,FCVC,CAEC,SCC,CALC



#variables restantes: Age,NCP,CH2O, FAF,TUE, family_history_with_overweight,FAVC, FCVC, CAEC, SCC,CALC
#Conversion: Gender                          female =0, male =1
#            family_history_with_overweight  no=0, yes=1
#            FAVC                            no=0, yes=1
#            CAEC                            no=0, sometimes = 1, frequently = 2 
#            SCC                             no=0, yes=1
#            CALC                            no=0, sometimes = 1, frequently = 2
    
    
    
#%% PRE-TEST
#ligne 626: Male,16.496978,1.691206,50,no,yes,2,1.630846,Sometimes,no,2.975528,no,0.548991,0.369134,Sometimes,Public_Transportation,Insufficient_Weight
#test_Insufficient_Weight =[16.496978,0,1,2,1.630846,1,2.975528,0,0.548991,0.369134,1]
test_Insufficient_Weight =[16.496978,1.630846,2.975528,0,0.548991,0.369134,1,2,1,0,1]
#ligne 2: Female,21,1.62,64,yes,no,2,3,Sometimes,no,2,no,0,1,no,Public_Transportation,Normal_Weight
#test_Normal_Weight=[21,1,0,2,3,1,2,0,0,1,0]
test_Normal_Weight=[21,3,2,0,1,1,0,2,1,0,0]
#ligne 5: Male,27,1.8,87,no,no,3,3,Sometimes,no,2,no,2,0,Frequently,Walking,Overweight_Level_I
#test_Overweight_Level_I=[27,0,0,3,3,1,2,0,2,0,2]
test_Overweight_Level_I=[27,3,2,2,0,0,0,3,1,0,2]
#ligne 6: Male,22,1.78,89.8,no,no,2,1,Sometimes,no,2,no,0,0,Sometimes,Public_Transportation,Overweight_Level_II
#test_Overweight_Level_II=[22,0,0,2,1,1,2,0,0,0,1]
test_Overweight_Level_II=[22,1,2,0,0,0,0,2,1,0,1]
#ligne 1813: Female,21.334585,1.729045,131.529267,yes,yes,3,3,Sometimes,no,1.302193,no,1.742453,0.94232,Sometimes,Public_Transportation,Obesity_Type_III 
#test_Overweight_Level_III=[21.334585,1,1,3,3,1,1.302193,0,1.742453,0.94232,1]
test_Overweight_Level_III=[21.334585,3,1.302193,1.742453,0.94232,1,1,3,1,0,1]



#%% TEST

test_Insufficient_Weight =[16.496978,1.630846,2.975528,0,0.548991,0.369134,1,2,1,0,1]
test_Normal_Weight=[21,3,2,0,1,1,0,2,1,0,0]
test_Overweight_Level_I=[27,3,2,2,0,0,0,3,1,0,2]
test_Overweight_Level_II=[22,1,2,0,0,0,0,2,1,0,1]
test_Overweight_Level_III=[21.334585,3,1.302193,1.742453,0.94232,1,1,3,1,0,1]


def p():  
    print("Pour Insufficient_Weight:",modelRF2.predict([test_Insufficient_Weight]))
    print("Pour Normal_Weight:",modelRF2.predict([test_Normal_Weight]))
    print("Pour test_Overweight_Level_I:",modelRF2.predict([test_Overweight_Level_I]))
    print("Pour test_Overweight_Level_II:",modelRF2.predict([test_Overweight_Level_II]))
    print("Pour test_Overweight_Level_III:",modelRF2.predict([test_Overweight_Level_III]))


def prediction():
    print('Age: ') 
    age = float(input())
    print('NCP/ Nombre de repas par jour: ')
    ncp = float(input())
    print("CH2O/ Consommation d'eau par jour : ") 
    ch2o = float(input())
    print("FAF/ Fréquence d'activité physique par semaine: ") 
    faf =float(input())
    print('TUE/ Temps par jour devant des écrans: ') 
    tue = float(input())
    print('family_history_with_overweight: ')
    f = float(input())
    print('FAVC/ Consommation régulière de nourriture très calorique: ') 
    favc = float(input())
    print('FCVC/  Consommation de légumes dans les repas: ') 
    fcvc = float(input())
    print('CAEC/ Consommation de nourriture entre les repas: ') 
    caec = float(input())
    print('SCC/ Compte le nombre de calories ingérées: ')
    scc = float(input())
    print("CALC/ Consommation d'alcool: ")
    calc = float(input())
    test=[age, ncp, ch2o, faf, tue, f, favc, fcvc, caec, scc, calc]
    print("Votre catégorie: ",modelRF2.predict([test]))


