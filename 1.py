#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:3rem;color:orange;">Obesity Levels</h1>

# ## 1- Importation et installation des différents modules

# ### 1-1 installation des différents modules qui ne sont pas directement dans anaconda

# In[39]:


#!pip install plotly
#!pip install prince


# ### 1-2 Importation

# In[137]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import prince


# ### 1-3 Données sur l'obésité
# 

# Elles sont téléchargées depuis <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/00544/ObesityDataSet_raw_and_data_sinthetic%20(2).zip">ce lien. </a> Elle donne une estimation du taux d'obésité des Mexicains,
# Peruvien et Columbien, basé en grande partie sur leur habitudes 
# alimentaire et leur condition physique. Ce dataset contient 2111 observations dont 77% a été générée
# artificiellement.

# ## 2- Creation de la dataframe

# ### 2-1 Chargement des données

# In[138]:


df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv',sep=',',decimal = '.') #creation de la DataFrame
# peu de temps d'attente car l'on charge environs 2000 lignes de données


# ### 2-2 Nettoyage

# Ce bloc sert à nettoyer la dataframe et à modifier les types de certaines valeurs pour faciliter l'utilisation

# In[139]:


#Nettoyage de la data


# ### 2-3 Creation de nouvelles colonnes

# In[140]:


df["IMC"] = df["Weight"]/(df["Height"]*df["Height"])


# ## Data Visualization

# In[141]:


df.head(50) #Affiche les 50 premières lignes


# In[142]:


df.describe() #Affiche des indications sur les variables quantitatives


# In[143]:


df.info()


# In[144]:


sns.scatterplot(x="FCVC", y="IMC" , data=df)


# In[145]:


sns.scatterplot(x="Height", y="IMC" , hue="FAVC", data=df)


# In[146]:


sns.scatterplot(x="Weight", y="IMC", hue="Gender", data=df)


# In[ ]:





# In[ ]:





# ### Factor analysis of mixed data (FAMD)

# Un grand merci à MaxHalford pour avoir créé la librairie prince. Pour accéder à son github, cliquez <a href="https://github.com/MaxHalford/prince">ici. </a>

# In[152]:


df_100 = df.head(100)
df_100 = df_100.drop('IMC', 1)
famd = prince.FAMD(n_components=2, n_iter=5, copy=True, check_input=True, engine='sklearn', random_state=50)
famd = famd.fit(df_100.drop('NObeyesdad', axis='columns'))
famd.row_coordinates(df_100)


# Ce graphique explique à peine 25% de l'informations, de plus les clusters sont embriqués les uns dans les autres, il n'est donc pas très pertinent.
# 

# In[153]:


ax = famd.plot_row_coordinates(df_100, ax=None, figsize=(8, 8), x_component=0, y_component=1, labels=df_100.index, color_labels=['NObeyesdad {}'.format(t) for t in df_100['NObeyesdad']], ellipse_outline=False, ellipse_fill=True, show_points=True)
ax


# ### Principal Component Analysis (PCA)

# La PCA ne marche que sur des données quantitatives, je vais donc transformer les données en données
# quantitatives, en espérant avoir une meilleur precision que dans le FAMD

# In[168]:


df_quantitatif = df

#Ce n'est pas cohérent de transformer le mode transport en donnée quantitative, je supprime donc cette colonne.
df_quantitatif = df_quantitatif.drop('MTRANS', 1)
df_quantitatif = df_quantitatif.drop('IMC', 1)
df_quantitatif = df_quantitatif.drop('NObeyesdad', 1)

# J'ai effectué les transformations suivante :
# no ==> 0
# yes ==> 1
# Sometimes ==> 1
# Frequently ==> 2
# Always ==> 3
# Femme ==> 0
# Homme ==> 1

#Remplace "Female" par 0 dans df_quantitatif["Gender"]
df_quantitatif.loc[df_quantitatif.Gender == "Female",'Gender'] = 0
#Remplace "no" par 0 dans df_quantitatif["Gender"]
df_quantitatif.loc[df_quantitatif.Gender == "Male",'Gender'] = 1

#Remplace "yes" par 1 dans df_quantitatif["family_history_with_overweight"]
df_quantitatif.loc[df_quantitatif.family_history_with_overweight == "yes",'family_history_with_overweight'] = 1
#Remplace "no" par 0 dans df_quantitatif["family_history_with_overweight"]
df_quantitatif.loc[df_quantitatif.family_history_with_overweight == "no",'family_history_with_overweight'] = 0

#Remplace "yes" par 1 dans df_quantitatif["FAVC"]
df_quantitatif.loc[df_quantitatif.FAVC == "yes",'FAVC'] = 1
#Remplace "no" par 0 dans df_quantitatif["FAVC"]
df_quantitatif.loc[df_quantitatif.FAVC == "no",'FAVC'] = 0

# Je fais ce changement car je pense que ici 1 correspond à : la personne consomme des légumes à chaque repas
# et 3 : la personne ne consomme jamais de légume, or je veux l'inverse
# Assertion faite en comparant la moyenne de FCVC en fonction du NObeyesdad
#Remplace 1 par 3 dans df_quantitatif["FCVC"]
df_quantitatif.loc[df_quantitatif.FCVC == 1,'FCVC'] = 3
#Remplace 1 par 3 dans df_quantitatif["FCVC"]
df_quantitatif.loc[df_quantitatif.FCVC == 3,'FCVC'] = 1

#Remplace no par 0 dans df_quantitatif["CAEC"]
df_quantitatif.loc[df_quantitatif.CAEC == "no",'CAEC'] = 0
#Remplace Sometimes par 1 dans df_quantitatif["CAEC"]
df_quantitatif.loc[df_quantitatif.CAEC == "Sometimes",'CAEC'] = 1
#Remplace Frequently par 2 dans df_quantitatif["CAEC"]
df_quantitatif.loc[df_quantitatif.CAEC == "Frequently",'CAEC'] = 2
#Remplace Always par 3 dans df_quantitatif["CAEC"]
df_quantitatif.loc[df_quantitatif.CAEC == "Always",'CAEC'] = 3

#Remplace "yes" par 1 dans df_quantitatif["SMOKE"]
df_quantitatif.loc[df_quantitatif.SMOKE == "yes",'SMOKE'] = 1
#Remplace "no" par 0 dans df_quantitatif["SMOKE"]
df_quantitatif.loc[df_quantitatif.SMOKE == "no",'SMOKE'] = 0

#Remplace "yes" par 1 dans df_quantitatif["SCC"]
df_quantitatif.loc[df_quantitatif.SCC == "yes",'SCC'] = 1
#Remplace "no" par 0 dans df_quantitatif["SCC"]
df_quantitatif.loc[df_quantitatif.SCC == "no",'SCC'] = 0

#Remplace no par 0 dans df_quantitatif["CALC"]
df_quantitatif.loc[df_quantitatif.CALC == "no",'CALC'] = 0
#Remplace Sometimes par 1 dans df_quantitatif["CALC"]
df_quantitatif.loc[df_quantitatif.CALC == "Sometimes",'CALC'] = 1
#Remplace Frequently par 2 dans df_quantitatif["CALC"]
df_quantitatif.loc[df_quantitatif.CALC == "Frequently",'CALC'] = 2
#Remplace Always par 3 dans df_quantitatif["CALC"]
df_quantitatif.loc[df_quantitatif.CALC == "Always",'CALC'] = 3

df_quantitatif_50 = df_quantitatif.head(50)
df_quantitatif.head(10)


# In[169]:


pca = prince.PCA( n_components=2, n_iter=3, rescale_with_mean=True, rescale_with_std=True, copy=True, check_input=True, engine='auto', random_state=42)
pca = pca.fit(df_quantitatif_50)
pca.transform(df_quantitatif_50).head()


# Les données étant entremêler, il est difficile pour l'algorithme de nous fournir des clusters

# In[171]:


df_50 = df.head(50)
ax = pca.plot_row_coordinates(df_quantitatif_50, ax=None, figsize=(9, 9), x_component=0, y_component=1, labels=None, color_labels=df_50["NObeyesdad"], ellipse_outline=False, ellipse_fill=True, show_points=True)
ax


# In[ ]:





# In[98]:


df_Obesity_Type_II = df[df['NObeyesdad'] == "Obesity_Type_II"]
df_Obesity_Type_II.describe()


# In[172]:


df_Obesity_Type_III = df[df['NObeyesdad'] == "Obesity_Type_III"]
df_Obesity_Type_III.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


df_Insufficient_Weight = df[df['NObeyesdad'] == "Insufficient_Weight"]


# In[72]:


df_Insufficient_Weight.describe()

