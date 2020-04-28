#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# In order to obtain reliable prediction results, we need to clean the data. We have to check if there are some missing or incorrect values, deal with categorical variables etc. In this part, we will also show the features' distributions.

# In[106]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns; sns.set(color_codes=True)


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# The main table is divided into two files - "train" and "test". We are also given additional loan information in other available files. Let's see the mail train file:

# In[3]:


train=pd.read_csv('application_train.csv')
print("The main train file is composed of", train.shape[0], "observations and", train.shape[1], "features")
print("We show only first five rows of a dataset because of its large size:")
train.head()


# The most crucial variable is TARGET. While its value equals 1, the observation (a human) has difficulties with repaying the loan. This is the variable we want to predict - in our model, we will try to say if the observation, given some features, will be able to repay the loan or not. We will test the model on "train" file which description is presented below.

# In[4]:


test = pd.read_csv('application_test.csv')
print("The main test file is composed of", test.shape[0], "observations and", test.shape[1], "features")
print("We show only first five rows of a dataset because of its large size:")
test.head()


# Test file is more than 6 times smaller thsn train file. It also has one column less - it does not have TARGET variable which is logic because we will predict it.

# We will load all others files from which we will extract some intersrting data later.

# In[5]:


bureau = pd.read_csv("bureau.csv")
bureau_balance = pd.read_csv("bureau_balance.csv")
POS_CASH_balance = pd.read_csv("POS_CASH_balance.csv")
credit_card_balance = pd.read_csv("credit_card_balance.csv")
previous_application = pd.read_csv("previous_application.csv")
installments_payments = pd.read_csv("installments_payments.csv")


# ## Missing values

# One of the most important things to do in data pre-processiong is check the amount of missing and incorrect values. We will present the number and percent of missing values in each dataset. In order to do it quickly, we will define missing values function:

# In[6]:


def missing_values(data):
    nulls=data.isnull().sum()
    nulls_sorted=data.isnull().sum().sort_values(ascending = False).round(2)
    percent=((nulls/data.shape[0])*100).sort_values(ascending = False).round(2)
    return pd.concat([nulls_sorted, percent], axis=1, keys=['Number', 'Percent'])


# We have to decide what we want to do with the missing values. There are two most common possible ways - remove them or replace with reasonable values. We arbitrarily dicided to drop the features which missing values exceed 45%.

# In[7]:


missing_values(train).head(10)


# In[8]:


#EXT1 = train.EXT_SOURCE_1


# In[7]:


def remove_missing_values(dataframe):
    to_drop=[] #We create a list storing columns to drop
    df=missing_values(dataframe)
    df["new"]=df["Percent"]>65 #We create a new column with True if missing values exceed 65% of all values

    for index, row in df.iterrows():
        if row["new"]==True:
            to_drop.append(index)

    for i in to_drop: #We drop corresponding columns
        dataframe = dataframe.drop(i, axis=1)

    return dataframe


# In[8]:


train = remove_missing_values(train)


# In[11]:


#train['EXT_SOURCE_1'] = EXT1


# In[12]:


train.shape


# In[9]:


missing_values(test).head(10)


# In[10]:


test = remove_missing_values(test)


# In[15]:


test.shape


# In[11]:


missing_values(bureau).head(10)


# In[17]:


#bureau = remove_missing_values(bureau)


# In[18]:


bureau.shape


# In[18]:


missing_values(bureau_balance).head(10)


# In[19]:


missing_values(POS_CASH_balance).head(10)


# In[20]:


missing_values(credit_card_balance).head(10)


# In[21]:


missing_values(previous_application).head(10)


# In[12]:


previous_application = remove_missing_values(previous_application)
    
previous_application.shape


# In[44]:


missing_values(installments_payments).head(10)


# ## Anomalies

# In the following part, we will examine the outliers and we will look for incorrect (impossible) variables. We will look for them in variables that represent days in "train" and "test" data sets because only in this case we can easily state if the value is correct or not. For this purpose, we will use histograms.

# In[45]:


plt.hist(train['DAYS_BIRTH'])


# In[14]:


plt.hist(train['DAYS_EMPLOYED'])


# 350000 days equals almost 1000 years. The variable describes for how many days an observation was employed so we can be sure that values near 1000 years are incorrect. As we can see in the plot, they do not exceed 45% of all the values. We will set them as NaN and then treat in the same way as all the other missing values in the next part of exploratory data analysis.

# In[14]:


for index, row in train.iterrows():
    if row['DAYS_EMPLOYED']>30000:
        train=train.replace({row['DAYS_EMPLOYED'] : np.nan})


# In[21]:


plt.hist(train['DAYS_REGISTRATION'])


# In[22]:


plt.hist(train['DAYS_ID_PUBLISH'])


# In[23]:


plt.hist(test['DAYS_BIRTH'])


# In[24]:


plt.hist(test['DAYS_EMPLOYED'])


# This is similar situation to "train['DAYS_EMPLOYED']" feature. We use the same code chunk to replace incorrect values with missing ones.

# In[16]:


for index, row in test.iterrows():
    if row['DAYS_EMPLOYED']>30000:
        test=test.replace({row['DAYS_EMPLOYED'] : np.nan})


# In[27]:


plt.hist(test['DAYS_REGISTRATION'])


# In[28]:


plt.hist(test['DAYS_ID_PUBLISH'])


# It turned out that incorrect values were present in two cases.

# ## Categorical variables

# Our prediction model will work properly only if all the variables in the database are numeric. Let us see the type of all variables in the database (float64 - numeric, object - categorical).

# In[21]:


train.dtypes.value_counts()


# In[22]:


test.dtypes.value_counts()


# We will use the code provided during lesson. Its idea is to use label encoding when a variable has only two possibilities of categorical values and one-hot encoding while there are more than 2 possible values.

# In[17]:


#Label encoding

le=LabelEncoder()

for col in train.columns:
    if (train[col].dtype == object) and (len(train[col].unique())<=2):
        print(col) #We print names of columns for which we did label encoding
        le.fit(train[col])
        train[col]=le.transform(train[col])
        test[col]=le.transform(test[col])


# In[18]:


#One-hot encoding

train=pd.get_dummies(train)
test=pd.get_dummies(test)


# Because one-hot encoding created new columns, we need to reshape our train and test database to be the same.

# In[19]:


target_train = train['TARGET'] #We remove this variable because it is not present in test database

#Main part for aligning the database
train, test = train.align(test, join = 'inner', axis = 1)


# ## Dealing with missing values

# As all the variables in th dataset are numerical and databse are aligned, the next step is to replace the missing values. We decided to replace them with the median of non-missing values.

# In[20]:


imputer=Imputer(strategy="median")

imputer.fit(train)
train.loc[:]=imputer.transform(train)
test.loc[:]=imputer.transform(test)


# In[21]:


train['TARGET'] = target_train #We add again the variable removed for the purpose of aligning and dealing with missing values


# ## Final database

# We check the size of final train and test databse.

# In[37]:


train.shape


# In[38]:


train.head()


# In[39]:


test.shape


# In[40]:


test.head()


# In[22]:


train.to_csv("train.csv",index=False)


# In[23]:


test.to_csv("test.csv",index=False)


# ## Final database quick analysis

# In this part, we would like to present some information in a graphical way.

# First of all, we want to see how many loans were repaid:

# In[57]:


train['TARGET'].value_counts()


# In[60]:


labels = '0','1'
sizes = [train['TARGET'].value_counts()[0], train['TARGET'].value_counts()[1]]
colors = ['yellow','lightblue']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# The vast majority of loans have been repaid. <br><br> We will examine the structure of households taking loans:

# In[92]:


train['CNT_FAM_MEMBERS'].value_counts()


# In[105]:


labels = '2','1', '3', '4', '5', '6', '7', '8', '9', '10', '14', '16', '12', '20', '11', '13', '15', 
sizes = [train['CNT_FAM_MEMBERS'].value_counts()[int(x)] for x in labels]
colors=['aqua','gray','pink', 'fuchsia', 'blue', 'green','lime','maroon','navy','olive', 'purple','red', 'silver','lightblue','lightgreen','lightpink','yellow']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)

plt.axis('equal')
plt.show()


# From data above we can conclude that the majority of hoseholds taking loans is composed of two people so most probably partners without children. In the next place are single people and then, the most probaby, families with children.

# Let us now examine the density estimate of age of the sample.

# In[109]:


x = sns.kdeplot(train['DAYS_BIRTH'])


# The shape of the graph above looks like normal distribution which is logic. There are slightly more younger people in the survey what could be associated with the household size - as already written, the majority of sample is composed by single people or partners, so probably young people.

# The analysis above present most general features, easy to explain without any further data. We will have a look on some 'deeper' variable like owning a car:

# In[110]:


train['FLAG_OWN_CAR'].value_counts()


# In[102]:


labels = '0','1' #0 stands for 'no', 1 - 'yes'
sizes = [train['FLAG_OWN_CAR'].value_counts()[0], train['FLAG_OWN_CAR'].value_counts()[1]]
colors = ['yellow','lightblue']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# This is already quite hard to explain. We will try to examine also the continuous varibale - the total income by first presenting its density estimate:

# In[122]:


sns.kdeplot(train['AMT_INCOME_TOTAL'])


# We can also examine the influence of income on the ability of repaying the loan:

# In[121]:


sns.kdeplot(train.loc[train['TARGET'] == 0, 'AMT_INCOME_TOTAL'], label = 'target == 0')

sns.kdeplot(train.loc[train['TARGET'] == 1, 'AMT_INCOME_TOTAL'], label = 'target == 1')

plt.xlabel('Total income'); plt.ylabel('Density'); plt.title('Distribution of total income');


# Because of the structure of income distribution, there are no clear findings from the graph.<br><br> As we had s quick look on the final dataset, we can go to the further analysis to draw more interesting conclusions.

# ## Feature engineering

# Feature engineering refers to a geneal process and can involve both feature construction: adding new features from the existing data, and feature selection: choosing only the most important features or other methods of dimensionality reduction. There are many techniques we can use to both create features and select features.

# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# Here we will add some features inorder to improve our accuracy at the end

# First we will make some polynomial features

# In[4]:


# Make a new dataframe for polynomial features
p_features = train[['EXT_SOURCE_1','EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
p_features_test = test[['EXT_SOURCE_1','EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
p_target = p_features['TARGET']
p_features = p_features.drop(columns = ['TARGET'])


# Since we cleaned and filled the missing values before, we don't have any missing values for these two dataframes:

# In[5]:


missing_values(p_features)


# In[6]:


missing_values(p_features_test)


# In[7]:


# Creating the polynomial object with specified degree
p_trans = PolynomialFeatures(degree = 3)


# In[8]:


# Train the polynomial features
p_trans.fit(p_features)
# Transform the features
p_features = p_trans.transform(p_features)
p_features_test = p_trans.transform(p_features_test)


# In[9]:


p_features.shape


# In[10]:


p_trans.get_feature_names(input_features = ['EXT_SOURCE_1','EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])[:15]


# In[11]:


# Creating a dataframe of the features 
p_features = pd.DataFrame(p_features, 
                             columns = p_trans.get_feature_names(['EXT_SOURCE_1','EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))


# In[12]:


# Add-in the target
p_features['TARGET'] = p_target


# In[13]:


# Find the correlations with the target
poly_corrs = p_features.corr()['TARGET'].sort_values()
# Display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))


# In[14]:


# Put test features into dataframe
p_features_test = pd.DataFrame(p_features_test, 
                                  columns = p_trans.get_feature_names(['EXT_SOURCE_1','EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 'DAYS_BIRTH']))


# In[15]:


# Merge polynomial features into training dataframe
p_features['SK_ID_CURR'] = train['SK_ID_CURR']
train_poly = train.merge(p_features, on = 'SK_ID_CURR', how = 'left')


# In[16]:


# Merge polnomial features into testing dataframe
p_features_test['SK_ID_CURR'] = test['SK_ID_CURR']
test_poly = test.merge(p_features_test, on = 'SK_ID_CURR', how = 'left')


# In[17]:


# Align the dataframes
train_poly, test_poly = train_poly.align(test_poly, join = 'inner', axis = 1)


# In[18]:


print(train_poly.shape)
print(test_poly.shape)


# In[19]:


train_poly.head()


# In[20]:


test_poly.head()


# Now we make some other features

# In[21]:


train_domain = train.copy()
test_domain = test.copy()


# In[22]:


train_domain['CREDIT_INCOME_PERCENT'] = train_domain['AMT_CREDIT'] / train_domain['AMT_INCOME_TOTAL']
train_domain['ANNUITY_INCOME_PERCENT'] = train_domain['AMT_ANNUITY'] / train_domain['AMT_INCOME_TOTAL']
train_domain['CREDIT_TERM'] = train_domain['AMT_ANNUITY'] / train_domain['AMT_CREDIT']
train_domain['DAYS_EMPLOYED_PERCENT'] = train_domain['DAYS_EMPLOYED'] / train_domain['DAYS_BIRTH']


# In[23]:


test_domain['CREDIT_INCOME_PERCENT'] = test_domain['AMT_CREDIT'] / test_domain['AMT_INCOME_TOTAL']
test_domain['ANNUITY_INCOME_PERCENT'] = test_domain['AMT_ANNUITY'] / test_domain['AMT_INCOME_TOTAL']
test_domain['CREDIT_TERM'] = test_domain['AMT_ANNUITY'] / test_domain['AMT_CREDIT']
test_domain['DAYS_EMPLOYED_PERCENT'] = test_domain['DAYS_EMPLOYED'] / test_domain['DAYS_BIRTH']


# In[24]:


missing_values(train_domain)


# In[25]:


missing_values(test_domain)


# In[26]:


train_domain.shape


# In[27]:


train = pd.merge(train_poly,train_domain, how = 'inner')


# In[28]:


test = pd.merge(test_poly,test_domain, how = 'inner')


# In[29]:


train.head()


# In[30]:


test.head()


# In[31]:


domain_corrs = train_domain.corr()['TARGET'].sort_values()


# In[32]:


print(domain_corrs.head(10))
print(domain_corrs.tail(10))


# In[33]:


bureau = pd.read_csv('bureau.csv')
bureau.head()


# In[34]:


# Groupby the client id (SK_ID_CURR), count the number of previous loans, and rename the column
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns = {'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()


# In[35]:


missing_values(previous_loan_counts)


# In[36]:


train = train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')


# In[37]:


# Fill the missing values with 0 
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)
train.head()


# In[38]:


#calculating the corr between desire feature and the df's target
def corr_calc(feature,df):
    corr = df['TARGET'].corr(df[feature])
    return corr


# In[39]:


corr_calc('previous_loan_counts',train)


# ### Feature selection

# In[40]:


# Calculate all correlations in dataframe
corrs = train.corr()


# In[41]:


corrs = corrs.sort_values('TARGET', ascending = False)


# In[45]:


# Ten most positive correlations
pd.DataFrame(corrs['TARGET'].head(10))


# In[46]:


# Ten most negative correlations
pd.DataFrame(corrs['TARGET'].dropna().tail(10))


# In[47]:


# Set a threshold
threshold = 0.8
# Empty dictionary to hold correlated variables
above_threshold_vars = {}
# For each column, record the variables that are above the threshold
for col in corrs:
    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])


# In[48]:


# Track columns to remove and columns already examined
cols_to_remove = []
cols_seen = []
cols_to_remove_pair = []
# Iterate through columns and correlated columns
for key, value in above_threshold_vars.items():
    # Keep track of columns already examined
    cols_seen.append(key)
    for x in value:
        if x == key:
            next
        else:
            # Only want to remove one in a pair
            if x not in cols_seen:
                cols_to_remove.append(x)
                cols_to_remove_pair.append(key)
            
cols_to_remove = list(set(cols_to_remove))
print('Number of columns to remove: ', len(cols_to_remove))


# In[49]:


train.shape


# In[50]:


test.shape


# In[51]:


train_corrs_removed = train.drop(columns = cols_to_remove)
test_corrs_removed = test.drop(columns = cols_to_remove)


# In[52]:


train_corrs_removed.shape


# In[53]:


test_corrs_removed.shape


# In[54]:


train_corrs_removed.to_csv("newtrain.csv",index=False)


# In[55]:


test_corrs_removed.to_csv("newtest.csv",index=False)


# ## Model

# In[ ]:




