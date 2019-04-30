
# coding: utf-8

# ## AppOnBoard Data Modeling Technical Assignment 
'''
######## AppOnBoard Data Modeling Technical Assignment ######
 `##### Submitted by : Rachit Mishra #####
##### AVAZU  -- Click Through rate Prediction #####
   ###### Predict whether an ad will be clicked or not #######
'''


# ### Specifying neccessary imports 

'''Specifying the necessary imports'''
import pandas as pd
import numpy as np
import dask.dataframe as dask_data
#Dask data frames used to help in in-memory large computing on a single machine

#visualization specific imports 
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot, download_plotlyjs

import sklearn
import matplotlib.dates as mdates


matplotlib.style.use('ggplot')

#A parse date variable to pass in the read_csv function later to take into account the date format 
parse_date = lambda val : pd.datetime.strptime(val, '%y%m%d%H')


# ### Sampling the training data

''' Specifying a sample of the original training data 
    Training data size == 5.87 gigabytes 
    Number of records === 40428966 (excess of 40 million records)
    
    Using random sampling to select a sample of 1 million records 
'''

import random
n = 40428966  #total number of records in the clickstream data 
sample_size = 1000000
skip_values = sorted(random.sample(range(1,n), n-sample_size)) 

#Tracking the indices of rows to be skipped at random in the next stage i.e the LOADING stage 


# ### Data Loading stage
''' LOADING stage 
    Reading the sampled train data
    Size : 1 million records
'''

train_data = pd.read_csv('./all/train/train.csv', parse_dates = ['hour'], date_parser = parse_date,
                        skiprows = skip_values )
train_data.info()

# ## Memory Optimization

'''
Memory optimization at this point ~~ 183 megabytes 

Optimization technique ::: Alter data types from int64 to int32 to reduce block memory usage

Then RELOADING the data 
'''
data_types = {
    'id': np.str,
    'click': np.bool_,
    'hour': np.str,
    'C1': np.uint16,
    'banner_pos': np.uint16,
    'site_id': np.object,
    'site_domain': np.object,
    'site_category': np.object,
    'app_id': np.object,
    'app_domain': np.object,
    'app_category': np.object,
    'device_id': np.object,
    'device_ip': np.object,
    'device_model': np.object,
    'device_type': np.uint16,
    'device_conn_type': np.uint16,
    'C14': np.uint16,
    'C15': np.uint16,
    'C16': np.uint16,
    'C17': np.uint16,
    'C18': np.uint16,
    'C19': np.uint16,
    'C20': np.uint16,
    'C21': np.uint16
}

train_data = pd.read_csv('./all/train/train.csv', parse_dates = ['hour'],
                        date_parser = parse_date, skiprows = skip_values , 
                        dtype = data_types )

# ## A separate data frame where clicks = 1
## convenient for usage at a later point
train_data_clicks = train_data[train_data['click']==1]
train_data.info() 

## Memory consumption reduced to 107.8 + MB

'''
% reduction in memory usage = 40% 
'''

'''
######## AppOnBoard Data Modeling Technical Assignment ######
''' ''' PART 1: EXPLORATORY DATA ANALYSIS '''
'''# Altering data types to reduce MEMORY CONSUMPTION.
# uint64 to uint16 ~~ reduction in block size memory/space
'''
train_data.describe()
train_data.head()
train_data.iloc[:, :24].head(5)

'''
24 features encompassing site attributes, application features, device attributes 

Target features - click 
>>C14 - C21 - Anonymized categorical variables 

Features kept anonymous via. md5 hashing encrypton : 
  
>>Site features - Site_id, Site_domain, Site_category
>>App features - app_id, app_domain 
>>Device features - device_type, device_conn_type 

'''
# ### CTR analysis ~ Click v/s No click distribution

#get_ipython().magic('matplotlib inline')

train_data.groupby('click').size().plot(kind = 'bar')
rows = train_data.shape[0]

click_through_rate = train_data['click'].value_counts()/rows 

click_through_rate

''' INSIGHT 1: 83.03% on average NOT clicked 
    16.94% on average clicked
    Note: Data set is unbalanced. Take into account sampling techniques.  
    '''

'''
FEATURE ENGINEERING - Dealing with each variable separately
Studying the relationships between different features 
and the target variable i.e 'Click'. 
Manipulating data in the process, introducing new metrics
'''

# ### HOUR 

'''
Metric 1. HOUR 
'''
train_data.hour.describe()

'''INSIGHT 2: Impressions V/S Clicks 
    MAXIMUM number of Impresisons around 1 P.M ~ 1561 '''

df_impressions = train_data.groupby('hour').agg({'click':'sum'})
#df_impressions
df_impressions.unstack().plot()

df_click = train_data[train_data['click']==1]
temp_click = df_click.groupby('hour').agg({'click' : 'sum'})
temp_click.unstack().plot()

train_data.hour.describe()

#Since Time Features are thought of in terms of cycles


''' HOUR as a metric is difficult to read because it is a time stamp 
    Introducing new metrics: 
     1. hour_in_day - Better KPI to assess the impressions v/s clicks behavior w.r.t hour in day
     2. weekday -- To study user behavior w.r.t clicks on each day 
     3. Day_name -- To extract the day name from the HOUR feature for a better understanding 
'''

train_data['hour_in_day'] = train_data['hour'].apply(lambda val : val.hour)
#train_data_clicks['hour_in_day'] = train_data_clicks['hour'].apply(lambda val : val.hour)

train_data['weekday'] = train_data['hour'].apply(lambda val: val.dayofweek)
#train_data_clicks['weekday'] = train_data_clicks['hour'].apply(lambda val: val.dayofweek)

train_data['day_name'] = train_data['hour'].apply(lambda x: x.strftime('%A'))
#train_data_clicks['day_name'] = train_data_clicks['hour'].apply(lambda x: x.strftime('%A'))

train_data.columns

# ### Hour In day, weekday and day_name columns added
# ##### Monday = 0, Sunday = 6

# # HOUR IN DAY

#train_data['hour_in_day'].nunique() ~ 0 TO 23
train_data.groupby(['hour_in_day', 'click']).size().unstack().plot(kind='bar', stacked=True, title="Hour in Day")
train_data[['hour','click']].groupby(['hour']).sum().sort_values('click',ascending=False)
train_data_clicks[['hour','click']].groupby(['hour']).sum().sort_values('click',ascending=False)

# ## Hour in day - CTR v/s impressions analysis
hour_df = pd.DataFrame()

hour_df['hr'] = train_data_clicks[['hour_in_day','click']].groupby(['hour_in_day']).count().reset_index().sort_values('click',ascending=False)['hour_in_day']
                        
hour_df
#hour_dataframe.drop("hr", axis = 1, inplace = True)

#train_data_clicks.head()


# ### Hour in day - Clicks
'''
Taking into account just the CLICKS 
'''
hour_df['pos_clicks'] = train_data_clicks[['hour_in_day','click']].groupby(['hour_in_day']).count().reset_index().sort_values('click',ascending=False)['click']
hour_df

### Hour in day - Impressions
'''
Taking into account the IMPRESSIONS
'''
hour_df['impressions_total'] = train_data[['hour_in_day','click']].groupby(['hour_in_day']).count().reset_index().sort_values('click',ascending=False)['click']
            
hour_df

#### Introducing Click through rate

'''
Introducing a new feature click through rate 
'''
hour_df['click_through_rate'] = 100*hour_df['pos_clicks']/hour_df['impressions_total']

#hour_df.sort_values(ascending = False, by = 'impressions_total')
hour_df.sort_values(ascending = False, by = 'click_through_rate')
list_of_hours = hour_df.sort_values(by='click_through_rate',ascending=False)['hr'].tolist()

import seaborn as sns
sns.barplot(y='click_through_rate',x='hr'            ,data=hour_df            ,order=list_of_hours)

## Weekday ~ day_name

train_data.groupby(['day_name','click']).size().unstack().plot(kind='bar', stacked=True, title="Day of the Week")
## weekday ~ day_name (for clicks)

train_data_clicks.groupby(['day_name','click']).size().unstack().plot(kind='bar', stacked=True, title="Day of the Week")
train_data_clicks[['day_name','click']].groupby(['day_name']).count().sort_values('click',ascending=False)

# ### Most clicks on Tuesday, then wednesday followed by Thursday

# # Day wise analysis of click through rates 
day_df = pd.DataFrame()
day_df['day'] = train_data_clicks[['day_name','click']].groupby(['day_name']).count().reset_index().sort_values('click',ascending=False)['day_name']
day_df           

# ### Day-wise clicks

day_df['pos_clicks'] = train_data_clicks[['day_name','click']].groupby(['day_name']).count()                        .reset_index()                        .sort_values('click',ascending=False)['click']
day_df
# ### Day-wise Impressions
day_df['total_impressions'] = train_data[['day_name','click']].groupby(['day_name']).count().reset_index().sort_values('click',ascending=False)['click']
day_df

day_df['click_pct'] = 100*day_df['pos_clicks']/day_df['total_impressions']
day_df.sort_values(ascending = False, by = 'click_pct')

# ### Sunday has the highest value of click through rate

list_of_days = day_df.sort_values(by='click_pct',ascending=False)['day'].tolist()
sns.barplot(y='click_pct',x='day'            ,data=day_df            ,order=list_of_days)


#####Banner Position #####

# ### Banner positions representing attractive and appealing designs that might highly affect a user's behavior and in turn trigger their decision to click. Or not. Hence making it an effective metric to predict clicks

train_data['banner_pos'].unique()


# #### It's unclear as to what the 7 banner positions (represented as integers) represent. Intuitively and based on research, the 7 positions might represent ad placing in a 2D webpage  

banner_temp =train_data[['banner_pos','click']].groupby(['banner_pos','click'])
banner_temp.size().unstack().plot(kind='bar',stacked=True, title='banner positions')
# #### Positions 0 and 1 ~ the most prominent banner positions garnering most impressions

train_data[['banner_pos','click']].groupby(['banner_pos']).count().sort_values('click',ascending=False)


# ### BANNER POSITIONS 0 and 1 generating most impressions and clicks

banner_temp =train_data_clicks[['banner_pos','click']].groupby(['banner_pos','click'])
banner_temp.size().unstack().plot(kind='bar',stacked=True, title='banner positions')

train_data_clicks[['banner_pos','click']].groupby(['banner_pos']).count().sort_values('click',ascending=False)

# ## CTR analysis on Banner position
# 

import pandas as pd 
banner_df = pd.DataFrame()

banner_df['position'] = train_data_clicks[['banner_pos','click']].groupby(['banner_pos']).count().reset_index().sort_values('click',ascending=False)['banner_pos']

banner_df['pos_clicks'] = train_data_clicks[['banner_pos','click']].groupby(['banner_pos']).count().reset_index().sort_values('click',ascending=False)['click']

banner_df['total_impressions'] = train_data[['banner_pos','click']].groupby(['banner_pos']).count().reset_index().sort_values('click',ascending=False)['click']

banner_df['click_pct'] = 100*banner_df['pos_clicks']/banner_df['total_impressions']

banner_df

banner_df.sort_values(ascending=False,by='click_pct')


# #### Banner 7 has the highest click through rate 

list_of_banners = banner_df.sort_values(by='click_pct',ascending=False)['position'].tolist()
sns.barplot(y='click_pct',x='position'            ,data=banner_df            ,order=list_of_banners)

'''#### Banner position 7 seems to be a nice choice 
for placing advertisements. As per click through rate. 
'''

## DEVICE TYPE Metrics

device_temp = train_data[['device_type','click']].groupby(['device_type','click'])

device_temp.size().unstack().plot(kind='bar',stacked=True, title='device types')


# ### Device type 1 getting most impressions among the 5 devices
train_data[['device_type','click']].groupby(['device_type']).count().sort_values('click',ascending=False)
train_data_clicks[['device_type','click']].groupby(['device_type','click']).size().unstack().plot(kind='bar',stacked=True, title='device types')

train_data_clicks[['device_type','click']].groupby(['device_type']).count().sort_values('click',ascending=False)


# ### Device Type 1 gets the maximum number of clicks too
device1_df = train_data_clicks[train_data_clicks['device_type']==1]
# extract CLICKS for DEVICE TYPE 1

# ### Hourly distribution of clicks on Device 1
temp_device_df = device1_df.groupby(['hour_in_day', 'click'])
temp_device_df.size().unstack().plot(kind='bar', stacked=True, title="Clicks spread across hour in day for Device 1")


# ### '''Device type 1 --- probably cell phone// Desktop Reasons ---
# Businesses might not prefer showing ads later in the evening-----
# after work hours// business hours ( Click spread max between 9 to 5 )
# #

# ## Click through rate analysis w.r.t Device type(merging data frames)
# 

# ### Had to merge data frames to ensure consistency

import pandas as pd
dev_type_df=pd.DataFrame()

dev_type_df_total_imp = pd.DataFrame()
#TOTAL CLICKS

dev_type_df = train_data_clicks.groupby('device_type').agg({'click':'sum'}).reset_index()

dev_type_df

#TOTAL IMPRESSIONS
dev_type_df_total_imp = train_data.groupby('device_type').agg({'click':'count'}).reset_index()

#dev_type_df_total_imp.drop([2], inplace = True)

dev_type_df_total_imp

dev_type_df['total_impressions'] = dev_type_df_total_imp['click']

dev_type_df

## sucess percentage == CTR

dev_type_df['success_pct'] = (dev_type_df['click']/dev_type_df['total_impressions'])*100

dev_type_df_total_imp.columns = ['device_type', 'click2']

merged_df = pd.merge(left = dev_type_df , right = dev_type_df_total_imp,
                    how = 'inner', on = 'device_type')
merged_df


# #del merged_df['total_impressions']
# 
# merged_df.columns = ['device_type', 'click','success_pct',
#                     'total_impressions']
# merged_df

merged_df['success_pct'] = 100*(merged_df['click']/merged_df['total_impressions'])

merged_df


# ### Device Type 0 with the highest click through rate

# ## App Related Metrics

# #### App_Id, App_Domain, App_Category

app_features = ['app_id', 'app_domain', 'app_category']

train_data.groupby('app_category').agg({'click':'sum'}).sort_values(by='click',ascending = False)
train_data['app_category'].value_counts().plot(kind='bar', title='App Category v/s Clicks')

# ### Studying Clicks behavior across different app categories

train_app_category = train_data.groupby(['app_category', 'click']).size().unstack()

train_app_category.div(train_app_category.sum(axis=1), axis=0).plot(kind='bar', stacked=True, title="Intra-category CTR")


# ## C1, C14-C21 features

features = ['C1', 'C14', 'C15', 'C16', 'C17', 'C18',
            'C20', 'C21']

train_data[features].astype('object').describe()
train_data.groupby(['C1', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C1 histogram')
train_data.groupby(['C15', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C1 histogram')
train_data.groupby(['C16', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C1 histogram')
train_data.groupby(['C18', 'click']).size().unstack().plot(kind='bar', stacked=True, title='C1 histogram')


'''
######## AppOnBoard Data Modeling Technical Assignment ######
''' ''' Part 2: Developing the prediction model '''
'''# Making use of Logistic Regression and XGBOOST techniques.
'''
# ### Using the key metrics discussed above as a part of the EDA to put
# together a predictive model in order to forecast Clicks

# ### Data preparation stage ~~ To be fed in the data pipeline 

model_features = ['weekday', 'hour_in_day',
                  'banner_pos', 'site_category',
                  'device_conn_type', 'app_category',
                  'device_type']
model_target = 'click'

train_model = train_data[model_features+[model_target]].sample(frac=0.1,random_state=42)


# #### Clubbing the model features with the target and selecting a fraction in order to speeden up computation

def one_hot_features(data_frame, feature_set):
    new_data_frame = pd.get_dummies(data_frame,
                                     columns = feature_set,
                                    sparse = True)

    return new_data_frame


# #### Features Site_category and App_category are hashed and need to be represented in a readable format

# #### Banner_pos is represented as integers hence we make use of one hot encoding to deal with all these features

train_model = one_hot_features(train_model,
                                ['site_category',
                                 'app_category',
                                 'banner_pos'])

train_data.head()

# ### Extracting all columns from the train model except the target mask column 
model_features = np.array(train_model.columns[train_model.columns!=model_target].tolist())

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    train_model[model_features].values,
    train_model[model_target].values,
    test_size=0.3,
    random_state=42
)


# ### Feature Selection ~ To reduce the dimensional space occupied and to
# deal with overfitting, use GRID SEARCH cross validation and regularization to
#  obtain trade off b/w number of features and F-1 score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score

# ### F1 score used as a performance metric because it represents the harmonic mean between precision and recall

num_splits = 3
c_values = np.logspace(-3,0,7)


stratified_k_fold = StratifiedKFold(n_splits=num_splits)

scores = np.zeros(7)
nr_params = np.zeros(7)


# ### Model: logistic Regression with L1 regularization and balanced class weights
for train_data, valid_data in stratified_k_fold.split(x_train,
                                                      y_train):
    for i, c in enumerate(np.logspace(-3, 0, 7)):
        lr_classify = LogisticRegression(penalty='l1',
                                         class_weight='balanced',
                                         C = c)
        lr_classify.fit(x_train[train_data],
                        y_train[train_data])

        #validation_Set evaluation

        y_prediction = lr_classify.predict(x_train[valid_data])
        score_f1 = f1_score(y_train[valid_data],
                            y_prediction, average='weighted' )

        scores[i] += score_f1 / num_splits

        ### spot the selected parameters ##

        model_selected = SelectFromModel(lr_classify, prefit=True)
        nr_params[i] += np.sum(model_selected.get_support()) / num_splits


plt.figure(figsize=(20, 10))
plt.plot(nr_params, scores)

for i, c in enumerate(c_values):
    plt.annotate(c, (nr_params[i], scores[i]))
plt.xlabel("Nr of parameters")
plt.ylabel("Avg F1 score")


# ### Parameters obtained using c = 0.1 manage to reduce parameters dimension which optimizes the execution time also improving generalization capacity. 
# 

lr_classify = LogisticRegression(C=0.1, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l1', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

lr_classify.fit(x_train, y_train)
model_selected = SelectFromModel(lr_classify,
                                 prefit=True )


pruned_params = model_selected.get_support()
pruned_params

model_features = model_features[pruned_params]

x_train = x_train[:, pruned_params]

x_test = x_test[:, pruned_params]


# ## Model : Gradient Boosting

# ### Part 3: Evaluating results using various performance metrics
###################################################################
###################################################################
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train,
    y_train,
    stratify=y_train,
    test_size=0.1,
    random_state=42)

model = XGBClassifier()
xgb_clf = model


# ### Log Loss values measuring the performances of a classification models
#  where the prediction label is a value between 0 and 1.
# The goal of the model is to minmize this value

xgb_clf.fit(x_train, y_train, early_stopping_rounds=10,
            eval_metric="logloss", eval_set=[(x_valid, y_valid)])

y_pred = xgb_clf.predict(x_test)
predictions = [round(value) for value in y_pred]

print(classification_report(y_test,
                            predictions))


# ### Other evaluation metrics: Accuracy score, Confusion Matrix, ROC/AUC score
from sklearn import metrics

print(metrics.accuracy_score(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.roc_auc_score(y_test, predictions))


# ### The model has an 83% accuracy score and 0.5 is the area
# under the receiver operating characteristic curve.
# ROCAUC implying the expected position of positives drawn  before a uniformly drawn random negative

# ## Saving the XGBoost and Logistic models


import pickle
filename = 'xgb_mod.sav'
filename2 = 'logistic.sav'
pickle.dump(xgb_clf,open(filename, 'wb' ))
pickle.dump(lr_classify, open(filename2, 'wb'))


# ## ############ End of AppOnBoard Technical modeling assignment ###############
'''
######## AppOnBoard Data Modeling Technical Assignment ######
 `##### Submitted by : Rachit Mishra #####
##### AVAZU  -- Click Through rate Prediction #####
   ###### Predict whether an ad will be clicked or not #######
'''

