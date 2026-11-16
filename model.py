import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

#reading the dataset --
df=pd.read_csv("/Users/aryanpal/Downloads/Student Social Media And Mental Health Impact.csv")
print(df.shape)
print(df.head().T.to_string())
print(df.info())
print(df.describe().T.to_string())

#checking the data 
print(df.isnull().sum())
print(df.duplicated().sum())

#now we will perform (EDA)

#for checking the  datga skewness of mental health score 
sn.histplot(df["Mental_Health_Score"],kde=True)

#checking correlation between all the columns
sn.heatmap(df.corr(numeric_only=True),annot=True)

#stress level vs mental health score 
print(df["Stress_Level"].unique())
#from box plot we can know about outliers in the column
sn.boxplot(x='Stress_Level',y="Mental_Health_Score",data=df)


#daily usage hours vs mental score
sn.scatterplot(x='Avg_Daily_Usage_Hours',y='Mental_Health_Score',data=df)


#sleephour per night vs mental score
sn.scatterplot(x='Sleep_Hours_Per_Night',y='Mental_Health_Score',data=df)


#most used platforms=
plt.figure(figsize=(10,10))
sn.countplot(x='Most_Used_Platform',data=df,order=df['Most_Used_Platform'].value_counts().index)


#checking outliers
numerical_features=df.select_dtypes(include='number')
Q1=numerical_features.quantile(0.25)
Q3=numerical_features.quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-IQR*1.5
upper_bound=Q3+IQR*1.5

outliers=(numerical_features < lower_bound)| (numerical_features > upper_bound)
#numerical_features=(numerical_features > lower_bound)| (numerical_features < upper_bound)
print(outliers.sum())

#i am removing outliers by cliping lower and upperbound values 
df[numerical_features.columns] = numerical_features.clip(
    lower=lower_bound,
    upper=upper_bound,
    axis=1
)

#now coming to the data cleaning part in which we have already performed outliers handeling 

#handeling duplicate values
df=df.drop_duplicates()

#handleling one all negative values in Physical_Activity_Hours which is practicaly not possible
df['Physical_Activity_Hours']=df['Physical_Activity_Hours'].clip(lower=0)
print(df.describe().T.to_string())

#now i will check skewness of each numerical coloumn
num_col=df.select_dtypes(include='number')
print(num_col.skew())

#now we are going to make country column ready for the one hot encoding if we are going to do it without changing than it has 111 different values which will creates 111 new coumn which will increase the burden of the model 
grouped_countries=df['Country'].value_counts().index[:10].to_list() ##we are taking only the top 10 index countries in the grouped and converting them in to list 
print(grouped_countries)

#i will create a function that will directly store the top 10 contries as it is and rest in other section
def group_country(countries):
    if countries in grouped_countries:
        return countries
    else:
        return "newother"
    
## this is where we call the function directly to seprate top 10 contries as it and remaining all contries in others section 
df['g_countries']=df['Country'].apply(group_country)    #apply(function ) its a direct method to apply the fucntion direclty on any data 
print(df['g_countries'].value_counts())


# to prevent the data leakage problem we will first do the train test split so the data of test should not leak to training 
#before train test split we will apply the log transformation in skewed coloumn and in other numerical column normak standard scalar and then we will perform ordinal encoding and one hot encoding in rest of the catagorical columns 
skewed_col=['Study_Hours']
other_numeric_col=['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night']
ordinal_col=['Stress_Level']
other_object_col=['Gender','Academic_Level', 'Most_Used_Platform','Purpose_Of_Use','g_countries']
feature_column=skewed_col+other_numeric_col+ordinal_col+other_object_col
x=df[feature_column]
y=df['Mental_Health_Score']
#now we are ready for the train test split --
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

# as we have different parts to implement on different columns so we can use column transfomer and pipelines in this case
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler,OrdinalEncoder,OneHotEncoder
#1-pipline for skewed features (this pipeline is for applying logtranfarmer and then normal standard scaler on input skeewed column )
skew_pipeline=Pipeline(steps=[('skew',FunctionTransformer(np.log1p)),
                              ('scaler',StandardScaler())])
#2-pipeline for normal numerical feature
numeric_pipeline=Pipeline(steps=[('scaler',StandardScaler())]) 

#3-pipeline for ordinal encoding feature 
ordinal_pipeline=Pipeline(steps=[('encode',OrdinalEncoder(categories=[['Low','Medium','High','Very High']]))])

#4pipeline for onehot encodig feture
onehot_pipeline=Pipeline(steps=[('encode',OneHotEncoder(handle_unknown="ignore"))])

##now we will make coloumn tranformer for giving the actual coloum to particular pipeline 
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(transformers=[('skewfeature',skew_pipeline,skewed_col),
                                ('numericfeature',numeric_pipeline,other_numeric_col),
                                ('ordinalfeature',ordinal_pipeline,ordinal_col),
                                ('onehotfeature',onehot_pipeline,other_object_col)])

#now we are going to apply first pipline of our linear regression model 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

lr_pipeline=Pipeline(steps=[('preprocessor',preprocessor),
                            ('model',LinearRegression())])

lr_pipeline.fit(x_train,y_train)
lr_preds=lr_pipeline.predict(x_test)
lr_preds_train=lr_pipeline.predict(x_train)

print(f"r2 score  of test data is {r2_score(y_test,lr_preds)}")
print(f"r2 score of train data is {r2_score(y_train,lr_preds_train)}")
print(f"mean eroor of test data is {mean_absolute_error(y_test,lr_preds)}")
print(f"mean square of test data is {mean_squared_error(y_test,lr_preds)}")


##now we will see another model for better accuracy specifically random forest 
from sklearn.ensemble import RandomForestRegressor

Rn_pipeline=Pipeline(steps=[('preprocessor',preprocessor),
                            ('random forest',RandomForestRegressor())])

Rn_pipeline.fit(x_train,y_train)
Rn_preds=Rn_pipeline.predict(x_test)
Rn_preds_train=Rn_pipeline.predict(x_train)

print("/")
print("random forest model results")
print(f"r2 score  of test data is {r2_score(y_test,Rn_preds)}")
print(f"r2 score of train data is {r2_score(y_train,Rn_preds_train)}")
print(f"mean eroor of test data is {mean_absolute_error(y_test,Rn_preds)}")
print(f"mean square of test data is {mean_squared_error(y_test,Rn_preds)}")


#now we can see that random forest is performing well here in this dataset and as we can see that there is 
# is a large differnce in accuracy of train and testing data so we will perform hyperparametric tuning of this data to reduce the error
from sklearn.model_selection import RandomizedSearchCV

param_grid={
    'random forest__n_estimators':[100,200,300],
    'random forest__max_depth':[5,10,15],
    'random forest__min_samples_split':[2,5,10],
    'random forest__min_samples_leaf':[1,2,4]
}

random_search=RandomizedSearchCV(estimator=Rn_pipeline,
                                 param_distributions=param_grid,
                                 n_iter=15,
                                 scoring='r2',
                                 random_state=42,
                                 cv=5,
                                 n_jobs=-1) ## njobs is for free proeccsor use 

random_search.fit(x_train,y_train)
print(" ")
print(" best parameter of hyperparameter tunning")
print(random_search.best_params_)

#now we check the accracy of hyperparametric tuned model 
best_pipeline_model=random_search.best_estimator_
best_preds=best_pipeline_model.predict(x_test)
best_preds_train=best_pipeline_model.predict(x_train)

print("hyper parametric tuned  model results")
print(f"r2 score  of test data is {r2_score(y_test,best_preds)}")
print(f"r2 score of train data is {r2_score(y_train,best_preds_train)}")
print(f"mean eroor of test data is {mean_absolute_error(y_test,best_preds)}")
print(f"mean square of test data is {mean_squared_error(y_test,best_preds)}")


#now we will import our best model in pkl format  for integration with fast api
import joblib
joblib.dump(best_pipeline_model,'Mental_health_model.pkl')




