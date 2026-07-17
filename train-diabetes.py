from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import mlflow
import dagshub
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/refs/heads/master/diabetes.csv")

mlflow.set_tracking_uri('https://dagshub.com/jaymavani16/mlflow-diabetes-practice.mlflow')

dagshub.init(repo_owner='jaymavani16', repo_name='mlflow-diabetes-practice', mlflow=True)


# splitting data into features and train, test data
X = df.drop('Outcome',axis=1)
y = df['Outcome']

# performing train, test split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.2)

# creating random forest classifier model 
rf = RandomForestClassifier(random_state=42)

# defining parameter grid and grid search cv
param_grid = {
    "n_estimators" : [10,50,100],
    "max_depth" : [None,10,15,25,22]
    }

# Applying grid search CV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, cv=5, verbose=2)

mlflow.set_experiment('diabetes-prediction_cv')
with mlflow.start_run(run_name='grid_search') as parent:
    grid_search.fit(X_train, y_train)

    # log all the children
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy",grid_search.cv_results_['mean_test_score'][i])

    
    # Displaying best parameters 
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # log params
    mlflow.log_params(best_params)

    # log Metrics
    mlflow.log_metric("accuracy",best_score)

    # adding signature
    signature = mlflow.models.infer_signature(X_train,grid_search.best_estimator_.predict(X_train))

    # log model 
    mlflow.sklearn.log_model(grid_search.best_estimator_,'Random Forest',signature=signature)

    # log data
    train_df = X_train
    train_df['Outcome'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"Training data")

    test_df = X_test
    test_df['Outcome'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df,'Validation data')

    # log source codes 
    mlflow.log_artifact(__file__)

    # tags 
    mlflow.set_tag("Author", 'Jay')


    print(best_params)
    print(best_score)