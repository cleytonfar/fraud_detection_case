import pandas as pd
import numpy as np
import datetime
from utils import feature_engineering_customer_spending_behaviour, feature_engineering_terminal_risk, get_train_test_set
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import GridSearchCV

# Reading:
transactions_df = pd.read_pickle("data/simulated-data-raw/fraud.pkl")
transactions_df

# 1. Feature Engineering

# 1.1. date and time transformations

# - whether a transaction occurs during a weekday or a weekend
transactions_df["TX_DURING_WEEKEND"] = (transactions_df["TX_DATETIME"].dt.weekday >= 5).astype("int")

# - whether a transaction is at night:
# night definition: 22 <= hour <= 6
transactions_df["TX_DURING_NIGHT"] = ((transactions_df["TX_DATETIME"].dt.hour <= 6) | (transactions_df["TX_DATETIME"].dt.hour >= 22)).astype("int")


# 1.2 Customer ID transformations

# - number of transactions that occur within a time window
# - average amount spent in these transactions
transactions_df = transactions_df.groupby("CUSTOMER_ID").apply(lambda x: feature_engineering_customer_spending_behaviour(x, windows_size_in_days=[1, 7, 30]))
transactions_df.reset_index(drop=True, inplace=True)
transactions_df.sort_values(by="TX_DATETIME", inplace=True)
transactions_df

# 1.3 Terminal ID transformations
# The main goal is to extract a risk score.  The risk score will be defined as the average number of fraudulent transactions that occurred on a terminal ID over a time window. 

# Contrary to customer ID transformations, the time windows will not directly precede a given transaction. Instead, they will be shifted back by a delay period. The delay period accounts for the fact that, in practice, the fraudulent transactions are only discovered after a fraud investigation or a customer complaint. 

# Hence, the fraudulent labels, which are needed to compute the risk score, are only available after this delay period. To a first approximation, this delay period will be set to one week.

# Let us perform the computation of the risk scores by defining a get_count_risk_rolling_window function. The function takes as inputs the DataFrame of transactions for a given terminal ID, the delay period, and a list of window sizes. In the first stage, the number of transactions and fraudulent transactions are computed for the delay period (NB_TX_DELAY and NB_FRAUD_DELAY). In the second stage, the number of transactions and fraudulent transactions are computed for each window size plus the delay period (NB_TX_DELAY_WINDOW and NB_FRAUD_DELAY_WINDOW). The number of transactions and fraudulent transactions that occurred for a given window size, shifted back by the delay period, is then obtained by simply computing the differences of the quantities obtained for the delay period, and the window size plus delay period.

# The risk score is finally obtained by computing the proportion of fraudulent transactions for each window size (or 0 if no transaction occurred for the given window).

#Additionally to the risk score, the function also returns the number of transactions for each window size. This results in the addition of six new features: The risk and number of transactions, for three window sizes.
transactions_df=transactions_df.groupby('TERMINAL_ID').apply(lambda x: feature_engineering_terminal_risk(x, delay_period=7, windows_size_in_days=[1,7,30], feature="TERMINAL_ID"))
transactions_df.sort_values('TX_DATETIME', inplace=True)
transactions_df.reset_index(drop=True, inplace=True)
transactions_df


# 3. Estimation

# Defining the training and test sets

# We will use the transactions from the 2018-07-25 to the 2018-07-31
# for the training set;
# data from the 2018-08-08 to the 2018-08-14 for the test set,

# It is worth noting that we choose our test set to take place one
# week after the last transaction of the training set.
# In a fraud detection context, this period separating the training
# and test set is referred to as the 'delay period' or 'feedback
# delay'.
# It accounts for the fact that, in a real-world fraud detection
# system, the label of a transaction (fraudulent or genuine) is only
# known after a customer complaint, or thanks to the result of a fraud
# investigation.

start_date = datetime.datetime.strptime("2023-08-11", "%Y-%m-%d")
print(start_date)

# separating train and test set sequentially with a delay period:
train_df, test_df=get_train_test_set(
    transactions_df,
    start_date_training=start_date,
    delta_train=7,
    delta_delay=7,
    delta_test=7
)

train_df["TX_DATETIME"].dt.date.min()
train_df["TX_DATETIME"].dt.date.max()

test_df["TX_DATETIME"].dt.date.min()
test_df["TX_DATETIME"].dt.date.max()

# Baseline Estimation
output_feature="TX_FRAUD"
input_features=['TX_AMOUNT',
                'TX_DURING_WEEKEND', 'TX_DURING_NIGHT',
                'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
                'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
                'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW',
                'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                'TERMINAL_ID_RISK_1DAY_WINDOW',
                'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                'TERMINAL_ID_RISK_7DAY_WINDOW',
                'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                'TERMINAL_ID_RISK_30DAY_WINDOW']

y_train = train_df[output_feature]
X_train = train_df[input_features]

y_test = test_df[output_feature]
X_test = test_df[input_features]

# dummy model:
acc_dummy = accuracy_score(y_test, [y_train.mean() > .5]*len(y_test))
f1_dummy = f1_score(y_test, [y_train.mean() > .5]*len(y_test))
avg_prec_dummy = average_precision_score(y_test,[y_train.mean()]*len(y_test))
gmean_dummy = geometric_mean_score(y_test,[y_train.mean() > .5]*len(y_test))
auc_dummy = roc_auc_score(y_test, [y_train.mean()]*len(y_test))

# convert to DF
dummy_res = pd.DataFrame(
    {"clf": ["dummy"],
     "acc": [acc_dummy],
     "f1_score": [f1_dummy],
     "gmean": [gmean_dummy],
     "auc_dummy": [auc_dummy],
     "average_precision": [avg_prec_dummy]}
)

# creating a empty list:
res = []

# append first result:
res.append(dummy_res)

# Estimamting ml algorithms
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# candidates:
clfs = {
    "knn": KNeighborsClassifier(),
    "logistic": LogisticRegression(max_iter=2000),
    "tree": DecisionTreeClassifier(),
    "bagging": BaggingClassifier(estimator=DecisionTreeClassifier()),
    "rf": RandomForestClassifier(),    
    "GradientBoost": GradientBoostingClassifier()
}

for name, clf in clfs.items():
    # instantiate a pipeline:
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer()),
        ("scaling", StandardScaler()),        
        ("clf", clf)
    ])
    # fitting to training data:
    pipe.fit(X_train, y_train)
    # predicting on test set:
    y_pred = pipe.predict_proba(X_test)
    # Evaluating:    
    # threshold-based metrics:
    ## accuracy
    acc = accuracy_score(y_test, y_pred[:,1]>.5)
    ## f1-score
    f1 = f1_score(y_test, y_pred[:,1]>.5)
    ## g-mean
    g_mean = geometric_mean_score(y_test, y_pred[:,1]>.5)
    # threshold-free metrics:
    auc = roc_auc_score(y_test, y_pred[:,1])
    # average precision
    avg_prec = average_precision_score(y_test, y_pred[:, 1])    
    # organizing:
    eval = pd.DataFrame(
        {"clf": [name],
         "acc": [acc],
         "f1_score": [f1],
         "gmean": [g_mean],
         "average_precision": [avg_prec]
         }
    )    
    res.append(eval)

# concatenate results
res = pd.concat(res)
res = res.sort_values("average_precision", ascending=True).reset_index(drop=True)
print(res)


## -------------------------------------------------------------------------------- ## 

# The limitation with the last estimation strategy is that we only
# have one estimation of the performance on real world scenario.

# cv strategy:
def sequential_train_test_split(transactions_df,
            start_date_training,
            n_folds=5,
            delta_train=7,
            delta_delay=7,
            delta_val=7):    
    sequential_split_indices = []
    # For each fold
    for fold in range(n_folds):
        # Shift back start date for training by the fold index times the validation period
        start_date_training_fold = start_date_training-datetime.timedelta(days=fold*delta_val)
        start_date_training_fold
        # Get the training and test (assessment) sets
        (train_df, val_df)=get_train_test_set(
            transactions_df,
            start_date_training=start_date_training_fold,
            delta_train=delta_train,
            delta_delay=delta_delay,
            delta_test=delta_val
        )
        # Get the indices from the two sets, and add them to the list of sequential splits
        indices_train = list(train_df.index)
        indices_val = list(val_df.index)
        sequential_split_indices.append((indices_train,indices_val))
    return sequential_split_indices

# determing the folders:
n_folds=5
delta_train=7
delta_delay=7
delta_val=7

# defining start date in the sequential cv:
start_date_training_seq = start_date+datetime.timedelta(days=-(delta_delay+delta_val))
start_date_training_seq

# creting the indexes for each sequential cv:
sequential_split_indices = sequential_train_test_split(
    transactions_df,
    start_date_training = start_date_training_seq,
    n_folds=n_folds,
    delta_train=delta_train,
    delta_delay=delta_delay,
    delta_val=delta_val
)

# checking dates of training:
for i in list(range(n_folds))[::-1]:   
    train_data1 = transactions_df.loc[sequential_split_indices[i][0]]["TX_DATETIME"].min()
    train_data2 = transactions_df.loc[sequential_split_indices[i][0]]["TX_DATETIME"].max()

    val_data1 = transactions_df.loc[sequential_split_indices[i][1]]["TX_DATETIME"].min()
    val_data2 = transactions_df.loc[sequential_split_indices[i][1]]["TX_DATETIME"].max()

    print("train range: "+datetime.datetime.strftime(train_data1, "%Y-%m-%d")+" - "+datetime.datetime.strftime(train_data2, "%Y-%m-%d"))
    print("val range: "+datetime.datetime.strftime(val_data1, "%Y-%m-%d")+" - "+datetime.datetime.strftime(val_data2, "%Y-%m-%d"))
    print("\n")

# creating a pipeline:
myPipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),    
    ('scaling', StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000))
])

# Hyperparameters to test
myGrid = [
    {
        "clf": [LogisticRegression(solver="liblinear", max_iter=2000)],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.5, 5]
    },
    {
        "clf": [BaggingClassifier(n_estimators=200,random_state=10) ],
        "clf__estimator": [DecisionTreeClassifier(), LogisticRegression()],
        "clf__max_features": [1, .5]
    },
    {     
        "clf": [RandomForestClassifier(n_estimators=200,random_state=10)]
    },        
    {        
        "clf": [GradientBoostingClassifier(learning_rate=.05,random_state=10)],
        "clf__n_estimators": [100, 200]
    }    
]

# Let us instantiate the GridSearchCV
# set refit=False. Do retrained the best model on the whole data.
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(
    myPipe,
    param_grid=myGrid,
    scoring="average_precision",
    cv=sequential_split_indices,
    refit=False,
    n_jobs=-1,
    verbose=4
)

# fitting:
grid.fit(transactions_df[input_features], transactions_df[output_feature])

# getting the results:
results = pd.DataFrame(grid.cv_results_)
# assuming normal distribution:
results["lower_bd"] = results["mean_test_score"]-2*results["std_test_score"]
results["upper_bd"] = results["mean_test_score"]+2*results["std_test_score"]
# sorting:
results = results.sort_values("rank_test_score", ascending=True).reset_index(drop=True)

# Estimating the final model:

# I use the model configuration with the best score - 1 standard deviation.

# Defining a Custom best model criteria
def get_bestModel(cv_results):
    # copying
    cv_results = results.copy()
    # sorting by ranking:
    cv_results = cv_results.sort_values("rank_test_score").reset_index()
    # threshold: max(mean_test_score) - 1*std(mean_test_score)
    threshold = cv_results["mean_test_score"].max() - 1*cv_results["mean_test_score"].std()
    # filtering candidates with score greater than threshold:
    cv_results = cv_results[cv_results["mean_test_score"] > threshold]
    # From the candidates, select the one with the smallest score:
    # get index:
    best_model_idx = cv_results["mean_test_score"].idxmin()
    # get model:
    best_params = cv_results.loc[best_model_idx]["params"].copy()
    for name in list(best_params.keys()):
        newkey = name.split("__")[-1]
        best_params[newkey]=best_params.pop(name)
    best_model = best_params.pop("clf")
    best_model.set_params(**best_params)

    return best_model_idx, best_model

# get the best configuration:
idx, best_clf_config = get_bestModel(results)
results.loc[idx]

# final pipeline:
mdl = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),    
    ('scaling', StandardScaler()),
    ("clf", best_clf_config)
])

# fitting:
mdl.fit(X_train, y_train)

# predicting on test set:
y_pred = mdl.predict_proba(X_test)
# average precision score:
avg_prec_test = average_precision_score(y_test, y_pred[:, 1])

print(f"CV prediction: {np.round(results.loc[idx, 'mean_test_score'], 2)}")
print(f"CV confidence interval: [{np.round(results.loc[idx, 'lower_bd'], 2)}; {np.round(results.loc[idx, 'upper_bd'], 2)}]")
print(f"Test score: {np.round(avg_prec_test, 2)}")

# The AUC ROC can be interpreted as the probability that the scores
# given by a classifier will rank a randomly chosen positive instance
# higher than a randomly chosen negative one.

# auc
# ap
# cp@k: Card Precision top-k for day d.


# the relevance of a credit card FDS from a more operational perspective,
# by explicitly considering their benefits for fraud investigators.
# Let us first recall that the purpose of an FDS is to provide
# investigators with alerts, that is, a set of transactions that are
# considered to be the most suspicious.

# - These transactions are manually checked, by contacting the cardholder.
# - The process of contacting cardholders is time-consuming,
# - the number of fraud investigators is limited.
# - The number of alerts that may be checked during a given period is therefore necessarily limited.

# Precision top-k metrics aim at quantifying the performance of an FDS in this setting.
# The parameter quantifies the maximum number of alerts that can be checked by investigators in a day.
# 
# the performance of a classifier is to maximize the precision in the subset of k alerts for a given day.
# his quantity is referred to as the Precision top-k for day d.
test_df["predictions"] = y_pred[:, 1]

def precision_top_k_day(df_day, top_k=100):
    # sorting by the highest chance of fraud:
    df_day = df_day.sort_values(by="predictions", ascending=False)
    # Get the top k most suspicious transactions
    df_day_top_k=df_day.head(top_k)
    # Compute precision top k
    precision_top_k = df_day_top_k["TX_FRAUD"].mean()
    return precision_top_k

prec_top_k_test = []
for day in test_df["TX_TIME_DAYS"].unique():
    prec_top_k_test.append(precision_top_k_day(test_df.query("TX_TIME_DAYS == @day")))

np.mean(prec_top_k_test)
np.std(prec_top_k_test)


# Ok. I would like to estimate an AI model in order to optimize this performance.
# We need to convert in a way that sklearn can understand so we can use its 
# functionalities.

# In order to do that, we have to use the sklearn.metrics.make_scorer.

# create a function that receivies y_true, y_pred and computes the daily precision:
def daily_avg_precision_top_k(y_true, y_pred, top_k, transactions_df):
    #y_true = y_test
    #y_pred = y_pred[:, 1]
    # get the test data:
    df = transactions_df.loc[y_true.index]
    # adding prediction
    df["predictions"] = y_pred
    # computing daily avg precision top-k:
    avg = df.groupby("TX_TIME_DAYS").apply(precision_top_k_day).mean()
    return avg

from sklearn.metrics import make_scorer
daily_avg_precision_top_k_score = make_scorer(
    daily_avg_precision_top_k,
    greater_is_better=True,
    needs_proba=True,
    top_k=100,
    transactions_df=transactions_df[['CUSTOMER_ID', 'TX_FRAUD','TX_TIME_DAYS']]
)

# Estimating:

# using the same pipe and grid:
myPipe
myGrid

# Let us instantiate the another GridSearchCV. This time with
# our custom scorer:
grid2 = GridSearchCV(
    myPipe,
    param_grid=myGrid,
    scoring=daily_avg_precision_top_k_score,
    cv=sequential_split_indices,
    refit=False,
    n_jobs=-1,
    verbose=4
)

# fitting:
grid2.fit(transactions_df[input_features], transactions_df[output_feature])

# getting the results:
results2 = pd.DataFrame(grid2.cv_results_)
# assuming normal distribution:
results2["lower_bd"] = results2["mean_test_score"]-2*results2["std_test_score"]
results2["upper_bd"] = results2["mean_test_score"]+2*results2["std_test_score"]
results2 = results2.sort_values("rank_test_score", ascending=True).reset_index(drop=True)

# get the best configuration:
idx2, best_clf_config2 = get_bestModel(results2)
results2.loc[idx2]
# final pipeline:
mdl2 = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),    
    ('scaling', StandardScaler()),
    ("clf", best_clf_config2)
])

# fitting:
mdl2.fit(X_train, y_train)

# predicting on test set:
y_pred2 = mdl2.predict_proba(X_test)

# daily average precision top_100 score:
avg_prec_top_100 = daily_avg_precision_top_k(y_test, y_pred2[:, 1], top_k=100, transactions_df=transactions_df)

# average precision score:
avg_prec_test = average_precision_score(y_test, y_pred2[:, 1])

print(f"CV prediction: {np.round(results2.loc[idx2, 'mean_test_score'], 2)}")
print(f"CV confidence interval: [{np.round(results2.loc[idx2, 'lower_bd'], 2)}; {np.round(results2.loc[idx2, 'upper_bd'], 2)}]")
print(f"Test score: {np.round(avg_prec_top_100, 2)}")




## PR curve
# the Precision-Recallcurve (PR curve) is obtained by plotting the
# precision againt the recall for all different classification
# thresholds. The main advantage of the PR curve is to put in evidence
# classifiers that can have both a high recall and a high precision.
# Let us now plot the PR curve, and compute its AUC. We will use the
# Average Precision (AP), which summarizes such a plot as the weighted
# mean of precisions achieved at each threshold, with the increase in
# recall from the previous threshold used as the weight .
average_precision_score(y_test, y_pred[:, 1])

# the performance of a random classifier depends on the class
# imbalance. It is 0.5  in the balanced case, and  P/(N+P) in the
# general case, where  P is the number of positive examples, and  N
# the number of negative examples.

# This property makes the AP more interesting than the AUC ROC in a
# fraud detection problem, since it better reflects the challenge
# related to the class imbalance problem (the AP of a random
# classifier decreases as the class imbalance ratio increases). 

# Precision top k:
# The Precision top-k can be computed for a day by ranking all fraud
# probabilities by decreasing order, and computing the precision for
# the top ranked transactions.
# When a test set spans multiple days, let be the mean of Precision
# top-k for a set of days. 

# Card Precision top-k:
# Multiple fraudulent transactions from the same card should count as
# a single correct, so the Card Precision top-k quantifies the number
# of correctly detected compromised cards out of the cards which have
# the highest risks of frauds.
# Instead of simply sorting transactions by decreasing order of their
# fraud probabilities, we first group transactions by customer ID. For
# each customer ID, we then take the maximum value of the fraud
# probability and the fraud label.
# The card precision top-k is then computed by sorting customer IDs by
# decreasing order of their fraud probabilities and computing the
# precision for the set of cards with the highest fraud probabilities. 
train_df["predictions"] = clf.predict_proba(train_df[input_features])[:,1]
test_df["predictions"] = clf.predict_proba(test_df[input_features])[:,1]
transactions_df.columns

## --------------------------------------------------------------------- ## 
# DEPLOYMENT

# getting the terminal and customer dict:
# save final model:
from joblib import dump
dump(mdl2, "output/model.joblib")

# save dict

# get the last info from terminal
dict_terminal = transactions_df[["TX_DATETIME", "TERMINAL_ID",
                                 'TERMINAL_ID_NB_TX_1DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW',
                                 'TERMINAL_ID_NB_TX_7DAY_WINDOW', 'TERMINAL_ID_RISK_7DAY_WINDOW',
                                 'TERMINAL_ID_NB_TX_30DAY_WINDOW', 'TERMINAL_ID_RISK_30DAY_WINDOW']].sort_values("TX_DATETIME").groupby("TERMINAL_ID").tail(1)
dump(dict_terminal, "output/dict_terminal.joblib")

# get the last info from customer:
dict_customer = transactions_df[["TX_DATETIME",
                                 "CUSTOMER_ID",
                                 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                                 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
                                 'CUSTOMER_ID_NB_TX_7DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
                                 'CUSTOMER_ID_NB_TX_30DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW',]].sort_values("TX_DATETIME").groupby("CUSTOMER_ID").tail(1)
dump(dict_customer, "output/dict_customer.joblib")
