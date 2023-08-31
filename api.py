from flask import Flask, request
from joblib import load
import pandas as pd

app = Flask(__name__)

@app.route("/fraud_alert", methods=["POST"])
def fraud_alert():
    # catching json:
    json_ = request.get_json()
    # convert to DF:
    df = pd.DataFrame(json_)
    # convert to datetime
    df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
    # creating features:    
    # - whether a transaction occurs during a weekday or a weekend
    df["TX_DURING_WEEKEND"] = (df["TX_DATETIME"].dt.weekday >= 5).astype("int")
    # - whether a transaction is at night: night definition: 22 <= hour <= 6
    df["TX_DURING_NIGHT"] = ((df["TX_DATETIME"].dt.hour <= 6) | (df["TX_DATETIME"].dt.hour >= 22)).astype("int")
    # customer features
    df = pd.merge(df, dict_customer, on="CUSTOMER_ID", how="left")
    # terminal features
    df = pd.merge(df, dict_terminal, on="TERMINAL_ID", how="left")
    # input features:
    input_features = [
    'TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT',
    'CUSTOMER_ID_NB_TX_1DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
    'CUSTOMER_ID_NB_TX_7DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
    'CUSTOMER_ID_NB_TX_30DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW',
    'TERMINAL_ID_NB_TX_1DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW',
    'TERMINAL_ID_NB_TX_7DAY_WINDOW', 'TERMINAL_ID_RISK_7DAY_WINDOW',
    'TERMINAL_ID_NB_TX_30DAY_WINDOW', 'TERMINAL_ID_RISK_30DAY_WINDOW'
    ]
    # prediction:
    df["prob_fraud"] = mdl.predict_proba(df[input_features])[:, 1]
    
    return df[["TRANSACTION_ID", "prob_fraud"]].to_dict(orient="records")


if __name__ == "__main__":
    # loading dicts:
    dict_customer = load("output/dict_customer.joblib")
    dict_terminal = load("output/dict_terminal.joblib")
    # loading model:
    mdl = load("output/model.joblib")
    # running debug mode:
    app.run(debug=True)


