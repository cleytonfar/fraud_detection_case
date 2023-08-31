from utils import generate_dataset, add_frauds

# simulate data:
# 5000 clientes
# 10000 terminais
# 100 dias de transação.

customer_profiles_table, terminal_profiles_table, transactions_df=generate_dataset(
    n_customers = 5000,
    n_terminals = 10000,
    nb_days=100,
    start_date="2023-06-01",
    r=5
)

# Generation of fraud scenarios
transactions_df = add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df)

# saving:
transactions_df.to_pickle("data/simulated-data-raw/fraud.pkl")

