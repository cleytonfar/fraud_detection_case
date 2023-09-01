# Stone Case
Cleyton Farias
August, 2023

### Pré-requisitos <a name="prerequisites"></a>

Este repositório contém scripts escritos em Python. Para configurar
todas asdependências, é essencial que você utilize a ferramenta de
gerenciamento de ambiente virtual do Python chamada
[`pipenv`](https://pipenv.pypa.io/en/latest/).

Após clonar este repositório, acesse o diretório correspondente. Haverá
um arquivo chamado `Pipfile.lock` especificando todas as dependências
utilizadas. Abra um terminal e execute o seguinte comando:

    pipenv sync

Este comando irá instalar todos os pacotes e suas respectivas versões
necessárias para executar os scripts.

## Credit Card Fraud Detection

Este exercício consiste no desenvolvimento de um sistema simples para
detectar fraudes em transações com cartão de crédito, utilizando dados
simulados.

Por meio da aplicação de algoritmos de *machine learning* e uma
estratégia de estimação que considera o aspecto temporal do problema,
foi possível criar uma abordagem sólida e confiável para estimar o
desempenho dos algoritmos e prever casos futuros.

Por fim, foi explorado como esse sistema pode ser implementado em
produção, transformando-o em uma RESTful API.

## Conteúdo

- [Data](#data)
  - [Data Understanding](#data-understanding)
  - [Feature Engineering](#feature-engineering)
- [Data Preparation](#data-preparation)
- [Modelling](#modelling)
- [Evaluation](#evaluation)
- [Deployment](#deployment)

## Data <a name="data"></a>

Dados públicos sobre transações reais de cartões de crédito são bastante
raros. Existe apenas uma [base de dados
disponível](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)
na plataforma kaggle disponibilizada pelo [Machine Learning
Group](https://mlg.ulb.ac.be/wordpress/). Apesar de sua limitação em
representar cenários reais, essa base tem sido extensivamente utilizada
por pesquisadores, profissionais e praticantes de modelagem estatística.

Com o intuito de disponibilizar o acesso de base de dados sobre o tema
para, o [Machine Learning Group](https://mlg.ulb.ac.be/wordpress/)
desenvolveu um simulador de dados de transações de cartões de crédito
que permite a criação de dados sintéticos que contempla características
reais de dados desse tipo.

Em particular, o simulador[^1] permite a criação de um conjunto de dados
com classes (transações fraudulentas ou não) desbalanceadas, incorpora
variáveis numéricas e categóricas (incluindo características categóricas
com um grande número de valores possíveis) e variáveis com
características *time-dependent*.

Para esse exercício foi gerado um amostra de transações no cartão de
crédito para 5000 clientes, em 10000 terminaios, durante 100 dias,
iniciado na data de `2023-06-01`.

``` python
from utils import generate_dataset, add_frauds

# simulate data for:
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
transactions_df.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | TRANSACTION_ID | TX_DATETIME         | CUSTOMER_ID | TERMINAL_ID | TX_AMOUNT | TX_TIME_SECONDS | TX_TIME_DAYS | TX_FRAUD | TX_FRAUD_SCENARIO |
|-----|----------------|---------------------|-------------|-------------|-----------|-----------------|--------------|----------|-------------------|
| 0   | 0              | 2023-06-01 00:00:31 | 596         | 3156        | 57.16     | 31              | 0            | 0        | 0                 |
| 1   | 1              | 2023-06-01 00:02:10 | 4961        | 3412        | 81.51     | 130             | 0            | 0        | 0                 |
| 2   | 2              | 2023-06-01 00:07:56 | 2           | 1365        | 146.00    | 476             | 0            | 0        | 0                 |
| 3   | 3              | 2023-06-01 00:09:29 | 4128        | 8737        | 64.49     | 569             | 0            | 0        | 0                 |
| 4   | 4              | 2023-06-01 00:10:34 | 927         | 9906        | 50.99     | 634             | 0            | 0        | 0                 |

</div>

## Data Understanding <a name="data-understanding"></a>

A base de dados contém 959,229 transações com cartão de crédito, durante
o período `2023-08-01` até `2023-09-08`, com as seguintes variáveis[^2]:

- `TRANSACTION_ID`: identificador para cada transação;
- `TX_DATETIME`: data e horário da transação;
- `CUSTOMER_ID`: identificador do cliente/usuário do cartão de crédito;
- `TERMINAL_ID`: identificador do terminal onde ocorreu a transação;
- `TX_AMOUNT`: valor da transação;
- `TX_FRAUD`: variável binária com valor 1 se a transação é fraudulenta;
  0 caso conctrário.

Nesse exercício, irei construir um simples sistema de detecção de
fraudes utilizando algoritmos de *machine learning*.

Uma das primeiras coisas a checar é a distribuição da variável
dependente, `TX_FRAUD`. O código a seguir demonstra:

``` python
transactions_df["TX_FRAUD"].mean()
```

    0.00793762490500183

### Feature Engineering <a name="feature-engineering"></a>

Como podemos observar, o conjunto de dados apresenta uma distribuição
bastante desbalanceada em relação aos casos de fraudes. Apenas 0.07% das
transações em nosso conjunto de dados está marcado como transação
fraudulenta. Essa característica é típico de cenário reais.

Olhando para as variáveis disponíveis, há somente quatro variáveis
disponíveis para podermos montar um sistema de detecção de fraude. Vamos
entender como essas variáveis se relacionam com o flag de fraude:

``` python
transactions_df.groupby("TX_FRAUD")["TX_AMOUNT"].mean()
```

    TX_FRAUD
    0     52.978073
    1    133.035867
    Name: TX_AMOUNT, dtype: float64

Como podemos observar, transações marcadas como fraude possuem um valor
cerca de 2.5 vezes maior do transações genuínas.

Um possível característica que pode estar relacionada a transações
suspeitas é o período em que a transação está sendo realizada. É
razoável supor que transações fraudulentas ocorram em períodos noturnos
e durante finais de semana. Para checar isso, podemos criar variáveis
que indiquem tais características nos dados:

``` python
# - flag se transação ocorre durante o fim de semana ou nao:
transactions_df["TX_DURING_WEEKEND"] = (transactions_df["TX_DATETIME"].dt.weekday >= 5).astype("int")

# - flag se transação ocorre a noite:
# definição de noite: 22 <= hour <= 6
transactions_df["TX_DURING_NIGHT"] = ((transactions_df["TX_DATETIME"].dt.hour <= 6) | (transactions_df["TX_DATETIME"].dt.hour >= 22)).astype("int")
```

``` python
transactions_df.groupby("TX_FRAUD")[["TX_DURING_WEEKEND", "TX_DURING_NIGHT"]].mean()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|          | TX_DURING_WEEKEND | TX_DURING_NIGHT |
|----------|-------------------|-----------------|
| TX_FRAUD |                   |                 |
| 0        | 0.278996          | 0.194978        |
| 1        | 0.279879          | 0.198976        |

</div>

Os resultados mostram que esses flags não parecem estar fortemente
correlacionados com a chance de uma transação ser fraude. Diferença de
pouco mais de 0.1 p.p. entre os grupos. Apesar disso, essas
características ainda podem ser úteis, uma vez que é razoável observar
padrões de fraude variam entre dias úteis e fins de semana, assim como
entre o dia e a noite.

Outro característica importante em problemas de detecção de faude
consiste no padrão de gasto de cada cliente. Cada consumidor possue um
padrão de consumo e sempre que uma transação (ou um conjunto de
transações) desvie desse padrão, cresce um alerta sobre o risco de
fraude. Para caracterizar o padrão de consumo do consumidor, podemos
criar variáveis que representem o número de transaçãoes e o gasto médio
no passado. Para isso vamos criar variáveis de descrevem a frequência e
o gasto médio de cada cliente em 3 janelas distintas de tempo: dia
anterior, semana anterior e mês anterior.

``` python
def feature_engineering_customer_spending_behaviour(customer_transactions, windows_size_in_days=[1,7,30]):    
    # ordenando de acordo com data
    customer_transactions=customer_transactions.sort_values('TX_DATETIME')
    
    # colocando data como index:
    # isso ajudará para utilizar as funcionalidades de operacoes de 'moving average':
    customer_transactions.index=customer_transactions.TX_DATETIME
    
    # loop sobre cada janela de tempo (em dias)
    for window_size in windows_size_in_days:
        # calculo da soma de transações:
        SUM_AMOUNT_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').sum()
        # Calculo do número:
        NB_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').count()    
        # calculo da média:
        AVG_AMOUNT_TX_WINDOW=SUM_AMOUNT_TX_WINDOW/NB_TX_WINDOW    
        # Save feature values
        customer_transactions['CUSTOMER_ID_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_AVG_AMOUNT_'+str(window_size)+'DAY_WINDOW']=list(AVG_AMOUNT_TX_WINDOW)
    
    # set transaction como index:
    customer_transactions.index=customer_transactions.TRANSACTION_ID

    return customer_transactions
```

``` python
# calculando as variáveis que caracterizam o padrao do cliente:
transactions_df = transactions_df.groupby("CUSTOMER_ID").apply(lambda x: feature_engineering_customer_spending_behaviour(x, windows_size_in_days=[1, 7, 30]))
transactions_df.reset_index(drop=True, inplace=True)
transactions_df.sort_values(by="TX_DATETIME", inplace=True)
transactions_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|        | TRANSACTION_ID | TX_DATETIME         | CUSTOMER_ID | TERMINAL_ID | TX_AMOUNT | TX_TIME_SECONDS | TX_TIME_DAYS | TX_FRAUD | TX_FRAUD_SCENARIO | TX_DURING_WEEKEND | TX_DURING_NIGHT | CUSTOMER_ID_NB_TX_1DAY_WINDOW | CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW | CUSTOMER_ID_NB_TX_7DAY_WINDOW | CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW | CUSTOMER_ID_NB_TX_30DAY_WINDOW | CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW |
|--------|----------------|---------------------|-------------|-------------|-----------|-----------------|--------------|----------|-------------------|-------------------|-----------------|-------------------------------|------------------------------------|-------------------------------|------------------------------------|--------------------------------|-------------------------------------|
| 112991 | 0              | 2023-06-01 00:00:31 | 596         | 3156        | 57.16     | 31              | 0            | 0        | 0                 | 0                 | 1               | 1.0                           | 57.160000                          | 1.0                           | 57.160000                          | 1.0                            | 57.160000                           |
| 951176 | 1              | 2023-06-01 00:02:10 | 4961        | 3412        | 81.51     | 130             | 0            | 0        | 0                 | 0                 | 1               | 1.0                           | 81.510000                          | 1.0                           | 81.510000                          | 1.0                            | 81.510000                           |
| 534    | 2              | 2023-06-01 00:07:56 | 2           | 1365        | 146.00    | 476             | 0            | 0        | 0                 | 0                 | 1               | 1.0                           | 146.000000                         | 1.0                           | 146.000000                         | 1.0                            | 146.000000                          |
| 789906 | 3              | 2023-06-01 00:09:29 | 4128        | 8737        | 64.49     | 569             | 0            | 0        | 0                 | 0                 | 1               | 1.0                           | 64.490000                          | 1.0                           | 64.490000                          | 1.0                            | 64.490000                           |
| 177280 | 4              | 2023-06-01 00:10:34 | 927         | 9906        | 50.99     | 634             | 0            | 0        | 0                 | 0                 | 1               | 1.0                           | 50.990000                          | 1.0                           | 50.990000                          | 1.0                            | 50.990000                           |
| ...    | ...            | ...                 | ...         | ...         | ...       | ...             | ...          | ...      | ...               | ...               | ...             | ...                           | ...                                | ...                           | ...                                | ...                            | ...                                 |
| 707910 | 959224         | 2023-09-08 23:53:37 | 3698        | 6046        | 40.14     | 8639617         | 99           | 0        | 0                 | 0                 | 1               | 3.0                           | 292.663333                         | 19.0                          | 258.636316                         | 75.0                           | 156.688533                          |
| 684872 | 959225         | 2023-09-08 23:53:39 | 3565        | 9889        | 13.42     | 8639619         | 99           | 0        | 0                 | 0                 | 1               | 1.0                           | 13.420000                          | 12.0                          | 10.672500                          | 89.0                           | 10.114719                           |
| 647282 | 959226         | 2023-09-08 23:57:39 | 3373        | 5963        | 67.22     | 8639859         | 99           | 0        | 0                 | 0                 | 1               | 2.0                           | 49.810000                          | 12.0                          | 60.322500                          | 54.0                           | 76.486111                           |
| 35655  | 959227         | 2023-09-08 23:58:24 | 187         | 9772        | 47.29     | 8639904         | 99           | 0        | 0                 | 0                 | 1               | 3.0                           | 31.593333                          | 14.0                          | 30.558571                          | 64.0                           | 33.645156                           |
| 657060 | 959228         | 2023-09-08 23:59:24 | 3422        | 9951        | 35.60     | 8639964         | 99           | 0        | 0                 | 0                 | 1               | 3.0                           | 37.886667                          | 30.0                          | 42.133000                          | 109.0                          | 40.277523                           |

<p>959229 rows × 17 columns</p>
</div>

Vamos checar como essas variáveis estão relacionadas ao risco de fraude:

``` python
transactions_df.groupby("TX_FRAUD")[['CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
                'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
                'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW']].mean()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|          | CUSTOMER_ID_NB_TX_1DAY_WINDOW | CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW | CUSTOMER_ID_NB_TX_7DAY_WINDOW | CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW | CUSTOMER_ID_NB_TX_30DAY_WINDOW | CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW |
|----------|-------------------------------|------------------------------------|-------------------------------|------------------------------------|--------------------------------|-------------------------------------|
| TX_FRAUD |                               |                                    |                               |                                    |                                |                                     |
| 0        | 3.555907                      | 53.257462                          | 18.354692                     | 53.420015                          | 66.491772                      | 53.487858                           |
| 1        | 3.546231                      | 99.325549                          | 18.820068                     | 77.383708                          | 71.865117                      | 64.302261                           |

</div>

Como podemos ver, as variáveis que caracterizam o padrão de gasto dos
cliente apresentam uma relação significativa.

Podemos também extrair mais um conjunto de variáveis que caracterizem os
terminais de pagamento. Utilizando a mesma ideia de utilizada
anteriormente, o objetivo é calcular um escore de risco que avalia a
vulnerabilidade de um determinado terminal a transações fraudulentas.
Esse risco será calculado como a média das fraudes registradas no
terminal ao longo de três intervalos de tempo diferentes.

Contudo, diferentemente das variáveis criadas baseadas no comportamento
de gasto do cliente, as janelas de tempo para os terminais não estarão
diretamente antes de uma transação específica. Em vez disso, elas serão
ajustadas para trás por um período de *delay*. O motivo disso é que em
um cenário real, as transações fraudulentas só são descobertas após uma
investigação ou reclamação do cliente. Essa descoberta leva um certo
tempo para acontecer. Por isso, as transações fraudelentas que são
usadas para calcular o score de risco, só estarão disponíveis após esse
período de *delay*. Então, para um período de delay *d* e uma janela de
7 dias, então a janela seria `[d-7; d]`. Para uma aproximação inicial,
esse período de *delay* será definido como uma semana (d = 7).

``` python
# definindo uma função auxiliar para calcular as variáveis de risco em diferentes janelas:
def feature_engineering_terminal_risk(terminal_transactions,
                                  delay_period=7,
                                  windows_size_in_days=[1,7,30],
                                  feature="TERMINAL_ID"):   
    # ordering por data
    terminal_transactions=terminal_transactions.sort_values('TX_DATETIME')

    # set index to date:    
    terminal_transactions.index=terminal_transactions.TX_DATETIME

    # calculando o número de fraudes no periodo de delay
    NB_FRAUD_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').sum()
    # calculando o número de total de transacoes no periodo de delay
    NB_TX_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').count()

    # calculando o número no periodo de (delay + windows_size)
    for window_size in windows_size_in_days:
        # calculando o número de fraudes
        NB_FRAUD_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').sum()
        # calculando o número de total de transacoes
        NB_TX_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').count()
        # difenca entre (delay+windows_size) - delay
        NB_FRAUD_WINDOW=NB_FRAUD_DELAY_WINDOW-NB_FRAUD_DELAY
        NB_TX_WINDOW=NB_TX_DELAY_WINDOW-NB_TX_DELAY
        # calculando o risco:
        RISK_WINDOW=NB_FRAUD_WINDOW/NB_TX_WINDOW
        
        terminal_transactions[feature+'_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
        terminal_transactions[feature+'_RISK_'+str(window_size)+'DAY_WINDOW']=list(RISK_WINDOW)
        
    terminal_transactions.index=terminal_transactions.TRANSACTION_ID
    
    # Replace NA values with 0 :    
    terminal_transactions.fillna(0,inplace=True)
    
    return terminal_transactions
```

``` python
transactions_df=transactions_df.groupby('TERMINAL_ID').apply(lambda x: feature_engineering_terminal_risk(x, delay_period=7, windows_size_in_days=[1,7,30], feature="TERMINAL_ID"))
transactions_df.sort_values('TX_DATETIME', inplace=True)
transactions_df.reset_index(drop=True, inplace=True)
transactions_df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|        | TRANSACTION_ID | TX_DATETIME         | CUSTOMER_ID | TERMINAL_ID | TX_AMOUNT | TX_TIME_SECONDS | TX_TIME_DAYS | TX_FRAUD | TX_FRAUD_SCENARIO | TX_DURING_WEEKEND | ... | CUSTOMER_ID_NB_TX_7DAY_WINDOW | CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW | CUSTOMER_ID_NB_TX_30DAY_WINDOW | CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW | TERMINAL_ID_NB_TX_1DAY_WINDOW | TERMINAL_ID_RISK_1DAY_WINDOW | TERMINAL_ID_NB_TX_7DAY_WINDOW | TERMINAL_ID_RISK_7DAY_WINDOW | TERMINAL_ID_NB_TX_30DAY_WINDOW | TERMINAL_ID_RISK_30DAY_WINDOW |
|--------|----------------|---------------------|-------------|-------------|-----------|-----------------|--------------|----------|-------------------|-------------------|-----|-------------------------------|------------------------------------|--------------------------------|-------------------------------------|-------------------------------|------------------------------|-------------------------------|------------------------------|--------------------------------|-------------------------------|
| 0      | 0              | 2023-06-01 00:00:31 | 596         | 3156        | 57.16     | 31              | 0            | 0        | 0                 | 0                 | ... | 1.0                           | 57.160000                          | 1.0                            | 57.160000                           | 0.0                           | 0.0                          | 0.0                           | 0.00                         | 0.0                            | 0.00                          |
| 1      | 1              | 2023-06-01 00:02:10 | 4961        | 3412        | 81.51     | 130             | 0            | 0        | 0                 | 0                 | ... | 1.0                           | 81.510000                          | 1.0                            | 81.510000                           | 0.0                           | 0.0                          | 0.0                           | 0.00                         | 0.0                            | 0.00                          |
| 2      | 2              | 2023-06-01 00:07:56 | 2           | 1365        | 146.00    | 476             | 0            | 0        | 0                 | 0                 | ... | 1.0                           | 146.000000                         | 1.0                            | 146.000000                          | 0.0                           | 0.0                          | 0.0                           | 0.00                         | 0.0                            | 0.00                          |
| 3      | 3              | 2023-06-01 00:09:29 | 4128        | 8737        | 64.49     | 569             | 0            | 0        | 0                 | 0                 | ... | 1.0                           | 64.490000                          | 1.0                            | 64.490000                           | 0.0                           | 0.0                          | 0.0                           | 0.00                         | 0.0                            | 0.00                          |
| 4      | 4              | 2023-06-01 00:10:34 | 927         | 9906        | 50.99     | 634             | 0            | 0        | 0                 | 0                 | ... | 1.0                           | 50.990000                          | 1.0                            | 50.990000                           | 0.0                           | 0.0                          | 0.0                           | 0.00                         | 0.0                            | 0.00                          |
| ...    | ...            | ...                 | ...         | ...         | ...       | ...             | ...          | ...      | ...               | ...               | ... | ...                           | ...                                | ...                            | ...                                 | ...                           | ...                          | ...                           | ...                          | ...                            | ...                           |
| 959224 | 959224         | 2023-09-08 23:53:37 | 3698        | 6046        | 40.14     | 8639617         | 99           | 0        | 0                 | 0                 | ... | 19.0                          | 258.636316                         | 75.0                           | 156.688533                          | 1.0                           | 0.0                          | 7.0                           | 0.00                         | 26.0                           | 0.00                          |
| 959225 | 959225         | 2023-09-08 23:53:39 | 3565        | 9889        | 13.42     | 8639619         | 99           | 0        | 0                 | 0                 | ... | 12.0                          | 10.672500                          | 89.0                           | 10.114719                           | 0.0                           | 0.0                          | 7.0                           | 0.00                         | 23.0                           | 0.00                          |
| 959226 | 959226         | 2023-09-08 23:57:39 | 3373        | 5963        | 67.22     | 8639859         | 99           | 0        | 0                 | 0                 | ... | 12.0                          | 60.322500                          | 54.0                           | 76.486111                           | 2.0                           | 0.0                          | 3.0                           | 0.00                         | 28.0                           | 0.00                          |
| 959227 | 959227         | 2023-09-08 23:58:24 | 187         | 9772        | 47.29     | 8639904         | 99           | 0        | 0                 | 0                 | ... | 14.0                          | 30.558571                          | 64.0                           | 33.645156                           | 1.0                           | 0.0                          | 4.0                           | 0.25                         | 25.0                           | 0.04                          |
| 959228 | 959228         | 2023-09-08 23:59:24 | 3422        | 9951        | 35.60     | 8639964         | 99           | 0        | 0                 | 0                 | ... | 30.0                          | 42.133000                          | 109.0                          | 40.277523                           | 1.0                           | 0.0                          | 10.0                          | 0.00                         | 42.0                           | 0.00                          |

<p>959229 rows × 23 columns</p>
</div>

Vamos checar como essas variáveis estão relacionadas ao risco de fraude:

``` python
transactions_df.groupby("TX_FRAUD")[[ 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                'TERMINAL_ID_RISK_1DAY_WINDOW',
                'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                'TERMINAL_ID_RISK_7DAY_WINDOW',
                'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                'TERMINAL_ID_RISK_30DAY_WINDOW']].mean()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|          | TERMINAL_ID_NB_TX_1DAY_WINDOW | TERMINAL_ID_RISK_1DAY_WINDOW | TERMINAL_ID_NB_TX_7DAY_WINDOW | TERMINAL_ID_RISK_7DAY_WINDOW | TERMINAL_ID_NB_TX_30DAY_WINDOW | TERMINAL_ID_RISK_30DAY_WINDOW |
|----------|-------------------------------|------------------------------|-------------------------------|------------------------------|--------------------------------|-------------------------------|
| TX_FRAUD |                               |                              |                               |                              |                                |                               |
| 0        | 0.926344                      | 0.002493                     | 6.269935                      | 0.004190                     | 23.403704                      | 0.005037                      |
| 1        | 0.973076                      | 0.264802                     | 6.765432                      | 0.365439                     | 25.802732                      | 0.175135                      |

</div>

Como podemos ver, transações fraudelentas apresentam uma relação com as
variáveis de risco de cada terminal.

## Data Preparation <a name="data-preparation"></a>

Agora que temos a base de dados de transação, verificamos a distribuicao
de nossa variável de interesse e extraímos novas características a
partir das informações existentes, podemos seguir para a fase de
preparar os dados para a estimação de nosso sistema de detecção de
fraude.

Primeira coisa que precisamos definir é nossa base de treinamento e base
de test. É importante tomar cuidado no aspecto temporal do problema de
detecção de fraude. Os dados apresentam um aspecto de *time-dependent*
entre as variáveis e é preciso levar isso em consideração na estratégia
de treinamento. É comum resoluções de problemas como *time-dependent*
onde não esse tipo de consideração. Um dos problemas que isso pode
acarretar é o *data leakage* em nosso processo de treinamento do
algoritmo de *machine learning*.

Para evitar esse problema, eu utilizarei transações durante o período de
`2023-08-11` até `2023-08-17` como base de treinamento e transações
entre `2023-08-25` e `2023-08-31` como base de test. Ou seja, escolhemos
1 semana de transações para a base de treinamento e 1 semana de
transações para o teste.

Vale ressaltar um aspecto importante na estratégia escolhida: a base de
teste ocorre uma semana após a última transação do base de treinamento.
Esse período que separa a base de treinamento e a base de teste é
chamado de *delay*. Esse período representa o fato de que, em um cenário
real detecção de fraudes, a classificação de uma transação (fraudulenta
ou não) só é conhecida após uma reclamação do cliente ou com base nos
resultados de uma investigação de fraude.

Outro ponto importante intríseco a cenários de detecção de fraude é a
exclusão de cartões fraudulentos da base de treinamento na base de
teste. Dado o aspecto temporal da separação da base de treinamento e
teste, cartões que já foram detectatos com transações fraudulentas no
*training set* serão excluídos na base de teste. Também, cartões que
vierem a ser descobertos como fraudulentos ao longo do período de delay
serão excluídos da base de test.

Para realizar essa separação, foi criado uma função auxiliar chamada
`get_train_est_set`:

``` python
def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7,delta_delay=7,delta_test=7):
    
    # training set data
    train_df = transactions_df[(transactions_df.TX_DATETIME>=start_date_training) &
                               (transactions_df.TX_DATETIME<start_date_training+datetime.timedelta(days=delta_train))]
    
    # test set data
    test_df = []
    
    # Note: cartoes fraudulentos da base de treino serão removidos da base de test. 
    # também, cartões que vierem a ser descobertos como fraudulentos ao longo
    # do período de delay, tbm serão excluidos da base de test.
    
    # pegando os cartoes fraudulentos do training set:
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD==1].CUSTOMER_ID)
    
    # pegando a data relativa:
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()
    
    # Then, for each day of the test set
    for day in range(delta_test):
    
        # Get test data for that day
        test_df_day = transactions_df[transactions_df.TX_TIME_DAYS==start_tx_time_days_training+
                                                                    delta_train+delta_delay+
                                                                    day]
        
        # Os cartões comprometidos do dia de teste, subtraindo o período de esperadelay, são adicionados ao grupo de clientes fraudulentos conhecidos.
        test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS==start_tx_time_days_training+
                                                                                delta_train+
                                                                                day-1]
        
        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD==1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)
        
        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]
        
        test_df.append(test_df_day)
        
    test_df = pd.concat(test_df)
    
    # Sort data sets by ascending order of transaction ID
    train_df=train_df.sort_values('TRANSACTION_ID')
    test_df=test_df.sort_values('TRANSACTION_ID')
    
    return (train_df, test_df)
```

``` python
import datetime
# setting start date:
start_date = datetime.datetime.strptime("2023-08-11", "%Y-%m-%d")
#print("Start date: {start_date}")

# separating train and test set sequentially with a delay period:
train_df, test_df=get_train_test_set(
    transactions_df,
    start_date_training=start_date,
    delta_train=7,
    delta_delay=7,
    delta_test=7
)
```

``` python
train_df.shape[0]
```

    67180

``` python
test_df.shape[0]
```

    58330

O training set consiste em 67,180 observações e o test set com 58,330
observações.

``` python
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
```

## Modelling <a name="modelling"></a>

A detecção de fraudes é frequentemente tratada como um problema de
classificação binária: um sistema de detecção de fraudes recebe
transações e tem como objetivo preverse essas transações são
provavelmente legítimas ou fraudulentas.

Uma característica inerente aos problemas de detecção de fraude é o
desequilíbrio significativo na distribuição das classes, em que a
proporção de casos de fraude é consideravelmente menor. Ao realizar a
modelagem, é crucial considerar esse aspecto.

É bastante comum em problemas de classificação binária o uso da acurácia
como métrica de desempenho dos algoritmos. No entanto, apesar de sua
interpretação simples, essa métrica não é apropriada para problemas em
que há um desequilíbrio significativo na distribuição da variável
dependente. Outras medidas são mais apropriadas para o tipo de problema
em questão.

O *F1-score* é uma métrica especialmente útil para lidar com casos de
desbalanceamento em problemas de classificação, como a detecção de
fraudes. Isso se deve ao fato de que o F1-score considera tanto a
precisão (precision) quanto o recall, o que o torna uma medida
equilibrada e sensível às situações em que há uma proporção desigual
entre as classes.

O *G-mean*, ou média geométrica, é outra métrica que se destaca quando
se trata de avaliar o desempenho de modelos em cenários de classes
desbalanceadas. A média geométrica é definida como $\sqrt{TNP*TNR}$,
onde `TNR` e `TPR` são taxas de verdadeiros positivos (casos
corretamente identificados como a classe minoritária) e verdadeiros
negativos (casos corretamente identificados como a classe majoritária).

A *AUC-ROC* mede a capacidade do modelo de classificar corretamente
instâncias positivas em relação às negativas em diveros thresholds de
classificação. Isso é especialmente útil em problemas desbalanceados,
onde a classe majoritária (transações legítimas) pode dominar a métrica
de acurácia.

Por fim, também será utilizado a *média de precisão (Average
Precision)*. Essa métrica é a area sobre a curva Precision-Recall e é
métrica valiosa para avaliar o desempenho de modelos em cenários de
classes desbalanceadas, como a detecção de fraudes.

Em primeiro lugar, vamos montar um *dummy classifier* e utilizá-lo como
*baseline* para os nossos modelos subsequentes. Esse classifier
utilizará como previsão a média da variável de interesse como previsão
se a transação será suspeita ou não.

``` python
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score
from imblearn.metrics import geometric_mean_score

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
     "auc": [auc_dummy],
     "average_precision": [avg_prec_dummy]}
)

# creating a empty list:
res = []

# append first result:
res.append(dummy_res)
```

A partir de agora vamos estimar seis algoritmos de *machine learning* e
compará-los entre si e como o *dummy classifier* para checarmos como
vamos melhorando nossas estimações. Especificamente iremos estimar os
seguintes algoritmos:

- Logistic Regression;
- *K-nearest neighbors*
- Decision Tree;
- Bagging of Decision trees;
- Random Forest;
- Gradient Boosting;

Todos os algoritmos serão estimados com configuração *default*.

Será utilizado um processo de imputação e de *scaling* nas variáveis
explicativas. Essas etapas serão agrupadas em um `Pipeline`. O uso
Pipelines permite estruturar o fluxo de trabalho de modelagem de maneira
mais organizada e legível.

``` python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
         "auc": [auc],
         "average_precision": [avg_prec]
         }
    )    
    res.append(eval)
```

``` python
# concatenate results
res = pd.concat(res)
res = res.sort_values("average_precision", ascending=True).reset_index(drop=True)
print(res)
```

                 clf       acc  f1_score     gmean       auc  average_precision
    0          dummy  0.992628  0.000000  0.000000  0.500000           0.007372
    1           tree  0.994360  0.622273  0.792706  0.813648           0.390012
    2            knn  0.995903  0.631741  0.690383  0.792873           0.532729
    3  GradientBoost  0.996434  0.725594  0.799343  0.866963           0.595534
    4       logistic  0.995868  0.637594  0.702017  0.875138           0.627553
    5        bagging  0.997000  0.754558  0.790841  0.851214           0.674920
    6             rf  0.996777  0.729885  0.768489  0.885207           0.697627

A tabela anterior mostra o performance dos algoritmos sobre a base de
teste. Como podemos observar, o *dummy classifier* possui o pior
desempenho entre todos os classificadores, enquanto o *random forest*
possui o melhor desempenho em termos de *average precision*.

É de suma importância salientar que a acurácia não é adequada como
métrica em cenários com distribuição desequilibrada da variável Y. O
dummy classifier atingiu uma taxa de acurácia de 99,26%, um resultado
que, em outra situação, poderia ser considerado excepcional. Entretanto,
ao examinar as demais métricas, fica claro que esse mesmo classificador
exibe um desempenho notavelmente inferior quando comparado a outros
modelos.

### Improvement

Uma limitação da última estratégia de estimação (train-test split) é que
possuímos apenas uma única medida de desempenho do respectivo
algoritmo.Apesar de ser uma estratégia valiosa, é fundamental contar com
métodos que nos permitam quantificar a incerteza em torno da performance
do algoritmo.

Uma forma de obtermos estimatas mais precisas é realizarmos uma
estratégia de *cross validation*(CV). Contudo, o problema de detecção de
fraude possui um aspecto temporal. Então será preciso adptar uma
estratégia de CV a esse aspecto temporal.

Uma forma de fazer isso é adaptar a estratégia de
[timeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
às características do problema de detecção de fraud (delay period). Em
outras palavras, iremos aplicar a mesma função de separação de
train-test set (`get_train_test_data`) tal que as bases de treino e test
sempre são deslocadas por um bloco de tempo.

A função `sequential_train_test_split` irá realizar esse separação. Essa
função irá receber a base de dados (*transactions_df*), a data que irá
começar o treinamento (start_date_training), o periodo para a base de
treino (delta_train), base de validacao (delta_val) e o delay entre
essas bases (delta_delay). Vale destacar que essa função irá retornar um
*tuple* com os índices das observações que irão compor cada base. Isso é
importante porque será possível integrar essa função a biblioteca do
`sklearn` e dessa forma utilizar toda sua funcionalidade.

``` python
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
```

Vamos a estratégia de estimacao como utilizando 5 folders, e 1 semana de
dados para treinament, 1 semanada de dados para o base de validacao e 1
semanda de delay period. Irei manter a base de test para que possamos
testar se nossa estimativa de performance irá bater como a performance
de na base de teste.

``` python
# determing the folders:
n_folds=5
delta_train=7
delta_delay=7
delta_val=7

# defining start date in the sequential cv:
start_date_training_seq = start_date+datetime.timedelta(days=-(delta_delay+delta_val))

# creting the indexes for each sequential cv:
sequential_split_indices = sequential_train_test_split(
    transactions_df,
    start_date_training = start_date_training_seq,
    n_folds=n_folds,
    delta_train=delta_train,
    delta_delay=delta_delay,
    delta_val=delta_val
)
```

Vamos checar as datas de cada base de treinamento e validacao nessa
estratégia:

``` python
# checking dates of training:
for i in list(range(n_folds))[::-1]:   
    train_data1 = transactions_df.loc[sequential_split_indices[i][0]]["TX_DATETIME"].min()
    train_data2 = transactions_df.loc[sequential_split_indices[i][0]]["TX_DATETIME"].max()

    val_data1 = transactions_df.loc[sequential_split_indices[i][1]]["TX_DATETIME"].min()
    val_data2 = transactions_df.loc[sequential_split_indices[i][1]]["TX_DATETIME"].max()

    print("train range: "+datetime.datetime.strftime(train_data1, "%Y-%m-%d")+" - "+datetime.datetime.strftime(train_data2, "%Y-%m-%d"))
    print("val range: "+datetime.datetime.strftime(val_data1, "%Y-%m-%d")+" - "+datetime.datetime.strftime(val_data2, "%Y-%m-%d"))
    print("\n")
```

    train range: 2023-06-30 - 2023-07-06
    val range: 2023-07-14 - 2023-07-20


    train range: 2023-07-07 - 2023-07-13
    val range: 2023-07-21 - 2023-07-27


    train range: 2023-07-14 - 2023-07-20
    val range: 2023-07-28 - 2023-08-03


    train range: 2023-07-21 - 2023-07-27
    val range: 2023-08-04 - 2023-08-10


    train range: 2023-07-28 - 2023-08-03
    val range: 2023-08-11 - 2023-08-17

Cada base de validação está separada por 1 semana da base de
treinamento. Como a base de teste possui transações entre `2023-08-25` e
`2023-08-31`, então todas as bases estão de acordo.

Com a estratégia de CV definida, agora é possível extender a estimação
dos algoritmos e testar algumas configurações de hyperparametros para
cada algoritmo, de modo otimizar a performance do sistema de detecção de
fraude.

Para isso, irei definir um conjunto de algoritmos e hyperparametros e
testá-los utilizando a estratégia de `GridSearchCV`. Os algoritmos serão
avaliados utilizando a métrica *average precision*.

``` python
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
    verbose=0
)

# fitting:
grid.fit(transactions_df[input_features], transactions_df[output_feature])
```

Como especificado em `GridSearchCV`, eu setei `refit=False` para que não
fosse reestimado a melhor configuração de modelo sobre todo o conjunto
de dados. Com isso, não é retornado o `.best_params_`, `.best_score_`,
`.best_estimator_`.

Para obter a performance dos modelos, vamos extrair o atributo
`.cv_results_`, converter para DataFrame. A seguir é calculado um
intervalo de confiança para cada estimação de performance. Dessa forma,
podemos ter uma medida de incerteza acerca da previsão da performance
dos nossos algoritmos sobre a base de test.

``` python
# getting the results:
results = pd.DataFrame(grid.cv_results_)
# assuming normal distribution:
results["lower_bd"] = results["mean_test_score"]-2*results["std_test_score"]
results["upper_bd"] = results["mean_test_score"]+2*results["std_test_score"]
# sorting:
results = results.sort_values("rank_test_score", ascending=True).reset_index(drop=True)
```

Para obter o modelo final, eu procedo da seguinte maneira: escolher o
modelo que está a 1 desvio padrão da melhor performance. Essa escolha
ajuda a reduzir ainda mais a dependencia do modelo sobre as bases de
validação e dessa forma diminuir o *overfitting*. Para isso é criado uma
função auxiliar (`get_bestModel`) que recebe esse DataFrame com as
performances e retorna um tupple com o indice do modelo escolhido e a
configuracao:

``` python
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
```

Aplicando a função anterior e obtendo o melhor modelo (segundo regra de
1 standard rule):

``` python
# get the best configuration:
idx, best_clf_config = get_bestModel(results)
results.loc[idx]
```

    mean_fit_time                                                       2.178353
    std_fit_time                                                        0.190987
    mean_score_time                                                      0.06542
    std_score_time                                                      0.001052
    param_clf                  LogisticRegression(C=0.5, max_iter=2000, solve...
    param_clf__C                                                             0.5
    param_clf__penalty                                                        l2
    param_clf__estimator                                                     NaN
    param_clf__max_features                                                  NaN
    param_clf__n_estimators                                                  NaN
    params                     {'clf': LogisticRegression(C=0.5, max_iter=200...
    split0_test_score                                                   0.652561
    split1_test_score                                                   0.601248
    split2_test_score                                                   0.612506
    split3_test_score                                                   0.610476
    split4_test_score                                                   0.584905
    mean_test_score                                                     0.612339
    std_test_score                                                      0.022347
    rank_test_score                                                            8
    lower_bd                                                            0.567645
    upper_bd                                                            0.657033
    Name: 7, dtype: object

Agora podemos estimar o modelo final utilizando a base de treino e
testar esse modelo sobre a base de teste:

``` python
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
```

    CV prediction: 0.61
    CV confidence interval: [0.57; 0.66]
    Test score: 0.63

Como é possível observar, a estratégia de CV foi uma boa alternativa
para estimar algoritmos em suas diversas configurações e também para
obter uma estimativa confiável acerca da performance futura do
algoritmo.

## Evaluation <a name="evaluation"></a>

Após desenvolver um modelo com um bom nível de qualidade (na perspectiva
técnica), é fundamental conduzir uma avaliação abrangente para assegurar
que o modelosistema atinge efetivamente os objetivos de negócio. Um
ponto crucial é determinar se há alguma questão de relevância
empresarial que não tenha recebido a devida consideração ao longo do
processo.

Quando se trata de um sistema de detecção de fraudes em transações com
cartão de crédito, é fundamental avaliar sua relevância operacional para
o cenário empresarial. O objetivo central de tal sistema é emitir
alertas sobre transações suspeitas, as quais são então submetidas à
análise por parte dos investigadores. Esse procedimento é consome tempo.
Em consequência, a quantidade de alertas que pode ser verificada durante
um período específico é limitada.

Uma forma de medirmos o benefício operacional é atráves da mensuração de
quanto tal sistema ajuda investigadores. Suponha que é possível analisar
*k* transações suspeitas em um dia. *Precision top-k* é uma medida de
performance de que maximiza a precisão de sua estimação sobre o número
*k* de alertas que um investigador consegue analisar.

*Precision top-k* é calculada da seguinte forma. Para um determinado dia
*d*:

1.  Ordene as transações suspeitas de acordo com a probabilidade;
2.  Selecione as top *k* transações mais suspeitas;
3.  Compute a precisão sobre essas *k* transações.

``` python
def precision_top_k_day(df_day, top_k=100):
    # sorting by the highest chance of fraud:
    df_day = df_day.sort_values(by="predictions", ascending=False)
    # Get the top k most suspicious transactions
    df_day_top_k=df_day.head(top_k)
    # Compute precision top k
    precision_top_k = df_day_top_k["TX_FRAUD"].mean()
    return precision_top_k
```

Podemos aplicar essa métrica ao nosso modelo e calcular a performance
operacional de para cada dia da base de teste:

``` python
test_df["predictions"] = y_pred[:, 1]
prec_top_k_test = []
for day in test_df["TX_TIME_DAYS"].unique():
    prec_top_k_test.append(precision_top_k_day(test_df.query("TX_TIME_DAYS == @day")))

# media de precision top100
np.mean(prec_top_k_test)
```

    0.4042857142857143

O nosso algoritmos é capaz de identificar uma média diária de 40% das
100 \*\*flagged\* como suspeitas.

Uma vez que tenhamos uma forma de medir a performance operacional do
sistema, podemos utilizar essa métrica como medida para otimizar os
algoritmos.

Vamos repetir o processo de estimação. Dessa vez utilizando a métrica a
média *precision-top-k* de cada dia. Para isso, precisamos criar uma
função que será utilizada junto com o `sklearn`. Para isso usaremos o
`sklearn.metrics.make_scorer`:

``` python
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
```

Uma vez criado a função para ser utilizada junto a biblioteca do
`sklearn`, podemos repetir o processo de estimação com a média
*precision-top-k*:

``` python
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
```

``` python
# predicting on test set:
y_pred2 = mdl2.predict_proba(X_test)

# daily average precision top_100 score:
avg_prec_top_100 = daily_avg_precision_top_k(y_test, y_pred2[:, 1], top_k=100, transactions_df=transactions_df)

# average precision score:
avg_prec_test = average_precision_score(y_test, y_pred2[:, 1])

print(f"CV prediction: {np.round(results2.loc[idx2, 'mean_test_score'], 2)}")
print(f"CV confidence interval: [{np.round(results2.loc[idx2, 'lower_bd'], 2)}; {np.round(results2.loc[idx2, 'upper_bd'], 2)}]")
print(f"Test score: {np.round(avg_prec_top_100, 2)}")
```

    CV prediction: 0.38
    CV confidence interval: [0.36; 0.4]
    Test score: 0.4

Assim como anterior, a estratégia de CV foi uma boa alternativa para
estimar algoritmos em suas diversas configurações e também para obter
uma estimativa confiável acerca da performance futura do algoritmo.

## Deployment <a name="deployment"></a>

A criação de um sistema de detecção de fraude não é o fim de um projeto.
Um vez que o sistema está estimado e os objetivos comerciais estão
alinhados, chega a hora de disponibilizar esse sistema para que ele seja
integrado em diferentes plataformas, aplicativos web, entre outros. A
forma mais comum é tornar o sistema disponível é em formato de REST API.

Primeira coisa a ser feita é salvar o modelo estimado para que ele seja
reutilizado:

``` python
# getting the terminal and customer dict:
# save final model:
from joblib import dump
dump(mdl2, "output/model.joblib")
```

Algumas considerações precisam ser feitas. Por exemplo, ao desenhar um
API, o desenvolvedor precisa imaginar como será o dado que será
utilizado como para o modelo gere previsões.

No caso de presente problema, é provável que dados de transações de
cartão de crédito contenha apenas as seguintes variáves:

- `TRANSACAO_ID`
- `TX_DATETIME`
- `CUSTOMER_ID`
- `TERMINAL_ID`
- `TX_AMOUNT`

Um primeir obstáculo que surge é como *encode* as variáveis de cliente e
terminal. Como vimos ao longo da estimação, essas variáveis foram
*encodadas* com valores históricos calculados. Em um modelo em produção,
não será possível fazer isso a extração dessas informações a partir de
dados históricos. Utilizaremos como dicionário, o último valor de cada
variável calculada para cada customer e terminal. Isso irá ser utilizado
como um dicionário:

``` python
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
```

Uma vez que tenhamos os dicionarios das variáveis categoricas e o modelo
salvos, podemos escrever nosso API usando a biblioteca `Flask`:

``` python
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
```

A partir de um terminal, podemos levantar a API com o seguinte comando:

    pipenv run python api.py

### Endpoints

- **URL**: `/fraud_alert`
- **Method**: `POST`
- **Descrição**: Esse endpoint irá calcular a probabilidade de uma
  transação ser considerada fraudulenta.
- **Input**: `JSON` com `TRANSACAO_ID`, `TX_DATETIME`, `CUSTOMER_ID`,
  `TERMINAL_ID`, `TX_AMOUNT`. Exemplo:

``` json
[{"TRANSACTION_ID":892153,"TX_DATETIME":1693612359000,"CUSTOMER_ID":3465,"TERMINAL_ID":2982,"TX_AMOUNT":101.76},{"TRANSACTION_ID":892154,"TX_DATETIME":1693612692000,"CUSTOMER_ID":3514,"TERMINAL_ID":9078,"TX_AMOUNT":99.75},{"TRANSACTION_ID":892155,"TX_DATETIME":1693612758000,"CUSTOMER_ID":1839,"TERMINAL_ID":4609,"TX_AMOUNT":15.95},{"TRANSACTION_ID":892156,"TX_DATETIME":1693612787000,"CUSTOMER_ID":2909,"TERMINAL_ID":6355,"TX_AMOUNT":85.64}]
```

- **Resposta**: `JSON` com o seguinte formato:

``` json
[
  {
    "TRANSACTION_ID": 892154,
    "prob_fraud": 0.0007436351351168843
  },
  {
    "TRANSACTION_ID": 892155,
    "prob_fraud": 0.0019805953668296816
  },
  {
    "TRANSACTION_ID": 892156,
    "prob_fraud": 0.00022534077315471119
  }
]
```

Há um conjunto de transações referente ao dia `2023-09-01` no endereço
`data/transactions_20230901.json`. Para realizar um `request` com esses
dados, podemos proceder da seguinte forma:

``` sh
curl -XPOST -H "Content-Type: application/json" -d @data/transactions_20230901.json http://127.0.0.1:5000/fraud_alert
```

[^1]: Para a replicação dos dados simulados, há um script chamado
    `simulate_data.py` com os passos necessários. O script `utils.py`
    contem as funções auxiliares necessárias para a execução da
    simulação. Para mais detalhes teóricos, acesse esse
    [link.](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html.)

[^2]: As variáveis `TX_TIME_SECONDS`, `TX_TIME_DAYS` são variáveis
    extraídas a partir da variável de data e `TX_FRAUD_SCENARIO` é uma
    variável *by-product* do processo de simulação. Elas não serão
    importantes para o exercício.
