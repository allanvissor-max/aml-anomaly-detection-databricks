# AML Machine Learning Pipeline with Databricks and Qlik Sense

## Overview

This project is an end-to-end proof of concept for detecting unusual transaction behaviour in an Anti-Money Laundering (AML) context.

The solution ingests JSON transaction data into Databricks, processes it through Bronze and Silver layers, creates customer behaviour features, applies an Isolation Forest model, writes the highest-risk transactions into a Gold analyst queue, and visualizes the results in Qlik Sense.

```text
JSON transactions
        ↓
Databricks Lakeflow Bronze
        ↓
Silver cleaning and standardization
        ↓
Customer behaviour feature engineering
        ↓
Isolation Forest anomaly detection
        ↓
ML risk score and outlier flag
        ↓
Gold analyst investigation queue
        ↓
Qlik Sense dashboard
```

The project was built as a practical portfolio solution to demonstrate how data engineering, machine learning, AML analytics, and business intelligence can be combined in one workflow.

---

## Business Objective

AML analysts cannot manually review every transaction. A detection solution must therefore identify unusual behaviour and prioritize cases for investigation.

This project aims to:

- detect statistically unusual transactions;
- compare each transaction with the customer’s previous behaviour;
- rank detected outliers by anomaly score;
- create a structured investigation queue;
- provide analysts with an interactive dashboard;
- demonstrate the trade-off between detection coverage and false positives.

The model identifies **anomalies**, not confirmed money laundering or fraud. Every flagged transaction still requires analyst review.

---

## Technology Stack

- **Databricks**
- **Lakeflow Spark Declarative Pipelines**
- **Apache Spark / PySpark**
- **Delta tables**
- **scikit-learn**
- **Isolation Forest**
- **Pandas**
- **Qlik Sense**
- **Git / GitHub**

---

## Dataset

The synthetic source dataset contains exactly **1,000 JSON transactions**.

Final scenario distribution:

| Scenario | Transactions |
|---|---:|
| NORMAL | 970 |
| HIGH_AMOUNT | 10 |
| HIGH_VELOCITY | 8 |
| NEW_COUNTRY | 8 |
| COMBINED | 4 |
| **Total** | **1,000** |

The generated anomaly scenarios represent **3% of the dataset**.

Each JSON record contains:

```json
{
  "transaction_id": "unique transaction identifier",
  "customer_id": "CUST_0001",
  "transaction_timestamp": "2026-07-01T10:15:00+00:00",
  "amount_eur": 125.40,
  "currency": "EUR",
  "country": "EE",
  "transaction_type": "CARD_PAYMENT",
  "generated_scenario": "NORMAL"
}
```

`generated_scenario` is retained only for offline model evaluation. It is never used as an Isolation Forest feature and is excluded from the final analyst queue.

---

## Data Pipeline

### Bronze Layer

The Bronze layer uses Databricks Auto Loader to ingest JSON files from a Unity Catalog volume.

Main responsibilities:

- preserve the original source fields;
- load files incrementally;
- add ingestion timestamp;
- record the source file path.

Output table:

```text
workspace.aml_ml_3pct.bronze_transactions
```

### Silver Layer

The Silver layer applies technical cleaning and standardization.

Main transformations:

- cast transaction timestamps;
- cast amounts to numeric values;
- standardize country, currency, and transaction type values;
- remove records with missing technical identifiers;
- remove invalid amounts;
- remove duplicate transaction IDs.

Output table:

```text
workspace.aml_ml_3pct.silver_transactions
```

These are technical data-quality checks, not AML business rules.

### Customer Behaviour Feature Layer

The feature table contains one row per transaction and calculates customer behaviour using only information available up to that transaction timestamp.

Output table:

```text
workspace.aml_ml_3pct.customer_behavior_features
```

---

## Machine Learning Features

The Isolation Forest model uses five features.

| Feature | Description |
|---|---|
| `amount_deviation_ratio` | Current amount divided by the customer’s historical median transaction amount |
| `transaction_count_1h` | Number of customer transactions within the previous hour |
| `transaction_count_24h` | Number of customer transactions within the previous 24 hours |
| `is_new_country` | Indicates whether the current country has not appeared previously for the customer |
| `distinct_countries_24h` | Number of distinct transaction countries used during the previous 24 hours |

The features capture different dimensions of behaviour:

```text
amount deviation
+ short-term velocity
+ daily activity
+ geographic novelty
+ geographic dispersion
```

No manually defined AML risk rules are included in the final score.

---

## Isolation Forest Model

The model is implemented with scikit-learn.

```python
model = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination=0.03,
    random_state=42,
    n_jobs=-1
)
```

### Why Isolation Forest?

Isolation Forest is suitable for this proof of concept because:

- it does not require labelled training data;
- it detects unusual combinations of behavioural features;
- it works well for anomaly-ranking use cases;
- it produces an anomaly score for every transaction.

The model does not predict whether a transaction is money laundering. It identifies transactions that are statistically unusual compared with the rest of the dataset.

---

## Risk Scoring

The raw Isolation Forest score is reversed so that a larger value means a more unusual transaction.

```python
anomaly_score_raw = -model.score_samples(X)
```

The raw score is then converted into a percentile-based score from 0 to 100.

```text
Higher ML risk score
        =
more unusual transaction relative to the dataset
```

Example:

```text
ML risk score 99
```

means that the transaction has a higher anomaly score than approximately 99% of the transactions in the dataset.

The score is not a fraud probability.

---

## Model Evaluation

The synthetic scenario label was used only after scoring to evaluate the model.

With `contamination=0.03`, the final model produced:

| Metric | Result |
|---|---:|
| True positives | 9 |
| False positives | 21 |
| False negatives | 21 |
| True negatives | 949 |
| Precision | 30% |
| Recall | 30% |
| Analyst queue size | 30 |

This result shows that controlling the queue size does not automatically guarantee strong anomaly detection.

An additional experiment using `contamination="auto"` produced much higher recall but also a much larger number of false positives:

| Setting | Alerts | Precision | Recall |
|---|---:|---:|---:|
| `contamination=0.03` | 30 | 30% | 30% |
| `contamination="auto"` | 234 | 12% | 93% |

This illustrates an important AML trade-off:

```text
higher recall
→ more detected anomalies
→ more false positives
→ greater analyst workload
```

In a production environment, the operating threshold should be calibrated using historical investigation outcomes, false-positive rates, risk appetite, and analyst capacity.

---

## Gold Analyst Queue

Transactions classified as outliers are written into a managed Delta table.

```text
workspace.aml_ml_3pct.gold_analyst_queue
```

The Gold table includes:

- analyst queue rank;
- transaction details;
- customer ID;
- ML risk score;
- raw anomaly score;
- transaction amount;
- country;
- transaction type;
- five behavioural features;
- scoring timestamp.

The synthetic `generated_scenario` field is deliberately excluded.

---

## Qlik Sense Dashboard

The Gold analyst queue is connected to Qlik Sense.

The dashboard visualizes AML alerts, transaction risk scores, model performance, and analyst investigation priorities.

![Qlik Sense AML ML Dashboard](https://github.com/allanvissor-max/aml-anomaly-detection-databricks/blob/main/QlikSense%20AML%20ML%20dashboard.jpg?raw=true)

The dashboard includes:

### KPI Cards

- Flagged Transactions
- Highest ML Risk Score
- Average ML Risk Score

### Filters

- Country
- Transaction Type
- New Country Flag
- Customer ID

### Visualizations

- Flagged Transactions by Country
- Flagged Transactions by Type
- Transaction Amount vs ML Risk Score
- AML Analyst Investigation Queue

The dashboard allows analysts to filter, compare, and prioritize flagged transactions.

---

## Project Structure

```text
aml-ml-anomaly-detection/
│
├── README.md
├── lakeflow/
│   └── 01_aml_pipeline.py
├── notebooks/
│   └── 02_train_isolation_forest.py
├── data/
│   └── aml_transactions_1000_realistic_3pct.json
└── screenshots/
    └── qlik_aml_dashboard.png
```

---

## How to Run

### 1. Create Databricks Objects

Create a schema and source volume.

```sql
CREATE SCHEMA IF NOT EXISTS workspace.aml_ml_3pct;

CREATE VOLUME IF NOT EXISTS workspace.aml_ml_3pct.source_files;
```

### 2. Upload the JSON File

Upload the source file into:

```text
/Volumes/workspace/aml_ml_3pct/source_files/
```

### 3. Create the Lakeflow Pipeline

Create a new Databricks ETL pipeline and configure:

```text
Target catalog: workspace
Target schema: aml_ml_3pct
```

Add:

```text
lakeflow/01_aml_pipeline.py
```

Run a full refresh.

### 4. Verify the Scenario Distribution

```python
display(
    spark.table("workspace.aml_ml_3pct.silver_transactions")
    .groupBy("generated_scenario")
    .count()
    .orderBy("generated_scenario")
)
```

Expected result:

```text
COMBINED           4
HIGH_AMOUNT       10
HIGH_VELOCITY      8
NEW_COUNTRY        8
NORMAL           970
```

### 5. Train and Score the Model

Run:

```text
notebooks/02_train_isolation_forest.py
```

The notebook:

- reads the feature table;
- trains Isolation Forest;
- calculates anomaly scores;
- evaluates model performance;
- creates the Gold analyst queue.

### 6. Connect Qlik Sense

Connect Qlik Sense to Databricks and load:

```text
workspace.aml_ml_3pct.gold_analyst_queue
```

Build the dashboard using the Gold table.

---

## Key Learnings

This project demonstrates that anomaly detection is not only a modelling problem.

A useful AML solution must also consider:

- data quality;
- customer history;
- feature design;
- score interpretation;
- threshold calibration;
- false positives;
- analyst workload;
- transparent reporting;
- human investigation.

The project also shows why anomaly detection should support analysts rather than replace them.

---

## Summary

> I built an end-to-end AML anomaly detection proof of concept using Databricks and Qlik Sense. JSON transactions are ingested through a Lakeflow Bronze and Silver pipeline, transformed into five customer behaviour features, and scored with an Isolation Forest model. The highest-risk transactions are written into a Gold analyst queue and visualized in Qlik Sense. I also evaluated precision and recall and explored how the alert threshold affects both anomaly coverage and analyst workload.
