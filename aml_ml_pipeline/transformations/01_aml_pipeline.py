
from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

# The JSON file is located directly in this Unity Catalog volume.
SOURCE_PATH = "/Volumes/workspace/aml_ml/source_files/"


# ============================================================
# 1. BRONZE LAYER
# ============================================================
#
# Purpose:
# - Read the original JSON transactions
# - Keep the original source columns
# - Add ingestion metadata
#
# Auto Loader reads existing files and can later process
# new JSON files added to the same folder.
# ============================================================

@dp.table(
    name="bronze_transactions",
    comment="Raw AML transactions loaded incrementally from JSON files."
)
def bronze_transactions():

    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .load(SOURCE_PATH)

        # Time when the transaction entered the Bronze layer.
        .withColumn(
            "ingested_at",
            F.current_timestamp()
        )

        # Full path of the original source file.
        .withColumn(
            "source_file",
            F.col("_metadata.file_path")
        )
    )


# ============================================================
# 2. SILVER LAYER
# ============================================================
#
# Purpose:
# - Select the required columns
# - Convert columns to correct data types
# - Standardize text values
# - Apply basic technical data cleaning
# - Remove duplicate transaction IDs
#
# These are technical cleaning rules, not AML business rules.
# ============================================================

@dp.materialized_view(
    name="silver_transactions",
    comment="Cleaned and correctly typed AML transactions."
)
def silver_transactions():

    bronze_df = spark.read.table(
        "bronze_transactions"
    )

    silver_df = (
        bronze_df

        # Select and cast the required fields.
        .select(
            F.col("transaction_id")
            .cast("string")
            .alias("transaction_id"),

            F.col("customer_id")
            .cast("string")
            .alias("customer_id"),

            F.to_timestamp(
                F.col("transaction_timestamp")
            ).alias("transaction_timestamp"),

            F.col("amount_eur")
            .cast("double")
            .alias("amount_eur"),

            F.upper(
                F.trim(
                    F.col("currency").cast("string")
                )
            ).alias("currency"),

            F.upper(
                F.trim(
                    F.col("country").cast("string")
                )
            ).alias("country"),

            F.upper(
                F.trim(
                    F.col("transaction_type").cast("string")
                )
            ).alias("transaction_type"),

            # This is a synthetic evaluation label.
            # It will not be given to the ML model.
            F.upper(
                F.trim(
                    F.col("generated_scenario").cast("string")
                )
            ).alias("generated_scenario"),

            F.col("ingested_at"),
            F.col("source_file")
        )

        # Keep only technically usable transactions.
        .filter(
            F.col("transaction_id").isNotNull()
            & F.col("customer_id").isNotNull()
            & F.col("transaction_timestamp").isNotNull()
            & F.col("amount_eur").isNotNull()
            & (F.col("amount_eur") > 0)
        )

        # Keep one row per transaction ID.
        .dropDuplicates(
            ["transaction_id"]
        )
    )

    return silver_df


# ============================================================
# 3. CUSTOMER BEHAVIOUR FEATURES
# ============================================================
#
# Purpose:
# Create five transaction-level behavioural features:
#
# 1. amount_deviation_ratio
# 2. transaction_count_1h
# 3. transaction_count_24h
# 4. is_new_country
# 5. distinct_countries_24h
#
# Each output row still represents one transaction.
# ============================================================

@dp.materialized_view(
    name="customer_behavior_features",
    comment="Transaction-level customer behaviour features for anomaly detection."
)
def customer_behavior_features():

    transactions = (
        spark.read
        .table("silver_transactions")

        # Unix timestamp used for time-based Spark windows.
        .withColumn(
            "event_epoch",
            F.col("transaction_timestamp").cast("long")
        )
    )


    # ========================================================
    # WINDOW 1: PREVIOUS CUSTOMER TRANSACTIONS
    # ========================================================
    #
    # Includes all earlier transactions of the customer.
    # The current transaction is excluded.
    #
    # Used for:
    # - historical transaction count
    # - historical median amount
    # ========================================================

    historical_window = (
        Window
        .partitionBy("customer_id")
        .orderBy(
            F.col("transaction_timestamp"),
            F.col("transaction_id")
        )
        .rowsBetween(
            Window.unboundedPreceding,
            -1
        )
    )


    # ========================================================
    # WINDOW 2: CUSTOMER TRANSACTION SEQUENCE
    # ========================================================
    #
    # Assigns a chronological sequence number to every
    # customer transaction.
    # ========================================================

    customer_sequence_window = (
        Window
        .partitionBy("customer_id")
        .orderBy(
            F.col("transaction_timestamp"),
            F.col("transaction_id")
        )
    )


    # ========================================================
    # WINDOW 3: CUSTOMER-COUNTRY SEQUENCE
    # ========================================================
    #
    # Assigns a chronological sequence number for each
    # customer and country combination.
    #
    # A value of 1 means that this is the first observed
    # transaction in that country for the customer.
    # ========================================================

    customer_country_window = (
        Window
        .partitionBy(
            "customer_id",
            "country"
        )
        .orderBy(
            F.col("transaction_timestamp"),
            F.col("transaction_id")
        )
    )


    # ========================================================
    # WINDOW 4: LAST 1 HOUR
    # ========================================================
    #
    # Includes transactions from the current transaction time
    # back to 3,600 seconds earlier.
    # ========================================================

    last_1h_window = (
        Window
        .partitionBy("customer_id")
        .orderBy("event_epoch")
        .rangeBetween(
            -3600,
            0
        )
    )


    # ========================================================
    # WINDOW 5: LAST 24 HOURS
    # ========================================================
    #
    # Includes transactions from the current transaction time
    # back to 86,400 seconds earlier.
    # ========================================================

    last_24h_window = (
        Window
        .partitionBy("customer_id")
        .orderBy("event_epoch")
        .rangeBetween(
            -86400,
            0
        )
    )


    # ========================================================
    # CALCULATE SUPPORTING COLUMNS AND FIVE ML FEATURES
    # ========================================================

    feature_df = (
        transactions

        # ----------------------------------------------------
        # SUPPORTING COLUMN:
        # Number of customer transactions before the current
        # transaction.
        # ----------------------------------------------------

        .withColumn(
            "historical_transaction_count",
            F.count(
                F.lit(1)
            ).over(
                historical_window
            )
        )


        # ----------------------------------------------------
        # SUPPORTING COLUMN:
        # Median amount of the customer's earlier transactions.
        #
        # The current transaction is excluded.
        # ----------------------------------------------------

        .withColumn(
            "historical_median_amount",
            F.percentile_approx(
                F.col("amount_eur"),
                0.5,
                1000
            ).over(
                historical_window
            )
        )


        # ----------------------------------------------------
        # SUPPORTING COLUMN:
        # Position of the transaction in the customer's full
        # transaction history.
        # ----------------------------------------------------

        .withColumn(
            "customer_transaction_sequence",
            F.row_number().over(
                customer_sequence_window
            )
        )


        # ----------------------------------------------------
        # SUPPORTING COLUMN:
        # Position of the transaction within the customer's
        # history for the current country.
        # ----------------------------------------------------

        .withColumn(
            "customer_country_sequence",
            F.row_number().over(
                customer_country_window
            )
        )


        # ====================================================
        # FEATURE 1: AMOUNT DEVIATION RATIO
        # ====================================================
        #
        # Formula:
        #
        # current amount / historical customer median amount
        #
        # Example:
        #
        # Current transaction:       600 EUR
        # Historical median:         150 EUR
        # amount_deviation_ratio:    4.0
        #
        # A neutral value of 1.0 is used when the customer has
        # fewer than three previous transactions.
        # ====================================================

        .withColumn(
            "amount_deviation_ratio",
            F.when(
                (
                    F.col(
                        "historical_transaction_count"
                    ) >= 3
                )
                & (
                    F.col(
                        "historical_median_amount"
                    ) > 0
                ),
                F.col("amount_eur")
                / F.col("historical_median_amount")
            ).otherwise(
                F.lit(1.0)
            )
        )


        # ====================================================
        # FEATURE 2: TRANSACTION COUNT IN LAST 1 HOUR
        # ====================================================
        #
        # Measures short-term transaction velocity.
        #
        # The current transaction is included in the count.
        # Therefore, the minimum value is 1.
        # ====================================================

        .withColumn(
            "transaction_count_1h",
            F.count(
                F.lit(1)
            ).over(
                last_1h_window
            )
        )


        # ====================================================
        # FEATURE 3: TRANSACTION COUNT IN LAST 24 HOURS
        # ====================================================
        #
        # Measures the customer's daily transaction activity.
        #
        # The current transaction is included in the count.
        # ====================================================

        .withColumn(
            "transaction_count_24h",
            F.count(
                F.lit(1)
            ).over(
                last_24h_window
            )
        )


        # ====================================================
        # FEATURE 4: IS NEW COUNTRY
        # ====================================================
        #
        # Value 1:
        # The country has not appeared previously in the
        # customer's transaction history.
        #
        # Value 0:
        # The country has appeared before.
        #
        # The first-ever customer transaction receives 0
        # because no meaningful country history exists yet.
        # ====================================================

        .withColumn(
            "is_new_country",
            F.when(
                (
                    F.col(
                        "customer_transaction_sequence"
                    ) > 1
                )
                & (
                    F.col(
                        "customer_country_sequence"
                    ) == 1
                ),
                F.lit(1)
            ).otherwise(
                F.lit(0)
            )
        )


        # ====================================================
        # FEATURE 5: DISTINCT COUNTRIES IN LAST 24 HOURS
        # ====================================================
        #
        # Counts how many different countries appeared in the
        # customer's transactions during the previous 24 hours.
        #
        # Example:
        #
        # EE, EE, FI, DE
        #
        # distinct_countries_24h = 3
        # ====================================================

        .withColumn(
            "distinct_countries_24h",
            F.size(
                F.collect_set(
                    "country"
                ).over(
                    last_24h_window
                )
            )
        )
    )


    # ========================================================
    # FINAL FEATURE TABLE
    # ========================================================
    #
    # Supporting calculation columns are removed.
    #
    # generated_scenario remains temporarily available only
    # for offline model evaluation. It must never be included
    # in the Isolation Forest feature list.
    # ========================================================

    return feature_df.select(
        "transaction_id",
        "customer_id",
        "transaction_timestamp",
        "amount_eur",
        "currency",
        "country",
        "transaction_type",

        "amount_deviation_ratio",
        "transaction_count_1h",
        "transaction_count_24h",
        "is_new_country",
        "distinct_countries_24h",

        "generated_scenario"
    )