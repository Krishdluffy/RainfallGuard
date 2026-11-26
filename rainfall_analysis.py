import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"
CHART_DIR = OUT_DIR / "charts"

DATASET_PATTERN = "rainfall_manual_daily_kerala_sw_kl_2021_2025*.csv"
HEAVY_MM = 64.5
MAX_PLAUSIBLE_MM = 1000.0
MIN_STATION_OBS = 100
Z_THRESHOLD = 3.0
SEED = 42

# Chronological split fractions: earliest 70% train, next 15% validation,
# latest 15% test. Validation is used for model selection; test is held out.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

COLUMN_MAP = {
    "SlNo": "sl_no",
    "Station": "station",
    "State": "state",
    "District LGD Code": "district_code",
    "District": "district",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Data Acquisition Time": "acq_time",
    "Manual Daily Rainfall (mm)": "rain_raw",
}

DATE_FORMATS = [
    "dd-MM-yyyy HH:mm",
    "dd-MM-yyyy HH:mm:ss",
    "dd-MM-yyyy",
    "dd/MM/yyyy HH:mm",
    "dd/MM/yyyy HH:mm:ss",
    "dd/MM/yyyy",
    "yyyy-MM-dd HH:mm:ss",
    "yyyy-MM-dd HH:mm",
    "yyyy-MM-dd",
    "MM/dd/yyyy HH:mm:ss",
    "MM/dd/yyyy HH:mm",
    "MM/dd/yyyy",
]

NUMERIC_FEATURES = [
    "rain_d1", "rain_d2", "d2_observed",
    "sum_3d", "mean_3d", "n_obs_3d",
    "sum_7d", "mean_7d", "max_7d", "n_obs_7d",
    "sum_14d", "mean_14d", "max_14d", "n_obs_14d",
    "days_since_prev_obs", "prev_observed",
    "month", "doy_sin", "doy_cos", "latitude", "longitude",
]

COLORS = {
    "blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a",
    "violet": "#6250d6", "red": "#e34948", "grey": "#8a8985",
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def make_dirs():
    for folder in (DATA_DIR, OUT_DIR, MODEL_DIR, CHART_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def build_spark():
    spark = (
        SparkSession.builder
        .appName("RainfallWatchKerala")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "4")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log(f"Spark {spark.version} started; master={spark.sparkContext.master}")
    return spark


def find_dataset():
    matches = sorted(DATA_DIR.glob(DATASET_PATTERN))
    if not matches:
        raise FileNotFoundError(
            f"No dataset matching '{DATASET_PATTERN}' was found in {DATA_DIR}."
        )
    if len(matches) > 1:
        log(f"Multiple matching CSVs found; using {matches[0].name}")
    return matches[0]


def write_spark_csv(df, name):
    target = OUT_DIR / name
    (
        df.coalesce(1).write.mode("overwrite")
        .option("header", True).csv(str(target))
    )
    log(f"Saved CSV output directory: {target}")


def save_model(model, name):
    target = MODEL_DIR / name
    model.write().overwrite().save(str(target))
    log(f"Saved model: {target}")


def require_rows(df, label):
    if df.limit(1).count() == 0:
        raise ValueError(f"{label} is empty. Cannot continue.")


def load_raw(spark):
    path = find_dataset()
    log(f"Reading dataset: {path}")
    df = (
        spark.read.option("header", True).option("inferSchema", False)
        .option("encoding", "UTF-8").csv(str(path))
    )
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}\nActual: {df.columns}")

    raw = df.select([
        F.col(f"`{source}`").alias(target)
        for source, target in COLUMN_MAP.items()
    ])
    require_rows(raw, "Input CSV")
    print("\nInput columns:", df.columns)
    raw.printSchema()
    raw.show(5, truncate=False)
    log(f"Input records: {raw.count():,}")
    return raw


def prepare(raw):
    def tidy(column):
        return F.trim(F.regexp_replace(F.col(column), r"\s+", " "))

    parsed_dates = [F.to_timestamp(tidy("acq_time"), fmt) for fmt in DATE_FORMATS]
    return (
        raw
        .withColumn("station_key", F.upper(tidy("station")))
        .withColumn("district_std", F.upper(tidy("district")))
        .withColumn("acq_ts", F.coalesce(*parsed_dates))
        .withColumn("obs_date", F.to_date("acq_ts"))
        .withColumn(
            "rain_blank",
            F.col("rain_raw").isNull() | (F.trim(F.col("rain_raw")) == ""),
        )
        .withColumn("rain_num", F.col("rain_raw").cast("double"))
        .withColumn("lat", F.col("latitude").cast("double"))
        .withColumn("lon", F.col("longitude").cast("double"))
    )


def data_quality(p):
    section("DATA QUALITY REPORT")
    total = p.count()
    stats = p.agg(
        F.sum(F.col("rain_blank").cast("int")).alias("blank"),
        F.sum(((~F.col("rain_blank")) & F.col("rain_num").isNull()).cast("int")).alias("nonnumeric"),
        F.sum((F.col("rain_num") < 0).cast("int")).alias("negative"),
        F.sum((F.col("rain_num") == 0).cast("int")).alias("zero"),
        F.sum((F.col("rain_num") > MAX_PLAUSIBLE_MM).cast("int")).alias("implausible"),
        F.sum(F.col("acq_ts").isNull().cast("int")).alias("bad_dates"),
        F.min("obs_date").alias("first_date"),
        F.max("obs_date").alias("last_date"),
        F.countDistinct("station_key").alias("stations"),
        F.countDistinct("district_std").alias("districts"),
        F.min("rain_num").alias("rain_min"),
        F.avg("rain_num").alias("rain_mean"),
        F.max("rain_num").alias("rain_max"),
    ).first()

    def n(name):
        return int(stats[name] or 0)

    report = [
        ("rows", total), ("distinct_stations", n("stations")),
        ("distinct_districts", n("districts")),
        ("unparseable_dates", n("bad_dates")),
        ("first_date", str(stats["first_date"])),
        ("last_date", str(stats["last_date"])),
        ("blank_rainfall", n("blank")), ("nonnumeric_rainfall", n("nonnumeric")),
        ("negative_rainfall", n("negative")), ("zero_rainfall", n("zero")),
        ("rainfall_above_limit", n("implausible")),
        ("rainfall_min_mm", stats["rain_min"]),
        ("rainfall_mean_mm", stats["rain_mean"]),
        ("rainfall_max_mm", stats["rain_max"]),
    ]
    quantiles = p.filter(F.col("rain_num").isNotNull()).approxQuantile(
        "rain_num", [0.5, 0.9, 0.95, 0.99], 0.001
    )
    for label, value in zip(["median_mm", "p90_mm", "p95_mm", "p99_mm"], quantiles):
        report.append((label, value))

    report.append((
        "duplicate_rows_ignoring_sl_no",
        total - p.drop("sl_no").distinct().count(),
    ))
    pdf = pd.DataFrame(report, columns=["check", "value"])
    pdf.to_csv(OUT_DIR / "data_quality_report.csv", index=False)
    print(pdf.to_string(index=False))
    if n("bad_dates") == total:
        raise ValueError("Every date failed to parse. Check Data Acquisition Time format.")
    return pdf


def clean_data(p):
    section("DATA CLEANING")
    valid = p.filter(
        F.col("obs_date").isNotNull()
        & F.col("rain_num").isNotNull()
        & (F.col("rain_num") >= 0)
        & (F.col("rain_num") <= MAX_PLAUSIBLE_MM)
        & F.col("station_key").isNotNull()
        & (F.col("station_key") != "")
        & F.col("district_std").isNotNull()
        & F.col("lat").isNotNull()
        & F.col("lon").isNotNull()
    )

    # One row per station/date. Maximum is used to avoid averaging duplicate
    # reports into an artificially lower daily rainfall value.
    cleaned = (
        valid.groupBy("station_key", "obs_date")
        .agg(
            F.max("district_std").alias("district"),
            F.avg("lat").alias("latitude"),
            F.avg("lon").alias("longitude"),
            F.max("rain_num").alias("rain_mm"),
            F.count("*").alias("source_records"),
            F.countDistinct("rain_num").alias("distinct_rain_values"),
        )
        .withColumnRenamed("station_key", "station")
        .withColumnRenamed("obs_date", "date")
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
        .withColumn("doy", F.dayofyear("date"))
        .withColumn(
            "season",
            F.when(F.col("month").isin(1, 2), "Winter")
            .when(F.col("month").isin(3, 4, 5), "Pre-monsoon")
            .when(F.col("month").isin(6, 7, 8, 9), "SW monsoon")
            .otherwise("Post-monsoon"),
        )
        .withColumn("day_num", F.datediff(F.col("date"), F.lit("1970-01-01")))
    ).cache()

    require_rows(cleaned, "Cleaned data")
    log(f"Clean station-day records: {cleaned.count():,}")
    log("Missing rainfall remains missing; it is not converted to zero.")
    return cleaned


def sql_analysis(spark, df):
    section("SPARK SQL ANALYSIS")
    df.createOrReplaceTempView("rainfall")
    queries = {
        "district_summary": """
            SELECT district, COUNT(DISTINCT station) AS stations,
            COUNT(*) AS observations, ROUND(AVG(rain_mm), 2) AS avg_recorded_rain_mm,
            ROUND(PERCENTILE_APPROX(rain_mm, 0.5), 2) AS median_mm,
            ROUND(PERCENTILE_APPROX(rain_mm, 0.95), 2) AS p95_mm,
            ROUND(MAX(rain_mm), 1) AS max_mm
            FROM rainfall GROUP BY district ORDER BY avg_recorded_rain_mm DESC
        """,
        "station_summary": """
            SELECT station, district, COUNT(*) AS observations,
            ROUND(AVG(rain_mm), 2) AS avg_rain_mm,
            ROUND(STDDEV(rain_mm), 2) AS std_rain_mm,
            ROUND(MAX(rain_mm), 1) AS max_rain_mm,
            MIN(date) AS first_date, MAX(date) AS last_date
            FROM rainfall GROUP BY station, district ORDER BY avg_rain_mm DESC
        """,
        "top_rainfall_records": """
            SELECT station, district, date, rain_mm FROM rainfall
            ORDER BY rain_mm DESC LIMIT 25
        """,
        "monthly_rainfall": """
            SELECT year, month, CONCAT(year, '-', LPAD(month, 2, '0')) AS year_month,
            ROUND(AVG(rain_mm), 2) AS avg_recorded_rain_mm, COUNT(*) AS observations,
            COUNT(DISTINCT station) AS stations_reporting, ROUND(MAX(rain_mm), 1) AS max_mm
            FROM rainfall GROUP BY year, month ORDER BY year, month
        """,
        "monthly_climatology": """
            SELECT month, ROUND(AVG(rain_mm), 2) AS avg_recorded_rain_mm,
            COUNT(*) AS observations,
            ROUND(PERCENTILE_APPROX(rain_mm, 0.95), 2) AS p95_mm
            FROM rainfall GROUP BY month ORDER BY month
        """,
        "yearly_summary": """
            SELECT year, ROUND(AVG(rain_mm), 2) AS avg_recorded_rain_mm,
            COUNT(*) AS observations, COUNT(DISTINCT station) AS stations_reporting,
            ROUND(MAX(rain_mm), 1) AS max_mm
            FROM rainfall GROUP BY year ORDER BY year
        """,
        "seasonal_summary": """
            SELECT season, district, ROUND(AVG(rain_mm), 2) AS avg_recorded_rain_mm,
            COUNT(*) AS observations FROM rainfall GROUP BY season, district
            ORDER BY season, avg_recorded_rain_mm DESC
        """,
    }
    results = {}
    for name, query in queries.items():
        result = spark.sql(query)
        write_spark_csv(result, name)
        results[name] = result.toPandas()
    print("\nDistrict summary:")
    print(results["district_summary"].to_string(index=False))
    print("\nYearly summary:")
    print(results["yearly_summary"].to_string(index=False))
    return results


def detect_anomalies(df):
    section("STATISTICAL ANOMALY DETECTION")
    station_stats = df.groupBy("station").agg(
        F.count("*").alias("station_n"),
        F.avg(F.log1p("rain_mm")).alias("station_mu"),
        F.stddev(F.log1p("rain_mm")).alias("station_sd"),
        F.expr("percentile_approx(rain_mm, 0.99, 10000)").alias("station_p99"),
    )
    district_stats = df.groupBy("district").agg(
        F.avg(F.log1p("rain_mm")).alias("district_mu"),
        F.stddev(F.log1p("rain_mm")).alias("district_sd"),
        F.expr("percentile_approx(rain_mm, 0.99, 10000)").alias("district_p99"),
    )
    scored = (
        df.join(station_stats, "station", "left")
        .join(district_stats, "district", "left")
        .withColumn("use_station", F.col("station_n") >= MIN_STATION_OBS)
        .withColumn("baseline_mu", F.when(F.col("use_station"), F.col("station_mu")).otherwise(F.col("district_mu")))
        .withColumn("baseline_sd", F.when(F.col("use_station"), F.col("station_sd")).otherwise(F.col("district_sd")))
        .withColumn("baseline_p99_mm", F.when(F.col("use_station"), F.col("station_p99")).otherwise(F.col("district_p99")))
        .withColumn(
            "z_log",
            (F.log1p("rain_mm") - F.col("baseline_mu"))
            / F.when(F.col("baseline_sd") > 0, F.col("baseline_sd")),
        )
        .withColumn("above_p99", F.col("rain_mm") >= F.col("baseline_p99_mm"))
        .withColumn("high_z", F.coalesce(F.col("z_log") >= Z_THRESHOLD, F.lit(False)))
        .withColumn("is_anomaly", F.col("above_p99") | F.col("high_z"))
        .withColumn("reaches_heavy_threshold", F.col("rain_mm") >= HEAVY_MM)
    )
    anomalies = scored.filter(F.col("is_anomaly")).select(
        "station", "district", "date", "rain_mm",
        F.when(F.col("use_station"), "station").otherwise("district").alias("baseline_level"),
        F.round("baseline_p99_mm", 2).alias("baseline_p99_mm"),
        F.round("z_log", 2).alias("z_log"),
        "above_p99", "high_z", "reaches_heavy_threshold",
    ).cache()

    count = anomalies.count()
    total = df.count()
    log(f"Flagged {count:,} of {total:,} observations ({100 * count / max(total, 1):.2f}%).")
    write_spark_csv(anomalies.orderBy(F.col("rain_mm").desc()), "rainfall_anomalies")
    by_district = anomalies.groupBy("district").count().orderBy(F.col("count").desc()).toPandas()
    by_month = anomalies.withColumn("month", F.month("date")).groupBy("month").count().orderBy("month").toPandas()
    anomalies.show(10, truncate=False)
    anomalies.unpersist()
    return {"by_district": by_district, "by_month": by_month, "count": count}


def engineer_features(df):
    section("FEATURE ENGINEERING")
    window = Window.partitionBy("station").orderBy("day_num")

    def day_range(first, last):
        return window.rangeBetween(first, last)

    previous_day = F.lag("day_num", 1).over(window)
    features = (
        df
        .withColumn("rain_d1", F.col("rain_mm"))
        .withColumn("rain_d2", F.max("rain_mm").over(day_range(-1, -1)))
        .withColumn("d2_observed", F.col("rain_d2").isNotNull().cast("double"))
        .withColumn("prev_observed", previous_day.isNotNull().cast("double"))
        .withColumn(
            "days_since_prev_obs",
            F.when(previous_day.isNotNull(), F.least(F.col("day_num") - previous_day, F.lit(30)))
            .otherwise(F.lit(30.0)),
        )
        .withColumn("sum_3d", F.sum("rain_mm").over(day_range(-2, 0)))
        .withColumn("n_obs_3d", F.count("rain_mm").over(day_range(-2, 0)))
        .withColumn("sum_7d", F.sum("rain_mm").over(day_range(-6, 0)))
        .withColumn("max_7d", F.max("rain_mm").over(day_range(-6, 0)))
        .withColumn("n_obs_7d", F.count("rain_mm").over(day_range(-6, 0)))
        .withColumn("sum_14d", F.sum("rain_mm").over(day_range(-13, 0)))
        .withColumn("max_14d", F.max("rain_mm").over(day_range(-13, 0)))
        .withColumn("n_obs_14d", F.count("rain_mm").over(day_range(-13, 0)))
        .withColumn("mean_3d", F.col("sum_3d") / F.col("n_obs_3d"))
        .withColumn("mean_7d", F.col("sum_7d") / F.col("n_obs_7d"))
        .withColumn("mean_14d", F.col("sum_14d") / F.col("n_obs_14d"))
        .withColumn("doy_sin", F.sin(2 * math.pi * F.col("doy") / 366.0))
        .withColumn("doy_cos", F.cos(2 * math.pi * F.col("doy") / 366.0))
        .withColumn("target_date", F.date_add("date", 1))
        .withColumn("target_day_num", F.datediff(F.date_add("date", 1), F.lit("1970-01-01")))
    )

    # Missing previous-day rainfall is represented with a value plus an indicator.
    # Rolling sums/counts also expose how many observations were available.
    features = features.fillna({
        "rain_d2": 0.0,
        "days_since_prev_obs": 30.0,
        "sum_3d": 0.0, "sum_7d": 0.0, "sum_14d": 0.0,
        "max_7d": 0.0, "max_14d": 0.0,
        "mean_3d": 0.0, "mean_7d": 0.0, "mean_14d": 0.0,
        "n_obs_3d": 0, "n_obs_7d": 0, "n_obs_14d": 0,
    })

    future = df.select(
        F.col("station").alias("future_station"),
        F.col("date").alias("future_date"),
        F.col("rain_mm").alias("next_rain"),
    )
    features = (
        features.join(
            future,
            (features.station == future.future_station)
            & (features.target_date == future.future_date),
            "left",
        )
        .drop("future_station", "future_date")
        .cache()
    )
    total = features.count()
    usable = features.filter(F.col("next_rain").isNotNull()).count()
    log(f"Feature rows: {total:,}; rows with observed next-day target: {usable:,}")
    log("Forecast features use information available through the observation date.")
    return features


def preprocessing_stages():
    return [
        StringIndexer(inputCol="district", outputCol="district_idx", handleInvalid="keep"),
        StringIndexer(inputCol="station", outputCol="station_idx", handleInvalid="keep"),
        OneHotEncoder(
            inputCols=["district_idx", "station_idx"],
            outputCols=["district_ohe", "station_ohe"],
            handleInvalid="keep",
        ),
        VectorAssembler(
            inputCols=NUMERIC_FEATURES + ["district_ohe", "station_ohe"],
            outputCol="features",
            handleInvalid="keep",
        ),
    ]


def split_chronologically(features):
    model_df = features.filter(F.col("next_rain").isNotNull())
    require_rows(model_df, "Model dataset")
    dates = model_df.approxQuantile("target_day_num", [TRAIN_FRACTION, TRAIN_FRACTION + VALIDATION_FRACTION], 0.001)
    train_cut = int(dates[0])
    val_cut = int(dates[1])

    train = model_df.filter(F.col("target_day_num") < train_cut).cache()
    validation = model_df.filter(
        (F.col("target_day_num") >= train_cut) & (F.col("target_day_num") < val_cut)
    ).cache()
    test = model_df.filter(F.col("target_day_num") >= val_cut).cache()

    for name, frame in [("Training", train), ("Validation", validation), ("Testing", test)]:
        count = frame.count()
        bounds = frame.agg(F.min("target_date"), F.max("target_date")).first()
        log(f"{name}: {count:,} rows; target dates {bounds[0]} to {bounds[1]}")
        if count == 0:
            raise ValueError(f"{name} split is empty.")
    return train, validation, test, train_cut, val_cut


def regression_metrics(df, prediction_col="pred_rain"):
    evaluator = RegressionEvaluator(labelCol="next_rain", predictionCol=prediction_col)
    return {
        metric: evaluator.setMetricName(metric).evaluate(df)
        for metric in ("mae", "rmse", "r2")
    }


def add_regression_baselines(predictions, train_mean):
    return (
        predictions
        .withColumn("baseline_persistence", F.col("rain_d1"))
        .withColumn("baseline_train_mean", F.lit(float(train_mean)))
    )


def run_regression(train, validation, test):
    section("REGRESSION MODEL COMPARISON")
    candidates = {
        "RandomForestRegressor": RandomForestRegressor(
            featuresCol="features", labelCol="next_rain", predictionCol="pred_rain",
            numTrees=200, maxDepth=8, maxBins=64, featureSubsetStrategy="sqrt", seed=SEED,
        ),
        "GBTRegressor": GBTRegressor(
            featuresCol="features", labelCol="next_rain", predictionCol="pred_rain",
            maxIter=100, maxDepth=5, stepSize=0.05, seed=SEED,
        ),
    }
    fitted = {}
    validation_scores = {}
    rows = []
    train_mean = train.agg(F.avg("next_rain")).first()[0]

    for name, estimator in candidates.items():
        log(f"Fitting {name}")
        model = Pipeline(stages=preprocessing_stages() + [estimator]).fit(train)
        valid_pred = add_regression_baselines(model.transform(validation), train_mean).cache()
        metrics = regression_metrics(valid_pred)
        validation_scores[name] = metrics["mae"]
        fitted[name] = model
        print(f"\n{name} validation metrics: {metrics}")
        for metric, value in metrics.items():
            rows.append({"task": "regression", "model": name, "split": "validation", "metric": metric, "value": value})
        valid_pred.unpersist()

    # Select by validation MAE only; final test data is not used for selection.
    winner = min(validation_scores, key=validation_scores.get)
    log(f"Selected regression model by validation MAE: {winner}")
    best_model = fitted[winner]
    test_pred = add_regression_baselines(best_model.transform(test), train_mean).cache()

    for name, col in [
        (winner, "pred_rain"),
        ("Baseline: today's rainfall", "baseline_persistence"),
        ("Baseline: training mean", "baseline_train_mean"),
    ]:
        metrics = regression_metrics(test_pred, col)
        print(f"\nTEST {name}: {metrics}")
        for metric, value in metrics.items():
            rows.append({"task": "regression", "model": name, "split": "test", "metric": metric, "value": value})

    # Feature importance is available for the selected tree-based estimator.
    estimator_model = best_model.stages[-1]
    importance_df = pd.DataFrame()
    if hasattr(estimator_model, "featureImportances"):
        importances = estimator_model.featureImportances.toArray()
        try:
            attrs = test_pred.schema["features"].metadata["ml_attr"]["attrs"]
            all_attrs = sorted([item for group in attrs.values() for item in group], key=lambda x: x["idx"])
            names = [item["name"] for item in all_attrs]
        except Exception:
            names = [f"feature_{i}" for i in range(len(importances))]
        if len(names) != len(importances):
            names = [f"feature_{i}" for i in range(len(importances))]
        importance_df = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
        print("\nTop feature importances:")
        print(importance_df.head(15).to_string(index=False))

    return winner, best_model, test_pred, rows, importance_df, fitted


def classifier_metrics(predictions, name):
    counts = {
        (int(row["label"]), int(row["prediction"])): int(row["count"])
        for row in predictions.groupBy("label", "prediction").count().collect()
    }
    tp, fp = counts.get((1, 1), 0), counts.get((0, 1), 0)
    fn, tn = counts.get((1, 0), 0), counts.get((0, 0), 0)
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    roc_auc = float("nan")
    pr_auc = float("nan")
    if tp + fn > 0 and tn + fp > 0:
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        roc_auc = evaluator.setMetricName("areaUnderROC").evaluate(predictions)
        pr_auc = evaluator.setMetricName("areaUnderPR").evaluate(predictions)

    metrics = {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }
    confusion = {"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    return metrics, confusion


def threshold_metrics(predictions, threshold, name):
    scored = predictions.withColumn(
        "prediction",
        (vector_to_array("probability")[1] >= F.lit(float(threshold))).cast("double"),
    )
    metrics, confusion = classifier_metrics(scored, name)
    return metrics, confusion


def run_classifiers(train, validation, test):
    section("HEAVY-RAINFALL CLASSIFICATION")
    train = train.withColumn("label", (F.col("next_rain") >= HEAVY_MM).cast("double"))
    validation = validation.withColumn("label", (F.col("next_rain") >= HEAVY_MM).cast("double"))
    test = test.withColumn("label", (F.col("next_rain") >= HEAVY_MM).cast("double"))

    train_counts = {int(r["label"]): r["count"] for r in train.groupBy("label").count().collect()}
    valid_counts = {int(r["label"]): r["count"] for r in validation.groupBy("label").count().collect()}
    test_counts = {int(r["label"]): r["count"] for r in test.groupBy("label").count().collect()}
    print(f"Heavy-rain threshold: {HEAVY_MM} mm")
    print(f"Training class counts: {train_counts}")
    print(f"Validation class counts: {valid_counts}")
    print(f"Test class counts: {test_counts}")

    if train_counts.get(0, 0) == 0 or train_counts.get(1, 0) == 0:
        log("Training data has only one class; skipping classification.")
        return None

    negatives, positives = train_counts.get(0, 0), train_counts.get(1, 0)
    weight = negatives / max(positives, 1)
    train = train.withColumn(
        "class_weight",
        F.when(F.col("label") == 1.0, F.lit(weight)).otherwise(F.lit(1.0)),
    )

    candidates = {
        "RandomForestClassifier": RandomForestClassifier(
            featuresCol="features", labelCol="label", weightCol="class_weight",
            numTrees=200, maxDepth=8, maxBins=64, seed=SEED,
        ),
        "GBTClassifier": GBTClassifier(
            featuresCol="features", labelCol="label", weightCol="class_weight",
            maxIter=100, maxDepth=5, stepSize=0.05, seed=SEED,
        ),
        "LogisticRegression": LogisticRegression(
            featuresCol="features", labelCol="label", weightCol="class_weight",
            maxIter=100, regParam=0.01, elasticNetParam=0.0,
        ),
    }

    models, val_thresholds, rows = {}, {}, []
    for name, estimator in candidates.items():
        log(f"Fitting {name}")
        model = Pipeline(stages=preprocessing_stages() + [estimator]).fit(train)
        valid_pred = model.transform(validation).cache()

        # Select threshold on validation data by F1. The final test set is untouched.
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
            metrics, _ = threshold_metrics(valid_pred, threshold, name)
            if metrics["f1"] > best_f1:
                best_f1, best_threshold = metrics["f1"], threshold

        val_metrics, _ = threshold_metrics(valid_pred, best_threshold, name)
        val_thresholds[name] = {"threshold": best_threshold, "f1": val_metrics["f1"]}
        models[name] = model
        for metric, value in val_metrics.items():
            rows.append({"task": "classification", "model": name, "split": "validation", "threshold": best_threshold, "metric": metric, "value": value})
        print(f"{name} validation threshold={best_threshold:.2f}; metrics={val_metrics}")
        valid_pred.unpersist()

    # Select best classifier by validation F1. Test metrics are only reported after selection.
    winner = max(val_thresholds, key=lambda name: val_thresholds[name]["f1"])
    threshold = val_thresholds[winner]["threshold"]
    log(f"Selected classifier by validation F1: {winner}; threshold={threshold:.2f}")

    test_raw = models[winner].transform(test).cache()
    test_pred = test_raw.withColumn(
        "prediction",
        (vector_to_array("probability")[1] >= F.lit(float(threshold))).cast("double"),
    ).cache()
    test_metrics, confusion = classifier_metrics(test_pred, winner)
    print(f"\nTEST {winner} at threshold {threshold:.2f}: {test_metrics}")
    print(f"Confusion matrix: {confusion}")
    for metric, value in test_metrics.items():
        rows.append({"task": "classification", "model": winner, "split": "test", "threshold": threshold, "metric": metric, "value": value})

    # Save predictions and confusion matrix for the selected classifier.
    confusion_rows = [confusion]
    return winner, models[winner], test_pred, rows, confusion_rows, threshold, models


def export_predictions(reg_predictions, classifier_result):
    section("EXPORTING PREDICTIONS")
    regression_output = reg_predictions.select(
        "station", "district",
        F.col("date").alias("observation_date"),
        "target_date",
        F.round("next_rain", 2).alias("actual_rain_mm"),
        F.round("pred_rain", 2).alias("predicted_rain_mm"),
    )
    write_spark_csv(regression_output, "next_day_predictions")

    if not classifier_result:
        return
    _, _, predictions, _, _, threshold, _ = classifier_result
    output = predictions.select(
        "station", "district",
        F.col("date").alias("observation_date"),
        "target_date",
        F.round("next_rain", 2).alias("actual_rain_mm"),
        F.col("label").cast("int").alias("actual_heavy"),
        F.col("prediction").cast("int").alias("predicted_heavy"),
        F.round(vector_to_array("probability")[1], 4).alias("heavy_rain_probability"),
        F.lit(float(threshold)).alias("classification_threshold"),
    )
    write_spark_csv(output, "heavy_rainfall_predictions")


def export_latest_forecasts(features, regression_model, classifier_result):
    latest_window = Window.partitionBy("station").orderBy(F.col("day_num").desc())
    latest = (
        features.withColumn("row_num", F.row_number().over(latest_window))
        .filter(F.col("row_num") == 1).drop("row_num")
    )
    output = regression_model.transform(latest).select(
        "station", "district",
        F.col("date").alias("last_observation_date"),
        "target_date",
        F.round("pred_rain", 2).alias("predicted_rain_mm"),
    )
    if classifier_result:
        _, classifier_model, _, _, _, threshold, _ = classifier_result
        classified = classifier_model.transform(latest).select(
            "station",
            F.round(vector_to_array("probability")[1], 4).alias("heavy_rain_probability"),
        ).withColumn(
            "predicted_heavy_rain",
            (F.col("heavy_rain_probability") >= F.lit(float(threshold))).cast("int"),
        )
        output = output.join(classified, "station", "left")
    log("Latest outputs are historical demonstration forecasts, not live weather forecasts.")
    write_spark_csv(output, "latest_station_forecasts")


def set_chart_style():
    sns.set_theme(style="whitegrid")


def save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved chart: {filename}")


def safe_chart(function, *args):
    try:
        function(*args)
    except Exception as exc:
        log(f"Chart {function.__name__} failed: {type(exc).__name__}: {exc}")


def chart_district(results):
    data = results["district_summary"].sort_values("avg_recorded_rain_mm")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(data["district"], data["avg_recorded_rain_mm"], color=COLORS["blue"])
    ax.set_xlabel("Average rainfall per recorded station-day (mm)")
    ax.set_title("Recorded rainfall by district")
    save_fig(fig, "district_rainfall_comparison.png")


def chart_monthly(results):
    data = results["monthly_climatology"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(data["month"], data["avg_recorded_rain_mm"], color=COLORS["blue"])
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Average recorded rainfall (mm)")
    ax.set_title("Monthly rainfall pattern")
    save_fig(fig, "monthly_rainfall_pattern.png")


def chart_trend(results):
    data = results["monthly_rainfall"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(data["year_month"], data["avg_recorded_rain_mm"], marker="o", color=COLORS["blue"])
    ax.set_title("Monthly average recorded rainfall")
    ax.set_ylabel("Rainfall (mm)")
    ax.tick_params(axis="x", rotation=60)
    save_fig(fig, "rainfall_trend_over_time.png")


def chart_distribution(df):
    values = df.select("rain_mm").toPandas()["rain_mm"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=60, color=COLORS["blue"])
    ax.set_yscale("log")
    ax.axvline(HEAVY_MM, color=COLORS["red"], label=f"{HEAVY_MM} mm threshold")
    ax.set_xlabel("Recorded daily rainfall (mm)")
    ax.set_ylabel("Number of observations (log scale)")
    ax.set_title("Rainfall distribution")
    ax.legend()
    save_fig(fig, "rainfall_distribution.png")


def chart_anomalies(anomalies):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    district = anomalies["by_district"].sort_values("count")
    axes[0].barh(district["district"], district["count"], color=COLORS["orange"])
    axes[0].set_title("Anomalies by district")
    axes[0].set_xlabel("Flagged observations")
    month = anomalies["by_month"]
    axes[1].bar(month["month"], month["count"], color=COLORS["orange"])
    axes[1].set_xticks(range(1, 13))
    axes[1].set_title("Anomalies by month")
    axes[1].set_xlabel("Month")
    save_fig(fig, "anomaly_counts.png")


def chart_actual_vs_predicted(predictions):
    data = predictions.select("next_rain", "pred_rain").toPandas()
    if data.empty:
        return
    if len(data) > 10000:
        data = data.sample(10000, random_state=SEED)
    maximum = max(float(data["next_rain"].max()), float(data["pred_rain"].max()), 1.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data["next_rain"], data["pred_rain"], s=8, alpha=0.3, color=COLORS["blue"])
    ax.plot([0, maximum], [0, maximum], color=COLORS["red"], label="Perfect prediction")
    ax.set_xlabel("Actual next-day rainfall (mm)")
    ax.set_ylabel("Predicted rainfall (mm)")
    ax.set_title("Actual vs predicted rainfall")
    ax.legend()
    save_fig(fig, "actual_vs_predicted.png")


def chart_confusion(confusion_matrices):
    if not confusion_matrices:
        return
    fig, axes = plt.subplots(1, len(confusion_matrices), figsize=(6, 5), squeeze=False)
    for ax, cm in zip(axes[0], confusion_matrices):
        matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Not heavy", "Heavy"], yticklabels=["Not heavy", "Heavy"],
        )
        ax.set_title(cm["model"])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
    save_fig(fig, "confusion_matrices.png")


def chart_model_comparison(metrics_df):
    if metrics_df.empty:
        return
    classifiers = metrics_df[
        (metrics_df["task"] == "classification")
        & (metrics_df["split"] == "test")
    ]
    if not classifiers.empty:
        selected = classifiers[classifiers["metric"].isin(["precision", "recall", "f1", "roc_auc", "pr_auc"])]
        if not selected.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=selected, x="metric", y="value", hue="model", ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title("Selected heavy-rainfall classifier: test metrics")
            save_fig(fig, "classifier_comparison.png")

    regression = metrics_df[
        (metrics_df["task"] == "regression")
        & (metrics_df["split"] == "test")
        & (metrics_df["metric"].isin(["mae", "rmse"]))
    ]
    if not regression.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=regression, x="metric", y="value", hue="model", ax=ax)
        ax.set_title("Regression test error comparison (lower is better)")
        save_fig(fig, "regression_comparison.png")


def chart_importance(importance):
    if importance is None or importance.empty:
        return
    data = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data["feature"], data["importance"], color=COLORS["violet"])
    ax.set_title("Selected regression model feature importance")
    save_fig(fig, "rf_feature_importance.png")


def run_ml(features):
    section("MACHINE-LEARNING PIPELINE")
    train, validation, test, train_cut, val_cut = split_chronologically(features)

    reg_winner, reg_model, reg_predictions, reg_rows, importance, reg_models = run_regression(
        train, validation, test
    )
    classifier_result = run_classifiers(train, validation, test)

    rows = list(reg_rows)
    confusion_matrices = []
    if classifier_result:
        _, classifier_model, _, classifier_rows, confusion_matrices, threshold, classifier_models = classifier_result
        rows.extend(classifier_rows)

    metrics_df = pd.DataFrame(rows)
    metrics_df["train_cut_target_day_num"] = train_cut
    metrics_df["validation_cut_target_day_num"] = val_cut
    metrics_df.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    log(f"Saved {OUT_DIR / 'model_metrics.csv'}")

    if confusion_matrices:
        pd.DataFrame(confusion_matrices).to_csv(OUT_DIR / "confusion_matrices.csv", index=False)

    export_predictions(reg_predictions, classifier_result)
    export_latest_forecasts(features, reg_model, classifier_result)

    save_model(reg_model, "selected_regressor_next_day_rainfall")
    for name, model in reg_models.items():
        save_model(model, name.lower())
    if classifier_result:
        save_model(classifier_result[1], "selected_classifier_heavy_rainfall")
        for name, model in classifier_models.items():
            save_model(model, name.lower())

    section("MACHINE-LEARNING CHARTS")
    safe_chart(chart_actual_vs_predicted, reg_predictions)
    safe_chart(chart_importance, importance)
    safe_chart(chart_confusion, confusion_matrices)
    safe_chart(chart_model_comparison, metrics_df)

    reg_predictions.unpersist()
    for frame in (train, validation, test):
        frame.unpersist()


def main():
    parser = argparse.ArgumentParser(description="RainfallWatch Kerala - Apache Spark project")
    parser.add_argument(
        "--skip-ml", action="store_true",
        help="Run data analysis and anomaly detection without model training",
    )
    args = parser.parse_args()
    started = time.time()
    spark = None
    make_dirs()
    set_chart_style()

    try:
        spark = build_spark()
        section("MODULE 1: DATA LOADING AND QUALITY")
        raw = load_raw(spark)
        prepared = prepare(raw).cache()
        data_quality(prepared)

        section("MODULE 2: DATA CLEANING")
        cleaned = clean_data(prepared)

        section("MODULE 3: SPARK SQL ANALYSIS")
        results = sql_analysis(spark, cleaned)

        section("MODULE 4: ANOMALY DETECTION")
        anomalies = detect_anomalies(cleaned)

        section("EDA CHARTS")
        safe_chart(chart_district, results)
        safe_chart(chart_monthly, results)
        safe_chart(chart_trend, results)
        safe_chart(chart_distribution, cleaned)
        safe_chart(chart_anomalies, anomalies)

        if args.skip_ml:
            log("ML skipped by request.")
        else:
            features = engineer_features(cleaned)
            try:
                run_ml(features)
            finally:
                features.unpersist()

        cleaned.unpersist()
        prepared.unpersist()
        log("Pipeline completed.")
    except Exception:
        log("Pipeline failed. Full error follows:")
        traceback.print_exc()
        raise
    finally:
        if spark is not None:
            spark.stop()
            log("Spark session stopped.")
        log(f"Elapsed time: {time.time() - started:.1f} seconds. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
