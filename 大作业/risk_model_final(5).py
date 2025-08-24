import os
import gc
from glob import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, ClassifierMixin

import lightgbm as lgb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class WeightedVotingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, weights=None):
        super().__init__()
        self.estimators = estimators
        # 如果没有提供weights，则默认所有模型权重相同
        self.weights = weights if weights is not None else [1.0] * len(estimators)
        self.weights = np.array(self.weights) / np.sum(self.weights)
        
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        y_preds = np.array([estimator.predict(X) for estimator in self.estimators])
        # 每个样本的预测结果加权平均
        weighted_preds = np.average(y_preds, axis=0, weights=self.weights)
        return weighted_preds
    
    def predict_proba(self, X):
        proba_preds = np.array([estimator.predict_proba(X) for estimator in self.estimators])
        # 每个样本的概率预测加权平均
        weighted_proba = np.average(proba_preds, axis=0, weights=self.weights)
        return weighted_proba

class Pipeline:
    @staticmethod
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))            

        return df
    
    @staticmethod
    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
                df = df.with_columns(pl.col(col).cast(pl.Float32))
                
        df = df.drop("date_decision", "MONTH")

        return df

    @staticmethod
    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()

                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df

class Aggregator:
    @staticmethod
    def get_exprs(df):
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]

        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols]

        cols2 = [col for col in df.columns if col[-1] in ("L", "A")]
        expr_2 = [pl.mean(col).alias(f"mean_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2] 
        return expr_1 + expr_2
    @staticmethod
    def deposit_exprs(df):
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols] 

        return expr_1 

    @staticmethod
    def debitcard_exprs(df):
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols] + [pl.min(col).alias(f"min_{col}") for col in cols] 
        
        return expr_1 


    @staticmethod
    def person_expr(df):
        cols1 = ['empl_employedtotal_800L', 'empl_employedfrom_271D', 'empl_industry_691L', 
                 'familystate_447L', 'incometype_1044T', 'sex_738L', 'housetype_905L', 'housingtype_772L',
                 'isreference_387L', 'birth_259D', ]
        expr_1 = [pl.first(col).alias(f"first_{col}") for col in cols1]
        
        expr_2 = [pl.col("mainoccupationinc_384A").max().alias("mainoccupationinc_384A_max"), 
                  pl.col("mainoccupationinc_384A").filter(pl.col("incometype_1044T") == "SELFEMPLOYED").max().alias("mainoccupationinc_384A_any_selfemployed")]
        
        # No Effect ...
        # cols = ['personindex_1023L', 'persontype_1072L', 'persontype_792L']
        # expr_3 = [pl.col(col).last().alias(f"last_{col}") for col in cols] + [pl.col(col).drop_nulls().mean().alias(f"mean_{col}") for col in cols]

        # cols2 = [col for col in df.columns if col not in cols1]
        # expr_4 = [pl.max(col).alias(f"max_{col}") for col in cols2] + [pl.min(col).alias(f"min_{col}") for col in cols2] #  good at cv, bad at lb ?
            # [pl.col(col).drop_nulls().last().alias(f"last_{col}") for col in cols2] + [pl.col(col).drop_nulls().first().alias(f"first_{col}") for col in cols2] # no effect

        return expr_1 + expr_2 # + expr_4 # + expr_3
    
    @staticmethod
    def bureau_a1(df):
        cols = [col for col in df.columns if (col[-1] in ("T","L","M","D","P","A")) or ("num_group" in col)]
        expr_1 = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_2 = [pl.min(col).alias(f"min_{col}") for col in cols]

        cols2 = [
        'annualeffectiverate_199L', 'annualeffectiverate_63L',
        'contractsum_5085717L', 
        'credlmt_230A', 'credlmt_935A',
        'nominalrate_281L', 'nominalrate_498L',
       'numberofcontrsvalue_258L', 'numberofcontrsvalue_358L',
       'numberofinstls_229L', 'numberofinstls_320L',
       'numberofoutstandinstls_520L', 'numberofoutstandinstls_59L',
       'numberofoverdueinstlmax_1039L', 'numberofoverdueinstlmax_1151L',
       'numberofoverdueinstls_725L', 'numberofoverdueinstls_834L',
       ]

        expr_3 = [pl.mean(col).alias(f"mean_{col}") for col in cols2] + [pl.std(col).alias(f"std_{col}") for col in cols2] + \
            [pl.sum(col).alias(f"sum_{col}") for col in cols2] + [pl.median(col).alias(f"median_{col}") for col in cols2] + \
            [pl.first(col).alias(f"first_{col}") for col in cols2]
        
        return expr_1 + expr_2 + expr_3    


def agg_by_case(path, df):
    path = str(path)
    if '_applprev_1' in path:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    elif '_credit_bureau_a_1' in path:
        df = df.group_by("case_id").agg(Aggregator.bureau_a1(df))
    elif '_deposit_1' in path:
        df = df.group_by("case_id").agg(Aggregator.deposit_exprs(df))
    elif '_debitcard_1' in path:
        df = df.group_by("case_id").agg(Aggregator.debitcard_exprs(df))
    elif '_person_1' in path:
        df = df.group_by("case_id").agg(Aggregator.person_expr(df))
    else :
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
    return df

def read_file(path, depth=None): 
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    
    if depth in [1, 2]:
        df = agg_by_case(path, df)
    
    return df

def read_files(regex_path, depth=None):
    print(regex_path)
    chunks = []
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = agg_by_case(path, df)
        chunks.append(df)
        
    df = pl.concat(chunks, how="vertical_relaxed")
    df = df.unique(subset=["case_id"])
    
    return df

def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = (
        df_base.with_columns(
            decision_month = pl.col("date_decision").dt.month(),
            decision_weekday = pl.col("date_decision").dt.weekday(),
        )
    )
        
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
        
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base

def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    print(df_data.info())
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    
    return df_data, cat_cols

ROOT            = Path("F:/risk_model/home-credit-credit-risk-model-stability")
TRAIN_DIR       = ROOT / "parquet_files" / "train"
TEST_DIR        = ROOT / "parquet_files" / "test"

data_store = {
    "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
    "depth_0": [
        read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
        read_files(TRAIN_DIR / "train_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
        read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_other_1.parquet", 1),
        read_file(TRAIN_DIR / "train_person_1.parquet", 1),
        read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
        read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
        read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),
    ]
}
df_train = feature_eng(**data_store)

print("train data shape:\t", df_train.shape)

data_store = {
    "df_base": read_file(TEST_DIR / "test_base.parquet"),
    "depth_0": [
        read_file(TEST_DIR / "test_static_cb_0.parquet"),
        read_files(TEST_DIR / "test_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TEST_DIR / "test_applprev_1_*.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_a_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_b_1.parquet", 1),
        read_file(TEST_DIR / "test_tax_registry_c_1.parquet", 1),
        read_files(TEST_DIR / "test_credit_bureau_a_1_*.parquet", 1),
        read_file(TEST_DIR / "test_credit_bureau_b_1.parquet", 1),
        read_file(TEST_DIR / "test_other_1.parquet", 1),
        read_file(TEST_DIR / "test_person_1.parquet", 1),
        read_file(TEST_DIR / "test_deposit_1.parquet", 1),
        read_file(TEST_DIR / "test_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TEST_DIR / "test_credit_bureau_b_2.parquet", 2),
        read_files(TEST_DIR / "test_credit_bureau_a_2_*.parquet", 2),
    ]
}
df_test = feature_eng(**data_store)

print("test data shape:\t", df_test.shape)

df_train = df_train.pipe(Pipeline.filter_cols)
df_test = df_test.select([col for col in df_train.columns if col != "target"])

print("train data shape:\t", df_train.shape)
print("test data shape:\t", df_test.shape)

df_train, cat_cols = to_pandas(df_train)
df_test, cat_cols = to_pandas(df_test, cat_cols)

del data_store

gc.collect()

print("Train is duplicated:\t", df_train["case_id"].duplicated().any())
print("Train Week Range:\t", (df_train["WEEK_NUM"].min(), df_train["WEEK_NUM"].max()))

print()

print("Test is duplicated:\t", df_test["case_id"].duplicated().any())
print("Test Week Range:\t", (df_test["WEEK_NUM"].min(), df_test["WEEK_NUM"].max()))

sns.lineplot(
    data=df_train,
    x="WEEK_NUM",
    y="target",
)
plt.show()


y = df_train["target"]

from sklearn.model_selection import train_test_split

sample_size = 100000  # 目标样本量
df_train_simple, _, y, _ = train_test_split(
    df_train, y, 
    train_size=sample_size/len(df_train),
    stratify=y,  # 保持正负样本比例
    random_state=42
)
print(f"减少到 {len(df_train)} 条")

del df_train

gc.collect()

X = df_train_simple.drop(columns=["target", "case_id", "WEEK_NUM"])
y = df_train_simple["target"]
weeks = df_train_simple["WEEK_NUM"]



# 检查非数值列
non_numeric_cols = X.select_dtypes(exclude=['number']).columns

# 对分类变量编码
from sklearn.preprocessing import LabelEncoder

for col in non_numeric_cols:
    X[col] = LabelEncoder().fit_transform(X[col])

cv = StratifiedGroupKFold(n_splits=4, shuffle=False)

lgb_params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "colsample_bytree": 0.8, 
    "colsample_bynode": 0.8,
    "verbose": -1,
    "random_state": 42,
    "device_type": "gpu",
}


import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

class StandardizedNN:
    def __init__(self, pipeline):
        self.pipeline = pipeline 

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)  # pipeline会处理所有预处理步骤

nn_params = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 256,
    "learning_rate_init": 0.001,
    "max_iter": 50,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "random_state": 42,
    "verbose": True,
}

from xgboost import XGBClassifier

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 30,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist'  
}

fitted_models = []

for idx_train, idx_valid in cv.split(X, y, groups=weeks):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    
    # --- 训练 LightGBM ---
    print("--- 开始训练 LightGBM ---")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.log_evaluation(30), lgb.early_stopping(50)]
    )
    fitted_models.append(lgb_model)
    print("--- LightGBM 训练结束---")
    
    # --- 训练神经网络 ---
    print("--- 开始训练神经网络 ---")
    nn_pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler(),
        MLPClassifier(**nn_params)
    )
    nn_pipeline.fit(X_train, y_train)
    fitted_models.append(StandardizedNN(nn_pipeline))
    print("--- 神经网络训练结束 ---")
    
    # --- 训练XGBoost ---
    print("--- 开始训练 XGBoost ---")
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=10
    )
    fitted_models.append(xgb_model)
    print("--- XGBoost 训练结束---")
    
n_folds = cv.get_n_splits()
weights = []
for i in range(n_folds):
    weights.extend([0.65, 0.2, 0.15])  # 每个fold中模型的权重

model = WeightedVotingModel(fitted_models, weights=weights)
X_test = df_test.drop(columns=["WEEK_NUM"])
X_test = X_test.set_index("case_id")

non_numeric_cols = X_test.select_dtypes(exclude=['number']).columns

# 对分类变量编码（如LabelEncoder/OneHotEncoder）
from sklearn.preprocessing import LabelEncoder

for col in non_numeric_cols:
    X_test[col] = LabelEncoder().fit_transform(X_test[col])

y_pred = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

df_subm = pd.read_csv(ROOT / "sample_submission.csv")
df_subm = df_subm.set_index("case_id")

df_subm["score"] = y_pred
print("Check null: ", df_subm["score"].isnull().any())

df_subm.head()
df_subm.to_csv("submission.csv")