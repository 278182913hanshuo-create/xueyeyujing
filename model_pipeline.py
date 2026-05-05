"""
学生当前 CGPA 预测流水线
=====================
功能：
  1. 使用 sklearn Pipeline + ColumnTransformer 进行无泄漏预处理
  2. RepeatedKFold 交叉验证（n_splits=5, n_repeats=3）
  3. 12 种模型 + RandomizedSearchCV 超参数调优
  4. 输出：控制台对比表、results.csv、四张 PNG 图表
  5. 消融实验：ablation_results.csv + ablation_r2.png

用法：
  python model_pipeline.py [--data DATA_PATH] [--output OUTPUT_DIR]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 可选依赖
# ──────────────────────────────────────────────
_OPTIONAL = {}

try:
    from xgboost import XGBRegressor
    _OPTIONAL["xgboost"] = True
except ImportError:
    _OPTIONAL["xgboost"] = False
    print("[警告] xgboost 未安装，将跳过 XGBRegressor。")

try:
    from lightgbm import LGBMRegressor
    _OPTIONAL["lightgbm"] = True
except ImportError:
    _OPTIONAL["lightgbm"] = False
    print("[警告] lightgbm 未安装，将跳过 LGBMRegressor。")

try:
    from catboost import CatBoostRegressor
    _OPTIONAL["catboost"] = True
except ImportError:
    _OPTIONAL["catboost"] = False
    print("[警告] catboost 未安装，将跳过 CatBoostRegressor。")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform, loguniform

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

# ──────────────────────────────────────────────
# 1. 数据加载与预处理
# ──────────────────────────────────────────────

def load_data(data_path: str):
    """加载 CSV，检测目标列（优先包含 'current cgpa' 的列），返回 X, y。"""
    df = pd.read_csv(data_path, low_memory=False)
    print(f"[数据] 加载成功：{df.shape[0]} 行 × {df.shape[1]} 列")

    # 检测目标列
    target_col = None
    for col in df.columns:
        if "current cgpa" in col.lower():
            target_col = col
            break
    if target_col is None:
        for col in df.columns:
            if "cgpa" in col.lower():
                target_col = col
                break
    if target_col is None:
        raise ValueError(
            "未能自动检测到目标列（需包含 'cgpa'）。"
            f"当前列名：{list(df.columns)}"
        )
    print(f"[数据] 目标列：'{target_col}'")

    # 目标列强制转数值
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).copy()

    y = df[target_col].values.astype(float)
    X = df.drop(columns=[target_col])

    # 识别数值列与类别列
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"[数据] 数值特征 {len(num_cols)} 个，类别特征 {len(cat_cols)} 个")

    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    """构建 ColumnTransformer：数值 → StandardScaler，类别 → OneHotEncoder。"""
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )
    return ColumnTransformer(transformers=transformers)


# ──────────────────────────────────────────────
# 2. 模型定义与超参数搜索空间
# ──────────────────────────────────────────────

def get_model_configs():
    """返回 list of (name, estimator, param_dist)。"""
    configs = [
        (
            "LinearRegression",
            LinearRegression(),
            {},
        ),
        (
            "Ridge",
            Ridge(random_state=42),
            {"model__alpha": loguniform(1e-3, 1e3)},
        ),
        (
            "Lasso",
            Lasso(random_state=42, max_iter=5000),
            {"model__alpha": loguniform(1e-4, 10)},
        ),
        (
            "ElasticNet",
            ElasticNet(random_state=42, max_iter=5000),
            {
                "model__alpha": loguniform(1e-4, 10),
                "model__l1_ratio": uniform(0.1, 0.89),
            },
        ),
        (
            "SVR_RBF",
            SVR(kernel="rbf"),
            {
                "model__C": loguniform(0.1, 1000),
                "model__gamma": loguniform(1e-4, 1.0),
                "model__epsilon": loguniform(1e-3, 1.0),
            },
        ),
        (
            "RandomForest",
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": randint(100, 500),
                "model__max_depth": [None, 5, 10, 15, 20],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
            },
        ),
        (
            "ExtraTrees",
            ExtraTreesRegressor(random_state=42, n_jobs=-1),
            {
                "model__n_estimators": randint(100, 500),
                "model__max_depth": [None, 5, 10, 15, 20],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
            },
        ),
        (
            "GradientBoosting",
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": randint(100, 400),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(2, 7),
                "model__subsample": uniform(0.6, 0.4),
                "model__min_samples_leaf": randint(1, 10),
            },
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingRegressor(random_state=42),
            {
                "model__max_iter": randint(100, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": [None, 5, 8, 12],
                "model__min_samples_leaf": randint(10, 50),
                "model__l2_regularization": loguniform(1e-6, 10),
            },
        ),
    ]

    if _OPTIONAL["xgboost"]:
        configs.append((
            "XGBoost",
            # 使用 n_jobs=1 避免与外层 sklearn 并行产生嵌套并发问题
            XGBRegressor(random_state=42, n_jobs=1, verbosity=0, eval_metric="rmse"),
            {
                "model__n_estimators": randint(100, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(2, 8),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__reg_alpha": loguniform(1e-5, 10),
                "model__reg_lambda": loguniform(1e-5, 10),
            },
        ))

    if _OPTIONAL["lightgbm"]:
        configs.append((
            "LightGBM",
            # 使用 n_jobs=1 避免与外层 sklearn 并行产生嵌套并发问题
            LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1),
            {
                "model__n_estimators": randint(100, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__num_leaves": randint(15, 127),
                "model__max_depth": [-1, 5, 8, 12],
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__reg_alpha": loguniform(1e-5, 10),
                "model__reg_lambda": loguniform(1e-5, 10),
            },
        ))

    if _OPTIONAL["catboost"]:
        configs.append((
            "CatBoost",
            CatBoostRegressor(random_state=42, verbose=0),
            {
                "model__iterations": randint(100, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__depth": randint(3, 8),
                "model__l2_leaf_reg": loguniform(1e-3, 10),
                "model__subsample": uniform(0.6, 0.4),
            },
        ))

    return configs


# ──────────────────────────────────────────────
# 3. 评估（CV + RandomizedSearchCV）
# ──────────────────────────────────────────────

def rmse_array(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate_models(X, y, num_cols, cat_cols, n_iter_search=50, random_state=42):
    """
    对每个模型：
      - 先用 RandomizedSearchCV（5折，n_iter_search次）找最优参数
      - 再用 RepeatedKFold（5折×3次）在最优参数上估计泛化性能
    返回 results list
    """
    preprocessor = build_preprocessor(num_cols, cat_cols)
    cv_outer = RepeatedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    cv_inner = RepeatedKFold(n_splits=5, n_repeats=1, random_state=random_state)

    model_configs = get_model_configs()
    results = []

    for name, estimator, param_dist in model_configs:
        print(f"  → 训练 {name} ...", end=" ", flush=True)

        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])

        if param_dist:
            search = RandomizedSearchCV(
                pipe,
                param_dist,
                n_iter=n_iter_search,
                cv=cv_inner,
                scoring="r2",
                n_jobs=-1,
                random_state=random_state,
                refit=True,
            )
            search.fit(X, y)
            best_pipe = search.best_estimator_
            best_params = {
                k.replace("model__", ""): v
                for k, v in search.best_params_.items()
            }
        else:
            pipe.fit(X, y)
            best_pipe = pipe
            best_params = {}

        # 外层 CV 评估（n_jobs=1 避免嵌套并行死锁）
        r2_scores = cross_val_score(best_pipe, X, y, cv=cv_outer, scoring="r2", n_jobs=1)
        neg_rmse = cross_val_score(
            best_pipe, X, y, cv=cv_outer,
            scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        neg_mae = cross_val_score(
            best_pipe, X, y, cv=cv_outer,
            scoring="neg_mean_absolute_error", n_jobs=1,
        )
        # MedianAE 需要自定义 scorer
        from sklearn.metrics import make_scorer
        median_ae_scorer = make_scorer(median_absolute_error, greater_is_better=False)
        neg_medae = cross_val_score(
            best_pipe, X, y, cv=cv_outer, scoring=median_ae_scorer, n_jobs=1,
        )

        row = {
            "Model": name,
            "Best Params": json.dumps(
                {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                 for k, v in best_params.items()},
                ensure_ascii=False,
            ),
            "R2 mean": r2_scores.mean(),
            "R2 std": r2_scores.std(),
            "RMSE mean": (-neg_rmse).mean(),
            "RMSE std": (-neg_rmse).std(),
            "MAE mean": (-neg_mae).mean(),
            "MAE std": (-neg_mae).std(),
            "MedianAE mean": (-neg_medae).mean(),
            "MedianAE std": (-neg_medae).std(),
        }
        results.append(row)
        print(f"R²={row['R2 mean']:.4f}±{row['R2 std']:.4f}  "
              f"RMSE={row['RMSE mean']:.4f}±{row['RMSE std']:.4f}")

    return results, model_configs


# ──────────────────────────────────────────────
# 4. 输出：控制台表格 + results.csv
# ──────────────────────────────────────────────

def print_table(results):
    sorted_res = sorted(results, key=lambda r: r["R2 mean"], reverse=True)
    header = (
        f"{'模型':<25} {'R²':>12} {'RMSE':>14} {'MAE':>14} {'MedianAE':>14}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in sorted_res:
        print(
            f"{r['Model']:<25} "
            f"{r['R2 mean']:>6.4f}±{r['R2 std']:.4f} "
            f"{r['RMSE mean']:>6.4f}±{r['RMSE std']:.4f} "
            f"{r['MAE mean']:>6.4f}±{r['MAE std']:.4f} "
            f"{r['MedianAE mean']:>6.4f}±{r['MedianAE std']:.4f}"
        )
    print(sep + "\n")
    return sorted_res


def save_results_csv(results, output_dir):
    df = pd.DataFrame(results).sort_values("R2 mean", ascending=False)
    path = os.path.join(output_dir, "results.csv")
    df.to_csv(path, index=False)
    print(f"[保存] results.csv → {path}")


# ──────────────────────────────────────────────
# 5. 图表生成
# ──────────────────────────────────────────────

def _bar_plot(names, means, stds, ylabel, title, save_path, color):
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, color=color, alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[保存] {os.path.basename(save_path)} → {save_path}")


def save_bar_plots(sorted_results, output_dir):
    names = [r["Model"] for r in sorted_results]
    r2_means = [r["R2 mean"] for r in sorted_results]
    r2_stds = [r["R2 std"] for r in sorted_results]
    rmse_means = [r["RMSE mean"] for r in sorted_results]
    rmse_stds = [r["RMSE std"] for r in sorted_results]

    _bar_plot(
        names, r2_means, r2_stds,
        "Mean R²", "各模型交叉验证 R²（均值±标准差）",
        os.path.join(output_dir, "model_r2.png"), "#4C9BE8",
    )
    _bar_plot(
        names, rmse_means, rmse_stds,
        "Mean RMSE", "各模型交叉验证 RMSE（均值±标准差）",
        os.path.join(output_dir, "model_rmse.png"), "#E8784C",
    )


def save_best_model_plots(X, y, num_cols, cat_cols, best_result, model_configs, output_dir,
                          random_state=42):
    """用最优模型做最终 hold-out 评估，绘制预测 vs 真实 & 残差图。"""
    best_name = best_result["Model"]
    estimator = None
    best_params_json = json.loads(best_result["Best Params"])

    for name, est, _ in model_configs:
        if name == best_name:
            estimator = est.__class__(**{
                k: v for k, v in est.get_params().items()
            })
            break
    if estimator is None:
        return

    # 设置最优参数（类型要匹配）
    for k, v in best_params_json.items():
        try:
            estimator.set_params(**{k: v})
        except Exception:
            pass

    preprocessor = build_preprocessor(num_cols, cat_cols)
    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # 预测 vs 真实
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.7, edgecolors="k", s=60, color="#4C9BE8")
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="完美预测线")
    ax.set_xlabel("真实 CGPA")
    ax.set_ylabel("预测 CGPA")
    ax.set_title(f"最优模型（{best_name}）：预测 vs 真实")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "best_pred_vs_true.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[保存] best_pred_vs_true.png → {path}")

    # 残差图
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, residuals, alpha=0.7, edgecolors="k", s=60, color="#E8784C")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("预测值")
    ax.set_ylabel("残差（真实 − 预测）")
    ax.set_title(f"最优模型（{best_name}）：残差图")
    plt.tight_layout()
    path = os.path.join(output_dir, "best_residuals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[保存] best_residuals.png → {path}")


# ──────────────────────────────────────────────
# 6. 消融实验
# ──────────────────────────────────────────────

# 按主题分组的特征（精确列名，若列不存在则跳过整组）
ABLATION_GROUPS = {
    "学业成绩": ["Previous CGPA", "Exam Score (%)", "Assignments Completed (%)"],
    "学习行为": ["Study Hours Per Day", "Study Habit", "Attendance (%)"],
    "背景信息": ["Age", "Gender", "Department", "Family Income Level",
                 "Parental Education Level"],
    "健康/课外": ["Stress Level", "Sleep Quality", "Extracurricular Activities",
                  "Part-Time Job", "Internet Access"],
    "课程负担": ["Number of Courses"],
}


def ablation_study(X, y, num_cols, cat_cols, model_name, model_configs, output_dir,
                   random_state=42):
    """消融实验：逐一移除特征组，用 RepeatedKFold CV 评估影响。"""
    print("\n[消融实验] 开始...")

    # 找到消融用的模型（RF 或指定最优模型）
    ablation_est = None
    for name, est, _ in model_configs:
        if name == model_name:
            ablation_est = est
            break
    if ablation_est is None:
        print(f"[消融实验] 未找到模型 {model_name}，跳过。")
        return

    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=random_state)
    ablation_rows = []

    all_cols = list(X.columns)

    def _cv_score(cols_to_use):
        num_c = [c for c in num_cols if c in cols_to_use]
        cat_c = [c for c in cat_cols if c in cols_to_use]
        prep = build_preprocessor(num_c, cat_c)
        pipe = Pipeline([("prep", prep), ("model", ablation_est.__class__(
            **{k: v for k, v in ablation_est.get_params().items()}
        ))])
        scores = cross_val_score(pipe, X[cols_to_use], y, cv=cv, scoring="r2", n_jobs=1)
        return scores.mean(), scores.std()

    # 基线（所有特征）
    base_mean, base_std = _cv_score(all_cols)
    ablation_rows.append({"特征组": "基线（全特征）", "移除列": "无",
                           "R2 mean": base_mean, "R2 std": base_std,
                           "R2 下降": 0.0})
    print(f"  基线 R²={base_mean:.4f}±{base_std:.4f}")

    for group_name, group_cols in ABLATION_GROUPS.items():
        present = [c for c in group_cols if c in all_cols]
        if not present:
            print(f"  跳过 '{group_name}'（列不存在）")
            continue
        remaining = [c for c in all_cols if c not in present]
        if not remaining:
            continue
        m, s = _cv_score(remaining)
        drop = base_mean - m
        ablation_rows.append({
            "特征组": group_name,
            "移除列": ", ".join(present),
            "R2 mean": m,
            "R2 std": s,
            "R2 下降": drop,
        })
        print(f"  移除 '{group_name}' → R²={m:.4f}±{s:.4f}（下降 {drop:+.4f}）")

    # 保存 CSV
    ablation_df = pd.DataFrame(ablation_rows)
    path_csv = os.path.join(output_dir, "ablation_results.csv")
    ablation_df.to_csv(path_csv, index=False)
    print(f"[保存] ablation_results.csv → {path_csv}")

    # 绘图
    labels = [r["特征组"] for r in ablation_rows]
    means = [r["R2 mean"] for r in ablation_rows]
    stds = [r["R2 std"] for r in ablation_rows]
    colors = ["#2ecc71"] + ["#e74c3c" if r["R2 下降"] > 0 else "#3498db"
                             for r in ablation_rows[1:]]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean R²")
    ax.set_title(f"消融实验（{model_name}）：移除特征组后 R² 变化")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path_png = os.path.join(output_dir, "ablation_r2.png")
    plt.savefig(path_png, dpi=150)
    plt.close()
    print(f"[保存] ablation_r2.png → {path_png}")


# ──────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="学生 CGPA 预测流水线")
    parser.add_argument(
        "--data",
        default="Students_Performance_dataset.csv",
        help="CSV 数据文件路径（默认：Students_Performance_dataset.csv）",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="输出目录（默认：当前目录）",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=50,
        help="RandomizedSearchCV 迭代次数（默认：50）",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="随机种子（默认：42）",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1. 加载数据
    if not os.path.exists(args.data):
        print(f"[错误] 数据文件不存在：{args.data}")
        sys.exit(1)

    X, y, num_cols, cat_cols = load_data(args.data)

    # 2. 训练与评估
    print(f"\n[训练] 使用 RepeatedKFold(n_splits=5, n_repeats=3) + "
          f"RandomizedSearchCV(n_iter={args.n_iter})...")
    results, model_configs = evaluate_models(
        X, y, num_cols, cat_cols,
        n_iter_search=args.n_iter,
        random_state=args.random_state,
    )

    # 3. 输出表格
    sorted_results = print_table(results)

    # 4. 保存 CSV
    save_results_csv(results, args.output)

    # 5. 保存图表
    save_bar_plots(sorted_results, args.output)

    best_result = sorted_results[0]
    print(f"\n[最优模型] {best_result['Model']}  "
          f"R²={best_result['R2 mean']:.4f}±{best_result['R2 std']:.4f}")
    save_best_model_plots(
        X, y, num_cols, cat_cols, best_result, model_configs, args.output,
        random_state=args.random_state,
    )

    # 6. 消融实验（使用最优模型，若最优不在消融支持列表则退到 RandomForest）
    ablation_model = best_result["Model"]
    supported_ablation = {name for name, _, _ in model_configs}
    if ablation_model not in supported_ablation:
        ablation_model = "RandomForest"
    ablation_study(
        X, y, num_cols, cat_cols, ablation_model, model_configs, args.output,
        random_state=args.random_state,
    )

    print("\n✅ 全部流程完成！输出目录：", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
