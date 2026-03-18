"""
modeling.py
===========
입원 기간(LOS) 예측 모델 학습, 평가, 해석(SHAP)을 수행하는 모듈입니다.

모델 구성
---------
- Baseline  : Linear Regression
- Main      : Random Forest + GridSearchCV
- Challenger: XGBoost
- 검증      : 5-Fold Cross Validation
- XAI       : SHAP TreeExplainer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.family"]       = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 1. 피처 선택 및 데이터 분할
# ---------------------------------------------------------------------------

FEATURE_KEYWORDS = ["Mean", "Std", "Trend", "Count", "Variability"]


def prepare_data(df: pd.DataFrame,
                 target: str = "LengthOfStay",
                 test_size: float = 0.2,
                 random_state: int = 42):
    """
    피처 선택 및 Train / Test 분할을 수행합니다.

    Parameters
    ----------
    df           : 최종 통합 데이터셋
    target       : 타겟 컬럼명
    test_size    : 테스트 비율 (기본 0.2)
    random_state : 재현성 시드

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names
    """
    features = [c for c in df.columns
                if any(kw in c for kw in FEATURE_KEYWORDS)]
    X = df[features].fillna(0)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"✅ 피처 수: {len(features)}")
    print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, features


# ---------------------------------------------------------------------------
# 2. 모델 학습
# ---------------------------------------------------------------------------

def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Linear Regression 기준 모델을 학습합니다."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Linear Regression 학습 완료")
    return model


def train_random_forest(X_train, y_train,
                        param_grid: dict = None,
                        cv: int = 3) -> RandomForestRegressor:
    """
    Random Forest + GridSearchCV로 최적 모델을 탐색합니다.

    Parameters
    ----------
    param_grid : 탐색할 하이퍼파라미터 그리드
                 None이면 기본 그리드 사용
    cv         : 교차 검증 fold 수
    """
    if param_grid is None:
        param_grid = {
            "n_estimators":     [100, 200],
            "max_depth":        [10, 20],
            "min_samples_split": [2, 5],
        }

    print("⚙️  Random Forest GridSearchCV 탐색 중...")
    search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"✅ 최적 파라미터: {search.best_params_}")
    return search.best_estimator_


def train_xgboost(X_train, y_train,
                  n_estimators: int = 100,
                  learning_rate: float = 0.1,
                  max_depth: int = 5) -> XGBRegressor:
    """XGBoost Challenger 모델을 학습합니다."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    print("✅ XGBoost 학습 완료")
    return model


# ---------------------------------------------------------------------------
# 3. 평가
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str = "") -> dict:
    """
    단일 모델의 Test Set 성능을 계산합니다.

    Returns
    -------
    dict : {"Model", "R²", "MAE", "RMSE"}
    """
    pred = model.predict(X_test)
    return {
        "Model": model_name,
        "R²":    round(r2_score(y_test, pred), 4),
        "MAE":   round(mean_absolute_error(y_test, pred), 4),
        "RMSE":  round(np.sqrt(mean_squared_error(y_test, pred)), 4),
    }


def cross_validate_models(models: dict, X_train, y_train,
                           cv: int = 5) -> pd.DataFrame:
    """
    여러 모델에 대해 K-Fold CV R² Score를 계산합니다.

    Parameters
    ----------
    models : {"모델명": sklearn_model, ...}
    cv     : fold 수

    Returns
    -------
    pd.DataFrame : 모델별 CV mean / std
    """
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        rows.append({
            "Model":        name,
            "CV R² (mean)": round(scores.mean(), 4),
            "CV R² (std)":  round(scores.std(), 4),
        })
        print(f"   {name:30s} CV R²: {scores.mean():.3f} ± {scores.std():.3f}")
    return pd.DataFrame(rows)


def compare_models(models: dict, X_test, y_test,
                   X_train=None, y_train=None,
                   cv: int = 5) -> pd.DataFrame:
    """
    전체 모델 성능 비교표를 생성합니다 (Test 지표 + CV R²).

    Parameters
    ----------
    models  : {"모델명": sklearn_model, ...}
    cv      : CV fold 수 (X_train / y_train이 제공된 경우에만 수행)
    """
    test_rows = [evaluate_model(m, X_test, y_test, name)
                 for name, m in models.items()]
    result = pd.DataFrame(test_rows)

    if X_train is not None and y_train is not None:
        cv_df = cross_validate_models(models, X_train, y_train, cv=cv)
        result = result.merge(cv_df, on="Model", how="left")

    return result


# ---------------------------------------------------------------------------
# 4. 시각화
# ---------------------------------------------------------------------------

def plot_model_comparison(result_df: pd.DataFrame,
                          save_path: str = None) -> None:
    """모델별 R², MAE, RMSE 막대 비교 차트를 그립니다."""
    metrics = ["R²", "MAE", "RMSE"]
    colors  = ["#A8C8E8", "#FF8C69", "#90EE90"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, metric in enumerate(metrics):
        bars = axes[i].bar(result_df["Model"], result_df[metric], color=colors)
        axes[i].set_title(metric, fontsize=13, weight="bold")
        axes[i].set_xticklabels(result_df["Model"], rotation=15, ha="right")
        axes[i].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, result_df[metric]):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10,
            )

    plt.suptitle("모델별 성능 비교 (Test Set)", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 저장: {save_path}")
    plt.show()


def plot_actual_vs_predicted(models: dict, X_test, y_test,
                             save_path: str = None) -> None:
    """각 모델의 Actual vs Predicted 산점도를 나란히 그립니다."""
    colors = ["#A8C8E8", "#FF8C69", "#90EE90"]
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))

    if len(models) == 1:
        axes = [axes]

    for ax, (name, model), color in zip(axes, models.items(), colors):
        pred = model.predict(X_test)
        ax.scatter(y_test, pred, alpha=0.5, color=color, s=40)
        lim = [min(y_test.min(), pred.min()) - 1,
               max(y_test.max(), pred.max()) + 1]
        ax.plot(lim, lim, "r--", lw=1.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual LengthOfStay")
        ax.set_ylabel("Predicted LengthOfStay")
        ax.set_title(f"{name}\n(R² = {r2_score(y_test, pred):.3f})",
                     fontsize=11, weight="bold")
        ax.grid(alpha=0.3)

    plt.suptitle("Actual vs Predicted", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 저장: {save_path}")
    plt.show()


def plot_shap_summary(model, X_test,
                      save_path: str = None) -> None:
    """SHAP Summary Plot으로 변수 기여도를 시각화합니다."""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot", fontsize=13, weight="bold", pad=15)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ 저장: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("outputs/processed_healthcare_data.csv", encoding="utf-8-sig")

    X_train, X_test, y_train, y_test, features = prepare_data(df)

    lr  = train_linear_regression(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    models = {
        "Linear Regression":     lr,
        "Random Forest (Tuned)": rf,
        "XGBoost":               xgb,
    }

    print("\n📊 전체 모델 성능 비교")
    result = compare_models(models, X_test, y_test, X_train, y_train)
    print(result.to_string(index=False))

    plot_model_comparison(result,      save_path="outputs/figures/model_comparison.png")
    plot_actual_vs_predicted(models, X_test, y_test,
                             save_path="outputs/figures/actual_vs_predicted.png")
    plot_shap_summary(rf, X_test,      save_path="outputs/figures/shap_summary.png")
