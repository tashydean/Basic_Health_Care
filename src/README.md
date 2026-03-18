# src/

노트북에서 검증된 핵심 로직을 재사용 가능한 Python 모듈로 분리한 폴더입니다.

```
src/
├── feature_engineering.py    # 데이터 로드 → 파생변수 생성 → 통합 데이터셋 저장
└── modeling.py               # 모델 학습 → 평가 → 시각화 (SHAP 포함)
```

---

## feature_engineering.py

| 함수 | 역할 |
|------|------|
| `load_raw_tables()` | 원본 4개 테이블 로드 |
| `compute_time_features()` | 입원 날짜 기반 파생 변수 생성 |
| `compute_hours_after_admission()` | 입원 후 검사 경과 시간 계산 |
| `compute_trend()` | 선형회귀 기울기로 Trend 계산 |
| `aggregate_lab_features()` | 검사 로우 → 입원 단위 집계 |
| `compute_patient_level_features()` | 입원 단위 → 환자 단위 종합 지표 |
| `build_final_dataset()` | 전체 파이프라인 실행 |
| `save_processed()` | 결과 CSV 저장 |

## modeling.py

| 함수 | 역할 |
|------|------|
| `prepare_data()` | 피처 선택 + Train/Test 분할 |
| `train_linear_regression()` | Baseline 모델 학습 |
| `train_random_forest()` | GridSearchCV 최적화 학습 |
| `train_xgboost()` | Challenger 모델 학습 |
| `evaluate_model()` | R² / MAE / RMSE 계산 |
| `compare_models()` | 전체 모델 비교표 생성 (CV 포함) |
| `plot_model_comparison()` | 성능 비교 막대 차트 |
| `plot_actual_vs_predicted()` | Actual vs Predicted 산점도 |
| `plot_shap_summary()` | SHAP Summary Plot |

## 실행 예시

```python
from src.feature_engineering import load_raw_tables, build_final_dataset, save_processed
from src.modeling import prepare_data, train_random_forest, compare_models, plot_shap_summary

# 1. 전처리
tables = load_raw_tables("data/raw")
df     = build_final_dataset(tables)
save_processed(df)

# 2. 모델링
X_train, X_test, y_train, y_test, _ = prepare_data(df)
rf = train_random_forest(X_train, y_train)

# 3. XAI
plot_shap_summary(rf, X_test, save_path="outputs/figures/shap_summary.png")
```
