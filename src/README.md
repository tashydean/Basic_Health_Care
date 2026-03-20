# src/

노트북에서 검증된 핵심 로직을 재사용 가능한 Python 모듈로 분리한 폴더입니다.

```
src/
├── feature_engineering.py    # 데이터 로드 → 파생변수 생성 → 통합 데이터셋 저장
└── modeling.py               # 모델 학습 → 평가 → SHAP 시각화
```

## feature_engineering.py

| 함수 | 역할 |
|------|------|
| `load_raw_tables()` | 원본 4개 테이블 로드 |
| `compute_time_features()` | 입원 날짜 기반 파생변수 |
| `compute_hours_after_admission()` | 입원 후 검사 경과 시간 |
| `compute_trend()` | 선형회귀 기울기로 Trend 계산 |
| `aggregate_lab_features()` | 검사 로우 → 입원 단위 집계 |
| `compute_patient_level_features()` | 환자 단위 종합 지표 |
| `build_final_dataset()` | 전체 파이프라인 실행 |
| `save_processed()` | 결과 CSV 저장 |

## modeling.py

| 함수 | 역할 |
|------|------|
| `prepare_data()` | 피처 선택(LabTestCount 제외) + Train/Test 분할 |
| `train_linear_regression()` | Baseline 학습 |
| `train_random_forest()` | GridSearchCV 최적화 학습 |
| `train_xgboost()` | Challenger 모델 학습 |
| `evaluate_model()` | R² / MAE / RMSE 계산 |
| `compare_models()` | 전체 모델 비교표 + CV |
| `plot_shap_summary()` | SHAP Summary Plot |

## 실행 예시

```python
from src.feature_engineering import load_raw_tables, build_final_dataset, save_processed
from src.modeling import prepare_data, train_xgboost, plot_shap_summary

tables = load_raw_tables("data/")
df     = build_final_dataset(tables)
save_processed(df)

X_train, X_test, y_train, y_test, _ = prepare_data(df)
xgb = train_xgboost(X_train, y_train)
plot_shap_summary(xgb, X_test, save_path="outputs/figures/shap_summary.png")
```
