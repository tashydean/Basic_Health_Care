# src/

노트북에서 검증된 핵심 로직을 재사용 가능한 Python 모듈로 분리한 폴더입니다.

```
src/
├── feature_engineering.py    # 데이터 로드 → 파생변수 생성 → 통합 데이터셋 저장
└── modeling.py               # 모델 학습 → 평가 → SHAP 시각화
```

> **주의:** 원본 데이터 파일은 `.txt` (탭 구분) 형식입니다.
> `load_raw_tables()`는 `data/raw/` 경로의 `.txt` 파일을 읽도록 설계되어 있습니다.

---

## feature_engineering.py

| 함수 | 역할 |
|------|------|
| `load_raw_tables(data_dir)` | 원본 4개 `.txt` 파일 로드 (`data/raw/` 기본) |
| `compute_time_features(admissions)` | 입원 날짜 기반 파생변수 (LOS, AgeGroup 등) |
| `compute_hours_after_admission(labs, admissions)` | 입원 후 검사 경과 시간 (Trend 계산 기반) |
| `aggregate_lab_features(labs)` | **검사명별 Wide Format 집계** — Lab_Mean/Std/Trend/Abnormal_Sum (35종 × 4 = 140개) |
| `compute_patient_level_features(labs, admissions)` | 환자 단위 종합 지표 (Total_Abnormal_Count, Total_Lab_Variability) |
| `build_final_dataset(tables)` | 전체 파이프라인 실행 → 372행 × 165열 |
| `save_processed(df, output_path)` | 결과 CSV 저장 (`data/processed/` 기본) |

**피처 설계 원칙:** 검사명(LabName) 35종별로 Mean/Std/Trend/Abnormal_Sum을 각각 생성하여
Wide Format으로 변환. `LabTestCount`는 역인과 위험으로 모델 피처에서 제외.

---

## modeling.py

| 함수 | 역할 |
|------|------|
| `prepare_data(df)` | 피처 선택(LabTestCount 제외) + Train/Test 분할 |
| `train_linear_regression(X_train, y_train)` | Baseline 학습 |
| `train_random_forest(X_train, y_train)` | GridSearchCV 최적화 학습 |
| `train_xgboost(X_train, y_train)` | Challenger 모델 학습 |
| `evaluate_model(model, X_test, y_test)` | R² / MAE / RMSE 계산 |
| `compare_models(models, X_test, y_test)` | 전체 모델 비교표 + CV |
| `plot_model_comparison(result_df)` | R²/MAE/RMSE 비교 막대 차트 |
| `plot_actual_vs_predicted(models, X_test, y_test)` | Actual vs Predicted 산점도 |
| `plot_shap_summary(model, X_test)` | SHAP Summary Plot |
| `plot_shap_group_contribution(model, X_test, features)` | **피처 그룹별 SHAP 기여도** 집계 및 시각화 |

**피처 선택 기준:** `FEATURE_INCLUDE_KEYWORDS = ["Mean", "Std", "Trend", "Variability", "Variety", "Abnormal", "Sum"]`
에 해당하는 컬럼 중 `LabTestCount`를 명시적으로 제외하여 143개 피처 구성.

---

## 실행 예시

```python
from src.feature_engineering import load_raw_tables, build_final_dataset, save_processed
from src.modeling import prepare_data, train_xgboost, plot_shap_summary, plot_shap_group_contribution

# 1. 데이터 로드 및 전처리
tables = load_raw_tables("data/raw")          # .txt 파일 로드
df     = build_final_dataset(tables)          # 372행 × 165열
save_processed(df)                            # data/processed/ 저장

# 2. 모델링
X_train, X_test, y_train, y_test, features = prepare_data(df)
xgb = train_xgboost(X_train, y_train)

# 3. SHAP 해석
plot_shap_summary(xgb, X_test,
                  save_path="outputs/figures/shap_summary.png")
plot_shap_group_contribution(xgb, X_test, features,
                             save_path="outputs/figures/shap_group_contribution.png")
```
