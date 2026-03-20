# EMR 기반 탐색적 데이터 분석 

**EMR(Electronic Medical Record) 정형 데이터 EDA — 검사 수치 패턴과 입원 기간(LOS)의 임상 신호 탐색**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

본 프로젝트는 EMR 정형 데이터에 대한 **탐색적 데이터 분석(EDA)** 을 주목적으로 한다.

단순 기술통계(평균값) 기반 표현만으로는 환자 상태의 이질성을 충분히 파악하기 어렵다는 점에 주목하여,
보다 구조적인 탐색을 위해 **입원 기간(LOS)과 검사 패턴의 관계**를 집중적으로 분석하였다.
모델링은 탐색의 도구로 활용하였으며, EDA 단계에서 세운 가설을 SHAP으로 실증하는 것이 핵심이다.

**분석 흐름:**
EDA에서 단순 평균 표현의 한계를 관찰 → 구조적 파생변수 설계 → 모델을 해석 도구로 활용 → SHAP으로 임상 신호 실증

**핵심 결과:**
SHAP 분석 결과, `Total_Abnormal_Count`(전체 이상 소견 누적) 단일 변수가
143개 피처 전체 SHAP 기여의 **70.1%** 를 차지하였으며,
검사 평균값 그룹(Lab_Mean 계열)의 기여도는 **10.1%** 로 하위였다.
→ 단순 평균은 임상 신호를 충분히 포착하지 못한다는 가설 실증

---

## Dataset

| 항목 | 내용 |
|------|------|
| 출처 | [csbond007/Basic_Health_Care](https://github.com/csbond007/Basic_Health_Care) (Synthea 기반 합성 EMR) |
| 기간 | 1941–2013년 |
| 환자 수 | 100명 |
| 입원 기록 | 372건 |
| 검사 결과 | 111,483건 |
| 분석 단위 | 입원(Admission) 단위 — 4개 테이블 병합 |

### 분석 방향 전환 배경

- 진단 코드(PrimaryDiagnosisCode)는 Long-tail 분포로 분류 모델 구축 불가
- 재입원 기록이 임상 맥락 없이 독립 이벤트로 처리되어 인과 추적 불가
- → **검사 데이터 기반 생리적 불안정성 ↔ LOS 관계 분석**으로 초점 전환

---

## Repository Structure

```
Basic_Health_Care/
├── data/
│   ├── raw/                                      # 원본 EMR 탭 구분 텍스트 파일
│   │   ├── PatientCorePopulatedTable.txt
│   │   ├── AdmissionsCorePopulatedTable.txt
│   │   ├── AdmissionsDiagnosesCorePopulatedTable.txt
│   │   └── LabsCorePopulatedTable.txt
│   ├── processed/
│   │   └── processed_healthcare_data.csv         # 전처리 완료 통합 데이터 (372행 × 165열)
│   └── README.md
├── notebooks/                                    # 분석 단계별 Jupyter Notebook
│   ├── 01_data_overview.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   ├── 04_modeling.ipynb
│   ├── appendix_detailed_analysis.ipynb          # 변수별 상세 탐색 (참고용)
│   └── README.md
├── outputs/
│   ├── processed_healthcare_data.csv
│   └── figures/                                  # 시각화 결과물
│       ├── eda_demographics.png
│       ├── eda_race_marital.png
│       ├── eda_los_dist.png
│       ├── eda_yearly_admission.png
│       ├── eda_los_by_group.png
│       ├── eda_top_labs.png
│       ├── eda_labcount_vs_los.png
│       ├── eda_top_diagnoses.png
│       ├── eda_pca_tsne.png
│       ├── eda_kmeans.png
│       ├── eda_cluster_viz.png
│       ├── model_comparison.png
│       ├── actual_vs_predicted.png
│       ├── shap_summary.png
│       ├── shap_group_contribution.png
│       ├── shap_dependence.png
│       └── residual_analysis.png
├── src/                                          # 재사용 가능한 Python 모듈
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── README.md
├── docs/
│   └── analysis_summary.md                       # 분석 전체 흐름 요약 문서
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Analysis Pipeline

```
4개 원본 EMR 테이블 (data/raw/*.txt)
        │
        ▼
[01] 데이터 탐색 (01_data_overview.ipynb)
     결측치 · 분포 · 테이블 관계 확인
        │
        ▼
[02] Feature Engineering (02_feature_engineering.ipynb)
     검사명별 Wide Format 피처 생성
     Lab_Std / Lab_Trend / Abnormal_Sum (검사명별 35종)
     Total_Abnormal_Count / Total_Lab_Variability (환자 단위)
     → data/processed/processed_healthcare_data.csv (372행 × 165열)
        │
        ▼
[03] EDA (03_exploratory_data_analysis.ipynb)
     ① 인구통계 · 입원 패턴 · 검사 빈도 기술통계
     ② PCA → PC1 설명력 7.0% → 지배적 구조 없음
     ③ t-SNE → 자연 군집 없음
     ④ K-Means → K=2에서만 미약한 분리
     → "단순 평균 기반 표현의 한계" 실증
        │
        ▼
[04] Modeling + XAI (04_modeling.ipynb)
     ┌─ Part 1: 예측 (구조적 지표만으로 LOS 예측 가능한가?)
     │   Baseline : Linear Regression   (R²=0.825, CV R²=0.712)
     │   Main     : Random Forest Tuned (R²=0.916, CV R²=0.890)
     │   Best     : XGBoost             (R²=0.918, MAE=1.12일)
     │
     └─ Part 2: 해석 — 모델을 임상 신호 발굴 도구로 전환
         SHAP Summary      → Total_Abnormal_Count 압도적 1위
         SHAP 그룹 기여도  → 이상 누적(70.1%) vs 검사 평균(10.1%)
         SHAP Dependence   → 효과 형태(선형/임계점) 탐색
         잔차 분석         → 극단 LOS 케이스 오차 집중 확인
```

---

## Feature Engineering

실제 데이터에서는 검사명(LabName) 35종별로 피처가 생성되어 총 **143개 모델 피처**가 구성된다.

| 피처 그룹 | 예시 컬럼 | 설계 의도 | 피처 수 |
|-----------|-----------|-----------|---------|
| `Lab_Mean_{검사명}` | `Lab_Mean_CBC: HEMOGLOBIN` | 기준 비교용 (단순 평균) | 35 |
| `Lab_Std_{검사명}` | `Lab_Std_CBC: HEMOGLOBIN` | 개별 검사 변동성 | 35 |
| `Lab_Trend_{검사명}` | `Lab_Trend_CBC: HEMOGLOBIN` | 수치 악화/호전 방향 (선형회귀 기울기) | 35 |
| `Abnormal_Sum_{검사명}` | `Abnormal_Sum_CBC: HEMOGLOBIN` | 검사 단위 이상 소견 누적 | 35 |
| `Total_Abnormal_Count` | — | **환자 전반적 중증도 → SHAP 1위 (70.1%)** | 1 |
| `Total_Lab_Variability` | — | 환자 단위 전반 변동성 | 1 |
| `LabTestVariety` | — | 검사 종류 다양성 | 1 |

> **`LabTestCount` 제외 이유:** LOS와 상관계수 r=0.993으로 역인과 관계 강하게 의심.
> 제거 후에도 XGBoost R²=0.918 달성 → 순수 임상 신호만으로 예측 가능 확인.

---

## Key Results

### Model Performance

| 모델 | R² (Test) | MAE | RMSE | CV R² (5-Fold) |
|------|-----------|-----|------|----------------|
| Linear Regression | 0.825 | 1.697일 | 2.172 | 0.712 ± 0.068 |
| Random Forest (Tuned) | 0.916 | 1.157일 | 1.502 | **0.890 ± 0.035** |
| **XGBoost** | **0.918** | **1.123일** | **1.491** | 0.872 ± 0.030 |

- XGBoost가 Test Set 기준 최우수, CV R²는 RF가 근소하게 높음 → 소규모 데이터 특성상 **두 모델 실질적으로 동등**
- LR 대비 트리 계열 R² 격차 +0.09 → **비선형 구조 존재 확인**

### SHAP — Clinical Signal Contribution

| 피처 그룹 | 기여도 | 해석 |
|-----------|--------|------|
| **이상 소견 누적 (Abnormal)** | **70.1%** | `Total_Abnormal_Count` 단독 압도 (2위의 31배) |
| 변동성 (Std / Total_Lab_Variability) | 중간 | 검사 수치 불안정성 반영 |
| **검사 평균 (Lab_Mean)** | **10.1%** | **하위 — 단순 평균의 한계 실증** |
| 시간 추이 (Trend) | 하위 | 수치 악화/호전 방향 |

![SHAP Group Contribution](outputs/figures/shap_group_contribution.png)

![SHAP Summary](outputs/figures/shap_summary.png)

### Model Comparison

![Model Comparison](outputs/figures/model_comparison.png)

![Actual vs Predicted](outputs/figures/actual_vs_predicted.png)

### Prediction Error Analysis

- 잔차 평균: **0.077일** (무편향)
- 잔차 std: **1.500일**
- 대오차 케이스: **19건** (Test 75건의 25%) — 극단 LOS에서 집중

![Residual Analysis](outputs/figures/residual_analysis.png)

---

## Limitations & Future Work

| 한계 | 설명 | 개선 방향 |
|------|------|-----------|
| Synthea 합성 데이터 | 실제 임상 복잡성 미반영, LabTestCount 선형성 과장 가능성 | MIMIC-III 등 실제 EHR 검증 |
| 소규모 (100명 / 372건) | 외부 일반화 제한, CV std 높음 | 대규모 데이터 재검증 |
| 사후 분석 | 입원 완료 데이터 기반 — 실시간 예측 불가 | 초기 N일 기반 조기 예측 모델 전환 |
| 진단 정보 미반영 | Long-tail 분포로 피처 제외 | 진단 코드 임베딩 + 동반 상병 추가 |

---

## Quick Start

```bash
git clone https://github.com/tashydean/Basic_Health_Care.git
cd Basic_Health_Care
pip install -r requirements.txt

# 노트북 순서대로 실행
jupyter notebook notebooks/01_data_overview.ipynb
```

또는 `src/` 모듈로 직접 실행:

```python
from src.feature_engineering import load_raw_tables, build_final_dataset, save_processed
from src.modeling import prepare_data, train_xgboost, plot_shap_summary

tables = load_raw_tables("data/raw")
df     = build_final_dataset(tables)
save_processed(df)

X_train, X_test, y_train, y_test, _ = prepare_data(df)
xgb = train_xgboost(X_train, y_train)
plot_shap_summary(xgb, X_test, save_path="outputs/figures/shap_summary.png")
```

---

## Tech Stack

`Python 3.8+` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `shap` · `matplotlib` · `seaborn` · `scipy`

---

## Author

**[@tashydean](https://github.com/tashydean)**
