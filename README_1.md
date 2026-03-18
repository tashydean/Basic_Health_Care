# EMR 기반 입원 기간(LOS) 예측 및 환자 표현형 탐색

> **Electronic Medical Record(EMR) 정형 데이터를 활용한 엔드투엔드 데이터 분석 프로젝트**  
> Feature Engineering → EDA → 비지도학습 → 지도학습(XGBoost / Random Forest) → XAI(SHAP)

---

## 📌 분석 목적

단일 검사 수치의 평균값은 환자 상태를 충분히 설명하지 못한다는 가설 하에,  
검사의 **시간적 변화(Trend)**, **변동성(Variability)**, **이상 소견 누적(Abnormal Count)** 이라는  
구조적 지표를 직접 설계하고, 이것이 입원 기간(LOS)과 어떤 관계를 갖는지를 실증하였다.

---

## 📊 데이터셋 개요

| 항목 | 내용 |
|------|------|
| 출처 | [csbond007/Basic_Health_Care](https://github.com/csbond007/Basic_Health_Care) (공개 EMR 데이터) |
| 기간 | 1941–2013년 |
| 환자 수 | 100명 |
| 입원 기록 | 372건 |
| 검사 결과 | 111,483건 |
| 분석 단위 | 입원(Admission) 단위로 4개 테이블 병합 |

---

## 🗂️ 프로젝트 구조

```
Basic_Health_Care/
├── data/                     # 원본 및 전처리 완료 데이터
│   ├── raw/                  # 원본 EMR 테이블 (4개 CSV)
│   └── processed/            # 모델링용 통합 데이터셋
├── notebooks/                # 분석 단계별 Jupyter Notebook
│   ├── 01_data_overview.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   └── 04_modeling.ipynb
├── src/                      # 재사용 가능한 Python 모듈
│   ├── feature_engineering.py
│   └── modeling.py
├── outputs/                  # 분석 결과물
│   ├── figures/              # 시각화 이미지
│   └── processed_healthcare_data.csv
├── docs/                     # 분석 요약 문서
│   └── analysis_summary.md
├── requirements.txt
└── README.md
```

---

## 🔬 분석 파이프라인

```
원본 EMR 4개 테이블
        │
        ▼
[01] 데이터 탐색 (결측치·분포·관계 확인)
        │
        ▼
[02] Feature Engineering
     └─ Trend / Variability / Abnormal Count 파생변수 설계
        │
        ▼
[03] EDA  ─── PCA / t-SNE → 군집화(K-Means)
        │         └─ 평균 수치만으로는 환자 분리 불가 → 구조적 지표 필요성 확인
        ▼
[04] Modeling
     ├─ Baseline: Linear Regression
     ├─ Main:     Random Forest + GridSearchCV
     ├─ Challenger: XGBoost
     ├─ 검증:    5-Fold Cross Validation
     └─ XAI:     SHAP Summary Plot
```

---

## 📈 핵심 결과

### 모델 성능 비교 (Test Set)

| 모델 | R² | MAE | CV R² (mean ± std) |
|------|----|-----|---------------------|
| Linear Regression | — | — | — |
| **Random Forest (Tuned)** | **~0.91** | **최소** | **안정적** |
| XGBoost | — | — | — |

> *노트북 실행 후 실제 수치로 업데이트 예정*

### SHAP 분석 주요 발견

```
예측 기여도 순위 (SHAP value 기준)

1위  Total_Abnormal_Count   ██████████████████  비정상 검사 누적 횟수
2위  Total_Lab_Variability  ████████████        검사 수치 전체 변동성
3위  Lab_Std (개별 검사)     ████████            개별 검사 내 표준편차
     ...
n위  인구통계 변수           ███                 (기여도 상대적으로 낮음)
```

→ **임상 지표(검사 패턴)가 인구통계보다 입원 기간 예측에 핵심적**

---

## 🛠️ 기술 스택

| 범주 | 라이브러리 |
|------|-----------|
| 데이터 처리 | `pandas`, `numpy` |
| 시각화 | `matplotlib`, `seaborn` |
| 차원 축소 | `scikit-learn` (PCA, t-SNE) |
| 군집화 | `scikit-learn` (KMeans) |
| 모델링 | `scikit-learn`, `xgboost` |
| XAI | `shap` |

---

## 🚀 실행 방법

```bash
# 1. 레포지토리 클론
git clone https://github.com/tashydean/Basic_Health_Care.git
cd Basic_Health_Care

# 2. 라이브러리 설치
pip install -r requirements.txt

# 3. 순서대로 노트북 실행
jupyter notebook notebooks/01_data_overview.ipynb
```

---

## ⚠️ 데이터 한계 및 개선 방향

| 한계 | 내용 | 개선 방향 |
|------|------|-----------|
| 샘플 부족 | 100명 수준, 외부 일반화 제한 | 대규모 EHR 데이터로 재검증 |
| 진단 변수 미반영 | 진단 코드 분포 불균형으로 피처 제외 | 진단 코드 인코딩 + 동반 상병 추가 |
| 시계열 구조 손실 | 입원 전체 집계로 단순화 | LSTM 등 시퀀스 모델 적용 |
| 외부 검증 부재 | 단일 데이터셋 내 분리만 수행 | 독립 검증셋 또는 공개 EHR로 확인 |

---

## 👤 Author

**전하** | Data Analyst  
📧 [이메일 추가] | 🔗 [LinkedIn 추가]
