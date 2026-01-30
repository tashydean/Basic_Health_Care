# EMR-based Length of Stay Prediction & Phenotype Exploration

## 1. Problem Statement
본 분석의 목적은 EMR(Electronic Medical Record) 정형 데이터를 활용하여  
환자의 검사 수치 패턴이 입원 기간(Length of Stay, LOS)에 어떤 구조적 신호를 제공하는지를 탐색하는 것이다.

특히 단일 검사 수치의 평균값이 환자 상태를 충분히 설명하지 못한다는 가설 하에,  
검사의 시간적 변화(Trend), 변동성(Variability), 이상 소견의 누적과 같은  
종합적 지표가 임상적 중증도 및 회복 지연과 어떤 관계를 가지는지를 검증하고자 한다.

---

## 2. Dataset Overview
본 프로젝트는 csbond007의  
**Basic HealthCare Data Analysis using Electronic Medical Records (EMR)** 데이터셋을 사용하였다.

### 데이터 구성
- 환자 정보 테이블
- 입원(Admission) 테이블
- 진단(Diagnosis) 테이블
- 검사(Lab) 테이블  
→ 환자 × 입원 ID를 기준으로 병합하여 분석 단위를 구성

### 한계 및 분석 방향 전환
- 진단 설명 텍스트는 진단 코드와 1:1 대응되지 않아 지도학습 기반 활용에 제약
- 진단 코드 분포가 불균형하여 특정 진단 패턴 도출 및 예측이 어려움

→ 이에 따라 본 분석은 **진단 예측이 아닌**,  
   검사 데이터 기반으로 환자의 **전반적 생리적 불안정성과 입원 기간(LOS)** 간의 관계 분석으로 초점을 전환하였다.

---

## 3. Feature Engineering
시계열 검사 로우 데이터로부터 환자 상태를 요약하는 구조적 변수를 설계하였다.

### 기본 변수
- 범주형: AdmissionID, PatientGender, PatientRace, PatientMaritalStatus, PatientLanguage, PrimaryDiagnosisCode, LabName
- 수치형: PatientPopulationPercentageBelowPoverty

### 파생 변수
- `HoursAfterAdmission`: 입원 후 검사까지 경과 시간
- 검사 요약 통계  
  - `Lab_Mean`, `Lab_Std`
- 구조적 지표  
  - `Lab_Trend`: 검사 수치의 시간적 변화 방향
  - `Abnormal_Sum`: 검사 이상 소견 누적 횟수
- 환자 단위 종합 지표  
  - `Total_Abnormal_Count`: 전체 검사에서의 이상 소견 누적
  - `Total_Lab_Variability`: 환자 단위 평균 변동성
- 타겟 변수  
  - `LOS_Days`: 입원 기간

> 단일 시점 수치가 아닌, 환자의 **전반적 불안정성**을 정량화하는 것을 목표로 변수 설계를 수행하였다.

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Dimension Reduction (PCA & t-SNE)
검사 평균값 기반 표현의 분산 구조 및 환자 간 응집성 여부를 확인하기 위해  
PCA(선형 차원 축소)와 t-SNE(비선형 임베딩)를 적용하였다.

- **PCA**
  - 개별 주성분 및 상위 주성분의 설명력이 전반적으로 낮음
  - 특정 검사가 환자 특성을 대표하는 주된 분산 축으로 작용하지 않음
- **t-SNE**
  - 환자 샘플들이 특정 군집 없이 고르게 분포
  - 국소적 응집 패턴 관찰되지 않음

### 4.2 주성분–임상 변수 정렬 분석
차원 축소 결과가 실제 임상 변수와 어떤 관계를 가지는지 확인하기 위해  
상위 주성분과 연령, 진단군, 재입원 여부 등 임상 변수 간 산점도를 추가적으로 평가하였다.

- 다수의 임상 변수에 대해 상위 주성분 값이 거의 일정한 분포를 보임
- 이는 주성분 축이 임상 변수들과 유의미한 선형 상관관계를 갖지 않음을 시사

### 요약
평균 검진 수치를 기반으로 한 차원 축소 분석에서는  
환자 임상 특성을 설명할 수 있는 유의미한 분산 구조가 관찰되지 않았으며,  
이는 단순 요약 통계만으로는 환자 상태의 이질성을 충분히 포착하기 어렵다는 한계를 보여준다.

---

## 5. Unsupervised Learning: Patient Clustering

PCA 및 t-SNE 분석에서 연속적인 분산 구조는 관찰되지 않았으나,  
평균 검사 수치 패턴에 기반한 **이산적 환자 하위군(phenotype)** 존재 가능성을 검토하기 위해  
K-Means 군집화를 수행하였다.

- Elbow method 및 silhouette score를 통해 군집 수 평가
- Silhouette score는 K=2에서만 상대적으로 높았으며,
  그 외 K에서는 명확한 군집 분리가 관찰되지 않음

→ 평균 검사 수치의 선형 결합만으로는  
   임상적으로 해석 가능한 자연 군집이 형성되지 않음을 시사

---

## 6. Modeling: Length of Stay Prediction & XAI

### 6.1 모델링 목적
차원 축소 및 군집화에서는 구조가 명확히 드러나지 않았으나,  
**특정 결과 변수(LOS)에 영향을 주는 요인은 존재하는지**를 확인하고자 지도학습을 수행하였다.

### 6.2 모델 구성
- 입력 변수
  - 검사 평균값
  - Trend, Variability, Abnormal Count
  - (참고용) Cluster Label
- 모델
  - Random Forest Regressor
- 평가 지표
  - R², Mean Absolute Error (MAE)

### 결과
- **R² Score: 0.9123**
- **MAE: 1.26 days**

> 검사 데이터 기반 변수만으로 입원 기간 변동의 약 91%를 설명하였으며,  
> 평균 예측 오차는 약 1.26일 수준으로 나타났다.

※ 주의사항  
본 모델은 입원 기간 중 관측된 검사 데이터를 활용한 **사후적 분석(post-hoc analysis)**으로,  
실제 입원 초기 예측 모델과는 목적이 다르며 외부 검증이 필요하다.

---

## 7. Model Interpretation: SHAP Analysis
SHAP(SHapley Additive exPlanations)을 활용하여  
모델이 LOS를 예측하는 데 기여한 주요 변수를 해석하였다.

- 단일 검사 수치의 평균은 상대적으로 기여도가 낮음
- `Total_Abnormal_Count`, `Total_Lab_Variability`와 같은
  종합적·구조적 지표가 예측에 핵심적인 역할 수행

이는 탐색적 데이터 분석 결과와 일관되며,  
단일 지표보다 **전반적 불안정성 지표**가 입원 기간과 더 밀접하게 연관됨을 시사한다.

---

## 8. Clinical Interpretation
환자 상태를 평가함에 있어  
개별 검사 수치보다 여러 검사에서 반복적으로 나타나는 이상 소견과  
회복 과정에서의 변동성이 임상적으로 더 중요한 신호로 작용할 수 있음을 보여준다.

---

## 9. Key Takeaways
- 평균 검사 수치 기반 표현만으로는 환자 임상 특성 분리가 어려움
- 시간적 변화, 변동성, 이상 누적과 같은 구조적 지표는 LOS 예측에 유효
- 고차원·노이즈가 많은 EMR 데이터에서도
  적절한 feature engineering을 통해 설명력 있는 모델 구축 가능성 확인
- 단일 데이터셋 기반 분석으로, 향후 외부 검증 및 시점 제한 모델링이 필요
