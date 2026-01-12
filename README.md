# 🏥 EMR-driven Clinical Phenotyping via Unsupervised Clustering

## Abstract
This project aims to identify clinically meaningful patient subgroups by applying unsupervised clustering techniques to structured Electronic Medical Record (EMR) data.  
Due to the high dimensionality of laboratory measurements and the extreme sparsity and imbalance of diagnosis labels, the project pivots from supervised diagnosis classification to data-driven clinical phenotyping.  
Through systematic exploratory data analysis (EDA), patient-level data integration, feature normalization, dimensionality reduction, and clustering, the study seeks to uncover latent physiological patterns that may correspond to different clinical states or disease severity levels.

## 국문 요약
본 프로젝트는 EMR(전자의무기록) 데이터에 대해 **비지도 학습 기반 군집화 분석을 수행하여 임상적으로 해석 가능한 환자 하위 집단(phenotype)을 도출하는 것**을 목표로 합니다.  
진단 코드가 349개로 매우 다양하고 표본 수 대비 클래스 불균형이 심각한 데이터 특성을 고려할 때, 지도학습 기반 진단 분류보다는 **검사 수치 패턴 자체에 기반한 군집 분석이 통계적·의학적으로 더 합리적인 접근**이라고 판단하였습니다.

---

## 🎯 프로젝트 목표 및 현재 진행 상황 (Milestones & Status)

이 프로젝트의 핵심 목표는  
**환자별 검사 수치 프로파일을 기반으로 잠재적인 임상 패턴을 식별하고, 각 군집의 특징을 해석 가능한 형태로 정리하는 것**입니다.

**현재 진행률: 🟢 65% (군집화 및 해석 단계)**

| 단계 | 목표 작업 (Task) | 상태 (Status) |
| :--- | :--- | :--- |
| **1단계** | Admissions, Diagnoses, Labs, Patients 데이터 로드 | ✅ 완료 |
| **2단계** | 데이터셋별 EDA 및 데이터 품질 검증 | ✅ 완료 |
| **3단계** | Patient–Admission 단위 통합 데이터셋 구축 | ✅ 완료 |
| **4단계** | 검사 수치 정규화 및 차원 축소 (PCA) | ✅ 완료 |
| **5단계** | 비지도 학습 기반 군집화 (K-means) | ✅ 완료 |
| **6단계** | 군집별 검사 수치 중앙값 기반 프로파일링 | 🏃 진행 중 |
| **7단계** | 임상적 의미 해석 및 확장 가능성 정리 | 📅 예정 |

---

## 🔍 분석 결론 요약 (Clustering Insights)

- PCA 결과, 설명 분산이 소수 차원에 집중되지 않아 **단일 지표 또는 단순 규칙 기반 분류가 어려운 데이터 구조**임을 확인
- 군집별 검사 수치 중앙값 프로파일을 통해 다음과 같은 상이한 패턴이 관찰됨
  - 전반적으로 정상 범위에 가까운 검사 수치를 보이는 군집
  - 다수 검사 항목에서 동시에 편차가 나타나는 군집
  - 특정 계열(예: 대사, 염증 관련 지표)에 국한된 이상 패턴을 보이는 군집
- 이는 진단 코드 예측보다는 **환자 상태 요약, 질병 중증도 구분, 위험군 탐색**과 같은 임상 보조 분석에 더 적합한 활용 가능성을 시사함

---

## 🚀 주요 특징 (Key Features)

- **문제 재정의 중심 접근**  
  데이터 제약을 고려하여 지도학습 기반 진단 분류에서 비지도 학습 기반 군집화로 분석 목표 전환
- **환자 중심 데이터 통합**  
  Patient–Admission 단위로 검사, 인구통계, 진단 정보 결합
- **해석 가능성 중시 설계**  
  군집별 중앙값 기반 프로파일링을 통해 의료 도메인 해석 용이
- **확장 가능성 확보**  
  대규모 코호트 데이터 적용 시 임상 phenotyping 연구로 확장 가능
