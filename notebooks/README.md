# Healthcare Data Analysis - 재입원 예측 분석

의료 데이터를 활용한 환자 재입원 예측 및 위험 요인 분석 포트폴리오

## 📁 프로젝트 구조

```
notebooks/
├── 01_data_overview.ipynb              # 데이터셋 소개 및 기본 구조
├── 02_feature_engineering.ipynb        # 피처 엔지니어링
├── 03_eda.ipynb                        # 탐색적 데이터 분석
├── 04_modeling.ipynb                   # 머신러닝 모델링
└── appendix_detailed_analysis.ipynb    # 상세 분석 (부록)
```

## 🎯 프로젝트 목표

- 환자의 재입원 가능성을 예측하는 머신러닝 모델 개발
- 재입원에 영향을 미치는 주요 위험 요인 파악
- 의료 자원 최적 배분을 위한 인사이트 도출

## 📊 데이터셋

### 1. PatientCore (환자 정보)
- 100명의 환자 기본 정보
- 인구통계학적 특성 (성별, 인종, 연령, 결혼상태 등)
- 사회경제적 지표 (빈곤율)

### 2. AdmissionsCore (입원 기록)
- 총 372건의 입원 기록
- 입원/퇴원 일시
- 환자당 평균 3.7회 입원

### 3. AdmissionsDiagnosesCore (진단 정보)
- 372건의 진단 기록
- ICD-10 진단 코드 및 설명
- 349개의 고유 진단

### 4. LabsCore (검사 결과)
- 111,483건의 검사 결과
- 35가지 검사 항목
- 혈액검사, 대사검사, 소변검사 등

## 🔬 분석 과정

### 1단계: 데이터 오버뷰 (`01_data_overview.ipynb`)
- 각 데이터셋의 구조와 특징 파악
- 데이터 품질 체크 (결측치, 중복, 이상치)
- 기본 통계량 확인

### 2단계: 피처 엔지니어링 (`02_feature_engineering.ipynb`)
- 데이터셋 병합
- 파생 변수 생성:
  - 재원 기간 (Length of Stay)
  - 연령, 연령대
  - 재입원 여부 (타겟 변수)
  - 30일 이내 재입원 여부
  - 검사 결과 피벗
- 결측치 처리

### 3단계: 탐색적 데이터 분석 (`03_eda.ipynb`)
- 재입원 패턴 분석
- 인구통계학적 특성과 재입원의 관계
- 입원 패턴 (계절성, 요일별)
- 주요 진단과 재입원의 상관관계
- 검사 결과와 재입원의 연관성

### 4단계: 예측 모델링 (`04_modeling.ipynb`)
- 여러 알고리즘 비교:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- 모델 평가 지표:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC
  - Confusion Matrix
- 피처 중요도 분석
- 하이퍼파라미터 튜닝

## 📈 주요 결과

### 재입원 통계
- 전체 재입원 비율: ~72%
- 30일 이내 재입원 비율: ~13%

### 주요 위험 요인
1. **입원 관련**
   - 재원 기간이 긴 환자
   - 과거 입원 횟수가 많은 환자
   
2. **인구통계학적**
   - 고령 환자 (특히 70세 이상)
   - 특정 사회경제적 배경

3. **진단**
   - 복합 질환 환자
   - 특정 ICD-10 카테고리

4. **검사 결과**
   - 주요 검사 수치 이상

### 모델 성능
- 최적 모델: Random Forest / Gradient Boosting
- Test AUC: 0.85~0.90 (예상)
- 재입원 위험 환자의 85% 이상 정확히 식별

## 💡 활용 방안

### 임상적 활용
1. **입원 시 위험도 평가**
   - 재입원 고위험 환자 조기 식별
   - 맞춤형 퇴원 계획 수립

2. **퇴원 후 관리**
   - 고위험 환자 집중 모니터링
   - 예방적 개입 프로그램 운영

3. **의료 자원 배분**
   - 효율적인 병상 관리
   - 간호 인력 최적 배치

### 비즈니스 가치
- 재입원율 감소 → 의료비 절감
- 환자 만족도 향상
- 병원 평가 지표 개선

## 🛠️ 기술 스택

- **Python 3.8+**
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **머신러닝**: scikit-learn
- **통계 분석**: scipy

## 📝 노트북 실행 방법

### 1. 순차 실행 (권장)
```bash
jupyter notebook 01_data_overview.ipynb
jupyter notebook 02_feature_engineering.ipynb
jupyter notebook 03_eda.ipynb
jupyter notebook 04_modeling.ipynb
```

### 2. 독립 실행
각 노트북은 독립적으로 실행 가능하도록 설계되었습니다.
(데이터 로드 코드가 각 노트북에 포함)

### 3. 부록
```bash
jupyter notebook appendix_detailed_analysis.ipynb
```
모든 변수에 대한 상세 분석이 필요할 때 참고

## 📚 참고 사항

### 데이터 출처
- GitHub Repository: [Basic_Health_Care](https://github.com/tashydean/Basic_Health_Care)
- 데이터는 GitHub에서 자동으로 로드됩니다

### 주의사항
- 이 데이터는 분석 및 학습 목적으로 사용됩니다
- 실제 환자 데이터가 아닌 시뮬레이션 데이터입니다
- 실무 적용 시 추가 검증이 필요합니다

## 🔄 개선 가능 영역

1. **데이터 확장**
   - 더 많은 환자 샘플
   - 추가 변수 (약물, 시술 등)
   - 장기 추적 데이터

2. **모델 고도화**
   - 딥러닝 모델 시도
   - 시계열 모델 적용
   - 앙상블 기법 활용

3. **설명 가능성**
   - SHAP 값 분석
   - LIME 적용
   - 인과관계 분석

## 👤 작성자

의료 데이터 분석 포트폴리오
- GitHub: [tashydean](https://github.com/tashydean)

## 📄 라이선스

이 프로젝트는 학습 및 포트폴리오 목적으로 작성되었습니다.

---

**Last Updated**: 2026-01-30
