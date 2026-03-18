# notebooks/

분석 단계별 Jupyter Notebook을 보관합니다.  
**반드시 번호 순서대로 실행하세요.** 각 노트북의 출력물이 다음 단계의 입력이 됩니다.

---

## 실행 순서 및 내용

| 파일 | 단계 | 주요 내용 | 출력 |
|------|------|-----------|------|
| `01_data_overview.ipynb` | 탐색 | 결측치, 분포, 테이블 관계 확인 | — |
| `02_feature_engineering.ipynb` | 전처리 | 파생변수 설계, 4개 테이블 병합 | `outputs/processed_healthcare_data.csv` |
| `03_eda.ipynb` | EDA | PCA, t-SNE, K-Means 군집화 | `outputs/figures/eda_*.png` |
| `04_modeling.ipynb` | 모델링 | LR / RF / XGBoost, CV, SHAP | `outputs/figures/model_*.png` |

---

## 노트북별 상세 설명

### 01_data_overview
- 4개 원본 테이블 로드 및 기본 통계 확인
- 결측치 분포 히트맵
- PatientID ↔ AdmissionID ↔ LabID 관계 확인
- 주요 수치형 변수 분포 시각화

### 02_feature_engineering
- 날짜 파싱 및 `LengthOfStay` 계산
- `HoursAfterAdmission`: 입원 후 검사 경과 시간
- `Lab_Trend`: 선형회귀 기울기로 시간적 변화 방향 포착
- `Total_Abnormal_Count`, `Total_Lab_Variability`: 환자 단위 종합 지표
- 최종 통합 데이터셋 저장

### 03_eda
- **PCA**: 주성분 설명력 분석 → 평균 수치 기반 표현의 한계 확인
- **t-SNE**: 비선형 임베딩으로 환자 분포 탐색
- **K-Means**: Elbow / Silhouette로 최적 군집 수 결정 → 자연 군집 부재 확인
- → 구조적 파생 변수의 필요성을 EDA로 실증

### 04_modeling
- **Baseline**: Linear Regression
- **Main**: Random Forest + GridSearchCV (하이퍼파라미터 최적화)
- **Challenger**: XGBoost
- **검증**: 5-Fold Cross Validation
- **XAI**: SHAP Summary Plot으로 변수 기여도 해석
- **결론**: 데이터 한계 및 개선 방향 제시

---

## 재사용 가능한 함수

노트북에서 반복 사용된 핵심 로직은 `src/` 모듈로 분리되어 있습니다.

```python
from src.feature_engineering import build_final_dataset
from src.modeling import prepare_data, train_random_forest, plot_shap_summary
```
