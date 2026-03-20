# notebooks/

분석 단계별 Jupyter Notebook. **번호 순서대로 실행하세요.**

| 파일 | 내용 | 주요 출력 |
|------|------|-----------|
| `01_data_overview.ipynb` | 데이터 탐색 — 결측치·분포·테이블 관계 확인 | — |
| `02_feature_engineering.ipynb` | 전처리 + 검사명별 Wide Format 파생변수 생성 | `outputs/processed_healthcare_data.csv` |
| `03_exploratory_data_analysis.ipynb` | EDA — 기술통계, PCA, t-SNE, K-Means 군집화 | `outputs/figures/eda_*.png` (11개) |
| `04_modeling.ipynb` | 예측 모델링(LR/RF/XGBoost) + SHAP 해석 | `outputs/figures/` 아래 6개 |
| `appendix_detailed_analysis.ipynb` | 변수별 상세 탐색 (참고용) | — |

## 실행 환경

```bash
pip install -r ../requirements.txt
```

## 재사용 가능한 함수

노트북의 핵심 로직은 `src/` 모듈로 분리되어 있습니다.

```python
from src.feature_engineering import load_raw_tables, build_final_dataset, save_processed
from src.modeling import prepare_data, train_xgboost, plot_shap_summary
```

## Figure 저장 목록

각 노트북 실행 시 `outputs/figures/` 에 자동 저장됩니다.

| 노트북 | 저장 파일 | 내용 |
|--------|-----------|------|
| `03_exploratory_data_analysis` | `eda_demographics.png` | 성별·연령대 분포 |
| | `eda_race_marital.png` | 인종·결혼 상태 분포 |
| | `eda_los_dist.png` | 입원 기간(LOS) 분포 |
| | `eda_yearly_admission.png` | 연도별 입원 추이 |
| | `eda_los_by_group.png` | 성별·연령대별 입원 기간 비교 |
| | `eda_top_labs.png` | 검사 항목 Top 10 빈도 |
| | `eda_labcount_vs_los.png` | 검사 횟수 vs LOS 산점도 |
| | `eda_top_diagnoses.png` | 진단명 Top 10 빈도 |
| | `eda_pca_tsne.png` | PCA Scree Plot + t-SNE 산점도 |
| | `eda_kmeans.png` | K-Means Elbow / Silhouette |
| | `eda_cluster_viz.png` | PCA / t-SNE 공간 군집 시각화 |
| `04_modeling` | `model_comparison.png` | 3모델 R²/MAE/RMSE 비교 막대 |
| | `actual_vs_predicted.png` | Actual vs Predicted (3모델) |
| | `shap_summary.png` | SHAP Summary Plot (XGBoost) |
| | `shap_group_contribution.png` | 피처 그룹별 SHAP 기여도 |
| | `shap_dependence.png` | SHAP Dependence Plot (상위 4개 변수) |
| | `residual_analysis.png` | 잔차 분포 및 패턴 |
