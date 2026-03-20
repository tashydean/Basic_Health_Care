# outputs/

분석 실행 결과물 보관 폴더. 노트북을 순서대로 실행하면 자동 생성됩니다.

```
outputs/
├── processed_healthcare_data.csv     # 전처리 완료 통합 데이터 (372행 × 165열)
└── figures/
    ├── eda_demographics.png          # 성별·연령대 분포
    ├── eda_race_marital.png          # 인종·결혼 상태 분포
    ├── eda_los_dist.png              # 입원 기간 분포
    ├── eda_yearly_admission.png      # 연도별 입원 추이
    ├── eda_los_by_group.png          # 성별·연령대별 입원 기간
    ├── eda_top_labs.png              # 검사 항목 Top 10
    ├── eda_labcount_vs_los.png       # 검사 횟수 vs LOS 산점도
    ├── eda_top_diagnoses.png         # 진단명 Top 10
    ├── eda_pca_tsne.png              # PCA Scree + t-SNE
    ├── eda_kmeans.png                # K-Means Elbow/Silhouette
    ├── eda_cluster_viz.png           # PCA/t-SNE 군집 시각화
    ├── model_comparison.png          # 3모델 R²/MAE/RMSE 비교
    ├── actual_vs_predicted.png       # Actual vs Predicted (3모델)
    ├── shap_summary.png              # SHAP Summary Plot (XGBoost)
    ├── shap_group_contribution.png   # 피처 그룹별 기여도
    ├── shap_dependence.png           # SHAP Dependence Plot (상위 4개)
    └── residual_analysis.png         # 잔차 분포 및 패턴
```

## 파일 생성 방법

```bash
# 전처리 데이터셋 생성
jupyter notebook notebooks/02_feature_engineering.ipynb

# EDA figures 생성
jupyter notebook notebooks/03_exploratory_data_analysis.ipynb

# Modeling figures 생성
jupyter notebook notebooks/04_modeling.ipynb
```

또는 `src/` 모듈로 직접 실행:

```bash
python src/feature_engineering.py
python src/modeling.py
```
