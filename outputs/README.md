# outputs/

노트북 실행 시 자동 생성되는 결과물 보관 폴더.

```
outputs/
├── processed_healthcare_data.csv     # 전처리 완료 통합 데이터
└── figures/
    ├── eda_demographics.png          # 성별·연령대 분포
    ├── eda_pca_tsne.png              # PCA Scree + t-SNE
    ├── eda_kmeans.png                # K-Means Elbow/Silhouette
    ├── eda_cluster_viz.png           # PCA/t-SNE 군집 시각화
    ├── model_comparison.png          # 3모델 R²/MAE/RMSE 비교
    ├── actual_vs_predicted.png       # Actual vs Predicted
    ├── shap_summary.png              # SHAP Summary Plot
    ├── shap_group_contribution.png   # 피처 그룹별 기여도
    ├── shap_dependence.png           # SHAP Dependence Plot
    └── residual_analysis.png         # 잔차 분포
```
