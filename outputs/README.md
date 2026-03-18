# outputs/

분석 실행 결과물을 보관합니다.  
노트북 또는 `src/` 스크립트를 실행하면 자동으로 생성됩니다.

```
outputs/
├── processed_healthcare_data.csv     # 모델링용 최종 통합 데이터셋
└── figures/
    ├── eda_pca_explained_variance.png    # PCA 설명 분산 그래프
    ├── eda_tsne_scatter.png              # t-SNE 2D 산점도
    ├── eda_kmeans_elbow.png              # K-Means Elbow Curve
    ├── eda_kmeans_silhouette.png         # Silhouette Score
    ├── model_comparison.png              # 모델별 R² / MAE / RMSE 비교
    ├── actual_vs_predicted.png           # Actual vs Predicted (3모델)
    └── shap_summary.png                  # SHAP Summary Plot
```

## 파일 생성 방법

```bash
# 전처리 완료 데이터셋 생성
python src/feature_engineering.py

# 모델 학습 및 시각화 생성
python src/modeling.py
```

또는 `notebooks/` 순서대로 실행.

## 주의

- `processed_healthcare_data.csv`는 `.gitignore`에 추가하거나,  
  파일 크기에 따라 Git LFS 사용을 권장합니다.
- `figures/`의 이미지 파일은 `README.md` 및 `docs/analysis_summary.md`에서 참조됩니다.
