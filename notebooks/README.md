# notebooks/

분석 단계별 Jupyter Notebook. **번호 순서대로 실행하세요.**

| 파일 | 내용 | 주요 출력 |
|------|------|-----------|
| `01_data_overview.ipynb` | 데이터 탐색 — 결측치·분포·테이블 관계 확인 | — |
| `02_feature_engineering.ipynb` | 전처리 + 파생변수 생성 | `outputs/processed_healthcare_data.csv` |
| `03_exploratory_data_analysis.ipynb` | EDA — PCA, t-SNE, K-Means 군집화 | `outputs/figures/eda_*.png` |
| `04_modeling.ipynb` | 모델링(LR/RF/XGBoost) + SHAP 해석 | `outputs/figures/model_*.png`, `shap_*.png` |

## 실행 환경

```bash
pip install -r ../requirements.txt
```

## 재사용 가능한 함수

노트북의 핵심 로직은 `src/` 모듈로 분리되어 있습니다.

```python
from src.feature_engineering import build_final_dataset
from src.modeling import prepare_data, train_random_forest, plot_shap_summary
```

## Figure 저장

각 노트북 실행 시 `outputs/figures/` 에 자동 저장됩니다.

| 노트북 | 저장 파일 |
|--------|----------|
| 03_eda | `eda_demographics.png`, `eda_pca_tsne.png`, `eda_kmeans.png`, `eda_cluster_viz.png` 외 |
| 04_modeling | `model_comparison.png`, `actual_vs_predicted.png`, `shap_summary.png`, `shap_group_contribution.png`, `shap_dependence.png`, `residual_analysis.png` |
