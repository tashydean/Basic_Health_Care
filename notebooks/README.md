# notebooks/

분석 단계별 Jupyter Notebook. **번호 순서대로 실행.**

| 파일 | 내용 | 주요 출력 |
|------|------|-----------|
| `01_data_overview.ipynb` | 데이터 탐색 — 결측치·분포·테이블 관계 확인 | — |
| `02_feature_engineering.ipynb` | 143개 구조적 파생변수 생성 | `outputs/processed_healthcare_data.csv` |
| `03_exploratory_data_analysis.ipynb` | PCA / t-SNE / K-Means 군집화 | `outputs/figures/eda_*.png` |
| `04_modeling.ipynb` | LR / RF / XGBoost 비교 + SHAP 해석 | `outputs/figures/` |
| `appendix_detailed_analysis.ipynb` | 변수별 상세 탐색 (참고용, 독립 실행 가능) | — |

```bash
pip install -r ../requirements.txt
jupyter notebook 01_data_overview.ipynb
```
