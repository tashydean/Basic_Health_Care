# src/

노트북에서 검증된 핵심 로직을 재사용 가능한 Python 모듈로 분리.

| 파일 | 역할 |
|------|------|
| `feature_engineering.py` | 데이터 로드 → 143개 파생변수 생성 → CSV 저장 |
| `modeling.py` | 3모델 학습 → 평가 → SHAP 시각화 |

**주의**: `temp.py`는 개발 중 임시 스크립트로 참고용. 실제 실행은 위 두 파일 사용.

```python
from src.feature_engineering import load_raw_tables, build_final_dataset, save_processed
from src.modeling import prepare_data, train_xgboost, plot_shap_summary

tables = load_raw_tables("data/raw")
df     = build_final_dataset(tables)
save_processed(df)
```
