# data/

원본 EMR 테이블 및 전처리 완료 데이터셋을 보관합니다.

```
data/
├── raw/                          # 원본 CSV (변경 금지)
│   ├── patients.csv              # 환자 인구통계 정보
│   ├── admissions.csv            # 입원 기록 (입퇴원일, LOS 등)
│   ├── diagnoses.csv             # 진단 코드 및 설명
│   └── labs.csv                  # 검사 결과 (111,483건)
└── processed/
    └── processed_healthcare_data.csv   # 4개 테이블 병합 + 파생변수 포함 최종 데이터셋
```

## 데이터 출처

- 원본: [csbond007/Basic_Health_Care](https://github.com/csbond007/Basic_Health_Care)
- 기간: 1941–2013년 / 환자 100명 / 입원 372건 / 검사 111,483건

## 주의사항

- `raw/` 폴더의 파일은 **절대 수정하지 않습니다.**
- `processed/`의 파일은 `src/feature_engineering.py` 또는 `notebooks/02_feature_engineering.ipynb` 실행으로 재생성 가능합니다.
- 개인정보 포함 가능성 있는 원본 데이터는 `.gitignore`에 추가하여 업로드를 제한합니다.

## 주요 컬럼 설명 (processed)

| 컬럼 | 설명 |
|------|------|
| `PatientID` / `AdmissionID` | 병합 키 |
| `LengthOfStay` | 입원 기간 (타겟 변수, 일 단위) |
| `Lab_Mean`, `Lab_Std` | 검사 수치 기초 통계 |
| `Lab_Trend` | 입원 기간 내 검사 수치 변화 기울기 |
| `Abnormal_Sum` | 입원별 이상 소견 누적 횟수 |
| `Total_Abnormal_Count` | 환자 전체 이상 소견 합 **(핵심 피처)** |
| `Total_Lab_Variability` | 환자 단위 전반적 변동성 **(핵심 피처)** |
