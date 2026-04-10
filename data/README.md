# data/

원본 EMR 테이블 (Synthea 생성, 변경 금지).

```
data/
├── raw/
│   ├── PatientCorePopulatedTable.txt               # 환자 인구통계
│   ├── AdmissionsCorePopulatedTable.txt            # 입원 기록
│   ├── AdmissionsDiagnosesCorePopulatedTable.txt   # 진단 코드
│   └── LabsCorePopulatedTable.txt                  # 검사 결과 (111,483건)
└── processed/
    └── processed_healthcare_data.csv               # 전처리 완료 (372행 × 165열)
```

**데이터 출처**: [csbond007/Basic_Health_Care](https://github.com/csbond007/Basic_Health_Care) — Synthea 합성 EMR

**주의**: AdmissionID는 전역 고유 ID가 아닌 **환자별 입원 순번**.<br>
모든 병합은 `['PatientID', 'AdmissionID']` 복합키로 수행 (단독 조인 시 행 폭발).
