# data/

원본 EMR 테이블 (Synthea 생성, 변경 금지).

```
data/
├── PatientCorePopulatedTable.txt           # 환자 인구통계
├── AdmissionsCorePopulatedTable.txt        # 입원 기록
├── AdmissionsDiagnosesCorePopulatedTable.txt  # 진단 코드
└── LabsCorePopulatedTable.txt              # 검사 결과 (111,483건)
```

## 데이터 출처

- 원본: [csbond007/Basic_Health_Care](https://github.com/csbond007/Basic_Health_Care)
- 기간: 1941–2013년 / 환자 100명 / 입원 372건

## 주요 컬럼

| 테이블 | 주요 컬럼 |
|--------|----------|
| Patient | PatientID, PatientGender, PatientDateOfBirth, PatientRace, PatientMaritalStatus |
| Admissions | AdmissionID, PatientID, AdmissionStartDate, AdmissionEndDate → **LengthOfStay** |
| Diagnoses | AdmissionID, PrimaryDiagnosisCode, PrimaryDiagnosisDescription |
| Labs | PatientID, AdmissionID, LabName, LabValue, LabUnits, LabDateTime |

## 주의사항

- 이 폴더의 파일은 **절대 수정하지 않습니다.**
- 전처리 완료 데이터는 `outputs/processed_healthcare_data.csv` 에 저장됩니다.
