"""
feature_engineering.py
=======================
EMR 검사 데이터로부터 입원 기간(LOS) 예측에 유효한 구조적 피처를 생성합니다.

파생 변수 설계 원칙
-------------------
단일 시점 수치보다 환자의 '전반적 생리적 불안정성'을 포착하는 지표에 집중합니다.
  - Trend        : 검사 수치의 시간적 변화 방향 (선형회귀 기울기)
  - Variability  : 검사 수치의 내부 변동성 (표준편차)
  - Abnormal Sum : 이상 소견 누적 횟수

⚠️ 데이터 구조 주의사항
-----------------------
AdmissionID는 전역 고유 ID가 아닌 환자별 입원 순번(1, 2, 3...)입니다.
따라서 모든 조인은 반드시 ['PatientID', 'AdmissionID'] 복합키 기준으로 수행해야 합니다.
AdmissionID 단독 조인 시 1:N 카르테시안 곱 발생 → 행 폭발 (111,483 → 8,676,187)

실제 출력 피처 구성 (143개 모델 피처)
--------------------------------------
  - Lab_Mean_{검사명}    : 35종 × 1 = 35개
  - Lab_Std_{검사명}     : 35종 × 1 = 35개
  - Lab_Trend_{검사명}   : 35종 × 1 = 35개
  - Abnormal_Sum_{검사명}: 35종 × 1 = 35개
  - Total_Abnormal_Count : 1개  (환자 전체 입원의 이상 소견 합)
  - Total_Lab_Variability: 1개  (환자 단위 평균 변동성)
  - LabTestVariety       : 1개  (검사 종류 수)
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------

def load_raw_tables(data_dir: str = "data/raw") -> dict:
    """
    원본 EMR 4개 테이블(.txt, 탭 구분, UTF-8 BOM)을 로드합니다.

    Parameters
    ----------
    data_dir : str
        원본 .txt 파일이 위치한 디렉토리 경로 (기본: "data/raw")

    Returns
    -------
    dict
        {"patients": df, "admissions": df, "diagnoses": df, "labs": df}

    Notes
    -----
    - 파일 형식: 탭(\\t) 구분, 첫 행이 헤더, UTF-8 BOM(utf-8-sig)
    - AdmissionID는 환자별 입원 순번 → 복합키 ['PatientID','AdmissionID'] 필수
    """
    file_map = {
        "patients":   "PatientCorePopulatedTable.txt",
        "admissions": "AdmissionsCorePopulatedTable.txt",
        "diagnoses":  "AdmissionsDiagnosesCorePopulatedTable.txt",
        "labs":       "LabsCorePopulatedTable.txt",
    }

    tables = {}
    for name, filename in file_map.items():
        path = os.path.join(data_dir, filename)
        tables[name] = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
        print(f"   {name}: {tables[name].shape[0]:,}행 × {tables[name].shape[1]}열")

    print(f"✅ 테이블 로드 완료 (총 {len(tables)}개)")
    print(f"   ※ AdmissionID unique: {tables['admissions']['AdmissionID'].nunique()}종 (환자별 순번)")
    return tables


# ---------------------------------------------------------------------------
# 2. 시간 파생 변수
# ---------------------------------------------------------------------------

def compute_time_features(admissions: pd.DataFrame) -> pd.DataFrame:
    """
    입원 테이블에 날짜 기반 파생 변수를 추가합니다.

    생성 컬럼
    ---------
    - AdmissionYear, AdmissionMonth, AdmissionDayOfWeek, AdmissionDayName
    - LengthOfStay (입원 기간, 일 단위) — 타겟 변수
    - LengthOfStayCategory (Short ≤7일 / Medium ≤14일 / Long >14일)
    """
    df = admissions.copy()
    df["AdmissionStartDate"] = pd.to_datetime(df["AdmissionStartDate"])
    df["AdmissionEndDate"]   = pd.to_datetime(df["AdmissionEndDate"])

    df["AdmissionYear"]      = df["AdmissionStartDate"].dt.year
    df["AdmissionMonth"]     = df["AdmissionStartDate"].dt.month
    df["AdmissionDayOfWeek"] = df["AdmissionStartDate"].dt.dayofweek
    df["AdmissionDayName"]   = df["AdmissionStartDate"].dt.day_name()

    df["LengthOfStay"] = (
        df["AdmissionEndDate"] - df["AdmissionStartDate"]
    ).dt.days

    df["LengthOfStayCategory"] = pd.cut(
        df["LengthOfStay"],
        bins=[0, 7, 14, float("inf")],
        labels=["Short", "Medium", "Long"],
    )
    return df


# ---------------------------------------------------------------------------
# 3. 검사 단위 파생 변수
# ---------------------------------------------------------------------------

def compute_hours_after_admission(labs: pd.DataFrame,
                                  admissions: pd.DataFrame) -> pd.DataFrame:
    """
    각 검사가 입원 후 몇 시간 경과 시점에 수행되었는지 계산합니다.

    ⚠️ 복합키 ['PatientID','AdmissionID']로 merge합니다.
    AdmissionID 단독 merge 시 행 폭발이 발생합니다.
    """
    df = labs.merge(
        admissions[["PatientID", "AdmissionID", "AdmissionStartDate"]],
        on=["PatientID", "AdmissionID"],   # ← 복합키 필수
        how="left",
    )
    df["LabDateTime"]        = pd.to_datetime(df["LabDateTime"])
    df["AdmissionStartDate"] = pd.to_datetime(df["AdmissionStartDate"])

    df["HoursAfterAdmission"] = (
        (df["LabDateTime"] - df["AdmissionStartDate"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    return df.drop(columns=["AdmissionStartDate"])


def _compute_trend(g: pd.DataFrame) -> float:
    """단순 선형회귀 기울기로 검사 수치의 시간적 변화 방향을 계산합니다."""
    v, t = g["LabValue"], g["HoursAfterAdmission"]
    if len(v) < 2 or t.nunique() < 2:
        return 0.0
    slope, *_ = linregress(t, v)
    return slope


# ---------------------------------------------------------------------------
# 4. 입원 × 검사명별 Wide Format 집계 (핵심)
# ---------------------------------------------------------------------------

def aggregate_lab_features(labs: pd.DataFrame) -> pd.DataFrame:
    """
    검사(Lab) 로우 데이터를 입원(PatientID+AdmissionID) × 검사명(LabName) 단위로
    집계한 뒤 Wide Format으로 변환합니다. (노트북 코드와 동일한 로직)

    생성 컬럼
    ---------
    - Lab_Mean_{검사명}    : 검사 수치 평균
    - Lab_Std_{검사명}     : 검사 수치 표준편차 (변동성)
    - Lab_Trend_{검사명}   : 입원 기간 내 수치 변화 기울기
    - Abnormal_Sum_{검사명}: 이상 소견(±1.5σ 기준) 누적 횟수
    - LabTestCount         : 총 검사 시행 횟수 (역인과 위험 — 모델 피처 제외 권장)
    - LabTestVariety       : 시행된 검사 종류 수

    이상치 기준
    -----------
    LabName별 전체 평균 ± 1.5 표준편차를 벗어나면 이상(IsAbnormal=1)으로 판별합니다.
    """
    # ── 이상치 판별 기준 생성 ──────────────────────────────────
    thresholds = (
        labs.groupby("LabName")["LabValue"]
        .agg(["mean", "std"])
        .reset_index()
    )
    thresholds["_upper"] = thresholds["mean"] + 1.5 * thresholds["std"]
    thresholds["_lower"] = thresholds["mean"] - 1.5 * thresholds["std"]

    labs = labs.merge(
        thresholds[["LabName", "_upper", "_lower"]], on="LabName", how="left"
    )
    labs["IsAbnormal"] = (
        (labs["LabValue"] > labs["_upper"]) | (labs["LabValue"] < labs["_lower"])
    ).astype(int)

    # ── 검사명별 기본 집계 (long format) ─────────────────────
    grp = labs.groupby(["PatientID", "AdmissionID", "LabName"])   # ← 복합키

    df_lab_features = grp.agg(
        Lab_Mean    =("LabValue",    "mean"),
        Lab_Std     =("LabValue",    "std"),
        Abnormal_Sum=("IsAbnormal",  "sum"),
    ).reset_index()

    # Trend: 입원 × 검사명별 선형 기울기
    trend_vals = (
        grp.apply(_compute_trend, include_groups=False)
        .rename("Lab_Trend")
        .reset_index()
    )
    df_lab_features = df_lab_features.merge(
        trend_vals, on=["PatientID", "AdmissionID", "LabName"], how="left"
    )
    df_lab_features["Lab_Std"]   = df_lab_features["Lab_Std"].fillna(0)
    df_lab_features["Lab_Trend"] = df_lab_features["Lab_Trend"].fillna(0)

    # ── Wide Format 변환 (노트북과 동일) ─────────────────────
    master_wide = df_lab_features.pivot_table(
        index=["PatientID", "AdmissionID"],   # ← 복합키
        columns="LabName",
        values=["Lab_Mean", "Lab_Std", "Abnormal_Sum", "Lab_Trend"],
    )
    master_wide.columns = [f"{col[0]}_{col[1]}" for col in master_wide.columns]
    master_wide = master_wide.reset_index()

    # ── 요약 변수 (입원 단위) ──────────────────────────────────
    df_lab_counts  = (
        labs.groupby(["PatientID", "AdmissionID"])
        .agg(LabTestCount=("LabValue", "count"))
        .reset_index()
    )
    df_lab_variety = (
        labs.groupby(["PatientID", "AdmissionID"])
        .agg(LabTestVariety=("LabName", "nunique"))
        .reset_index()
    )

    master_wide = master_wide.merge(df_lab_counts,  on=["PatientID", "AdmissionID"], how="left")
    master_wide = master_wide.merge(df_lab_variety, on=["PatientID", "AdmissionID"], how="left")
    master_wide["LabTestCount"]   = master_wide["LabTestCount"].fillna(0).astype(int)
    master_wide["LabTestVariety"] = master_wide["LabTestVariety"].fillna(0).astype(int)

    return master_wide


# ---------------------------------------------------------------------------
# 5. 최종 통합 데이터셋 생성
# ---------------------------------------------------------------------------

def build_final_dataset(tables: dict) -> pd.DataFrame:
    """
    4개 테이블을 병합하여 모델링용 최종 데이터셋을 생성합니다. (노트북 순서 동일)

    병합 순서
    ---------
    Step 1 : admissions + patients          (on='PatientID')
    Step 2 : + diagnoses                    (on=['PatientID','AdmissionID'])
    Step 3 : + master_wide (Wide format)    (on=['PatientID','AdmissionID'], how='inner')
    Step 4 : 통합 중증도 지표 생성
             - Total_Abnormal_Count  = Abnormal_Sum 계열 합산
             - Total_Lab_Variability = Lab_Std 계열 평균

    Returns
    -------
    pd.DataFrame  372행 × ~156열
    """
    patients   = tables["patients"]
    admissions = tables["admissions"]
    diagnoses  = tables["diagnoses"]
    labs       = tables["labs"]

    # 2. 시간 파생 변수
    admissions = compute_time_features(admissions)

    # 3. 검사 시간 경과 (복합키 merge)
    labs = compute_hours_after_admission(labs, admissions)

    # 4. Wide Format 집계
    master_wide = aggregate_lab_features(labs)

    # 5. 메인 병합 (노트북과 동일한 순서)
    df_main = admissions.merge(patients,    on="PatientID",                  how="left")
    df_main = df_main.merge(diagnoses,      on=["PatientID", "AdmissionID"], how="left")
    df_main = df_main.merge(master_wide,    on=["PatientID", "AdmissionID"], how="inner")

    # 6. 통합 중증도 지표 (노트북과 동일)
    df_main["Total_Abnormal_Count"]  = df_main.filter(like="Abnormal_Sum").sum(axis=1)
    df_main["Total_Lab_Variability"] = df_main.filter(like="Lab_Std").mean(axis=1)

    print(f"✅ 최종 데이터셋 생성 완료: {df_main.shape[0]}행 × {df_main.shape[1]}열")
    print(f"   입원당 평균 이상 소견: {df_main['Total_Abnormal_Count'].mean():.1f}건")
    print(f"   전체 평균 Lab 가변성: {df_main['Total_Lab_Variability'].mean():.2f}")
    return df_main


def save_processed(df: pd.DataFrame,
                   output_path: str = "data/processed/processed_healthcare_data.csv") -> None:
    """전처리 완료 데이터셋을 CSV로 저장합니다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {output_path}  ({df.shape[0]}행 × {df.shape[1]}열)")


# ---------------------------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tables = load_raw_tables("data/raw")
    df     = build_final_dataset(tables)
    save_processed(df)
