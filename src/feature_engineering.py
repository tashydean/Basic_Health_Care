"""
feature_engineering.py
=======================
EMR 검사 데이터로부터 입원 기간(LOS) 예측에 유효한 구조적 피처를 생성합니다.

파생 변수 설계 원칙
-------------------
단일 시점 수치보다 환자의 '전반적 생리적 불안정성'을 포착하는 지표에 집중합니다.
  - Trend        : 검사 수치의 시간적 변화 방향
  - Variability  : 검사 수치의 내부 변동성
  - Abnormal Count: 이상 소견 누적 횟수
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# 1. 데이터 로드
# ---------------------------------------------------------------------------

def load_raw_tables(data_dir: str = "data/raw") -> dict:
    """
    원본 EMR 4개 테이블을 로드합니다.

    Parameters
    ----------
    data_dir : str
        원본 CSV 파일이 위치한 디렉토리 경로

    Returns
    -------
    dict
        {"patients": df, "admissions": df, "diagnoses": df, "labs": df}
    """
    tables = {
        "patients":   pd.read_csv(f"{data_dir}/patients.csv"),
        "admissions": pd.read_csv(f"{data_dir}/admissions.csv"),
        "diagnoses":  pd.read_csv(f"{data_dir}/diagnoses.csv"),
        "labs":       pd.read_csv(f"{data_dir}/labs.csv"),
    }
    print(f"✅ 테이블 로드 완료")
    for name, df in tables.items():
        print(f"   {name}: {df.shape[0]:,}행 × {df.shape[1]}열")
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
    - LengthOfStay (입원 기간, 일 단위)
    - LengthOfStayCategory (Short / Medium / Long)
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

    Parameters
    ----------
    labs : pd.DataFrame
        검사 테이블 (AdmissionID, LabDateTime 등 포함)
    admissions : pd.DataFrame
        입원 테이블 (AdmissionID, AdmissionStartDate 포함)

    Returns
    -------
    pd.DataFrame
        HoursAfterAdmission 컬럼이 추가된 labs 테이블
    """
    df = labs.merge(
        admissions[["AdmissionID", "AdmissionStartDate"]],
        on="AdmissionID",
        how="left",
    )
    df["LabDateTime"]        = pd.to_datetime(df["LabDateTime"])
    df["AdmissionStartDate"] = pd.to_datetime(df["AdmissionStartDate"])

    df["HoursAfterAdmission"] = (
        (df["LabDateTime"] - df["AdmissionStartDate"]).dt.total_seconds() / 3600
    ).clip(lower=0)

    return df.drop(columns=["AdmissionStartDate"])


def compute_trend(group: pd.Series, time_col: pd.Series) -> float:
    """
    단순 선형회귀 기울기로 검사 수치의 시간적 변화 방향(Trend)을 계산합니다.

    시점이 2개 미만이거나 시간 축에 분산이 없으면 0.0을 반환합니다.
    """
    if len(group) < 2 or time_col.nunique() < 2:
        return 0.0
    slope, *_ = linregress(time_col, group)
    return slope


# ---------------------------------------------------------------------------
# 4. 입원 단위 집계 피처
# ---------------------------------------------------------------------------

def aggregate_lab_features(labs: pd.DataFrame) -> pd.DataFrame:
    """
    검사(Lab) 로우 데이터를 입원(AdmissionID) 단위로 집계합니다.

    생성 컬럼
    ---------
    - Lab_Mean         : 검사 수치 평균
    - Lab_Std          : 검사 수치 표준편차 (변동성)
    - Lab_Trend        : 입원 기간 내 검사 수치 변화 기울기
    - Abnormal_Sum     : 이상 소견(ResultFlag) 누적 횟수
    - LabTestCount     : 총 검사 시행 횟수
    - LabTestVariety   : 시행된 검사 종류 수
    """
    agg = (
        labs.groupby("AdmissionID")
        .agg(
            Lab_Mean    =("LabValue",      "mean"),
            Lab_Std     =("LabValue",      "std"),
            Abnormal_Sum=("ResultFlag",    "sum"),
            LabTestCount=("LabValue",      "count"),
            LabTestVariety=("LabName",     "nunique"),
        )
        .reset_index()
    )

    # Trend: 입원별 선형 기울기 계산
    trend_vals = (
        labs.groupby("AdmissionID")
        .apply(lambda g: compute_trend(g["LabValue"], g["HoursAfterAdmission"]))
        .rename("Lab_Trend")
        .reset_index()
    )
    agg = agg.merge(trend_vals, on="AdmissionID", how="left")
    agg["Lab_Std"] = agg["Lab_Std"].fillna(0)

    return agg


# ---------------------------------------------------------------------------
# 5. 환자 단위 종합 지표
# ---------------------------------------------------------------------------

def compute_patient_level_features(lab_agg: pd.DataFrame,
                                   admissions: pd.DataFrame) -> pd.DataFrame:
    """
    입원 단위 집계를 환자(PatientID) 단위로 재집계하여
    전반적 생리적 불안정성 지표를 생성합니다.

    생성 컬럼
    ---------
    - Total_Abnormal_Count  : 전체 입원에서 이상 소견 누적 합
    - Total_Lab_Variability : 입원별 Lab_Std의 평균 (전반적 변동성)
    """
    merged = lab_agg.merge(
        admissions[["AdmissionID", "PatientID"]],
        on="AdmissionID",
        how="left",
    )
    patient_feats = (
        merged.groupby("PatientID")
        .agg(
            Total_Abnormal_Count =("Abnormal_Sum", "sum"),
            Total_Lab_Variability=("Lab_Std",       "mean"),
        )
        .reset_index()
    )
    return patient_feats


# ---------------------------------------------------------------------------
# 6. 최종 통합 데이터셋 생성
# ---------------------------------------------------------------------------

def build_final_dataset(tables: dict) -> pd.DataFrame:
    """
    4개 테이블을 병합하여 모델링용 최종 데이터셋을 생성합니다.

    Parameters
    ----------
    tables : dict
        load_raw_tables()의 반환값

    Returns
    -------
    pd.DataFrame
        processed_healthcare_data.csv 로 저장할 최종 데이터셋
    """
    patients   = tables["patients"]
    admissions = tables["admissions"]
    diagnoses  = tables["diagnoses"]
    labs       = tables["labs"]

    # 2. 시간 파생 변수 추가
    admissions = compute_time_features(admissions)

    # 3. 검사 시간 경과 계산
    labs = compute_hours_after_admission(labs, admissions)

    # 4. 입원 단위 집계
    lab_agg = aggregate_lab_features(labs)

    # 5. 환자 단위 지표
    patient_feats = compute_patient_level_features(lab_agg, admissions)

    # 6. 병합
    df = (
        admissions
        .merge(patients,      on="PatientID",   how="left")
        .merge(diagnoses,     on="AdmissionID", how="left")
        .merge(lab_agg,       on="AdmissionID", how="left")
        .merge(patient_feats, on="PatientID",   how="left")
    )

    print(f"✅ 최종 데이터셋 생성 완료: {df.shape[0]}행 × {df.shape[1]}열")
    return df


def save_processed(df: pd.DataFrame,
                   output_path: str = "outputs/processed_healthcare_data.csv") -> None:
    """전처리 완료 데이터셋을 CSV로 저장합니다."""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {output_path}")


# ---------------------------------------------------------------------------
# 실행 예시
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tables = load_raw_tables("data/raw")
    df     = build_final_dataset(tables)
    save_processed(df)
