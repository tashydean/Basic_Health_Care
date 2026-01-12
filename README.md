# 🏥 EMR to Image: Diagnostic Classification Project

### [Abstract]
The goal of this project is to classify patient diagnoses by transforming structured Electronic Medical Record (EMR) data into image formats. To achieve high-accuracy deep learning classification, the project follows a systematic pipeline: individual dataset exploration, data integration into a unified patient-centric format, image transformation, and finally, CNN-based diagnosis classification.

### [국문 요약]
본 프로젝트는 정형 데이터인 EMR(전자의무기록)을 이미지 형식으로 변환하여 진단명을 자동으로 분류하는 딥러닝 모델 개발을 목표로 합니다. 이를 위해 각 데이터셋의 특성을 탐색(EDA)하고, 환자별로 데이터를 통합한 뒤, 딥러닝 모델이 학습 가능한 이미지 형태로 변환하는 파이프라인을 구축합니다.

---

## 🎯 프로젝트 목표 및 현재 진행 상황 (Milestones & Status)

이 프로젝트의 최종 목표는 **텍스트 기반의 EMR 데이터를 시각화된 이미지로 변환하여, 이미지 분류 모델(CNN 등)을 통해 환자의 진단명을 정확히 예측하는 것**입니다.

**현재 진행률: 🟢 40% (데이터 확보 및 기초 탐색 단계)**

| 단계 | 목표 작업 (Task) | 상태 (Status) |
| :--- | :--- | :--- |
| **1단계** | 4개 핵심 데이터셋(Admissions, Diagnoses, Labs, Patients) 다운로드 및 로드 | ✅ 완료 |
| **2단계** | 데이터셋별 항목 분석 및 전처리 (EDA) | ✅ 완료 |
| **3단계** | 환자 ID와 입원 ID 기준 4개 테이블 통합 (Data Integration) | 🏃 진행 중 |
| **4단계** | 통합 데이터를 이용한 이미지 형식 변환 (Tabular to Image Transformation) | 📅 예정 |
| **5단계** | 딥러닝 모델 학습 및 진단명 분류 (CNN Classification) | 📅 예정 |

---

## 🚀 주요 특징 (Key Features)
- **하이브리드 로드**: `LOCAL_MODE` 설정을 통해 로컬/원격 데이터를 유연하게 사용하여 분석 환경에 구애받지 않음
- **데이터 통합 전략**: 여러 테이블에 흩어진 환자 정보를 이미지 한 장에 담기 위한 전처리 로직 포함
- **딥러닝 친화적 설계**: 정형 데이터를 비정형 데이터(이미지)로 변환하는 특수 파이프라인 구축 예정
