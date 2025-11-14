# OLS Analysis Toolkit

이 저장소는 "공공기여와 인센티브가 사업 추진 기간에 미치는 영향" 연구를 위한 데이터 구축 및 회귀 분석 자동화 도구입니다. 전체 파이프라인은 다음 네 단계로 구성되며 `data/` 폴더 하위에 중간 산출물이 저장됩니다.

1. **데이터 전처리** → `data/step1_data/`
2. **파생변수 생성** → `data/step2_data/`
3. **기초 통계 산출** → `data/basic_statistic/`
4. **회귀 분석(OLS)** → `data/ols_result/`

현재는 1단계 전처리 스크립트가 구현되어 있으며, 이후 단계용 파일은 비워둔 상태입니다.

## 환경 설정

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: 데이터 전처리 실행

```powershell
python src\data_preprocessing.py --input data\raw_data\251114 신속통합기획_DB_1951.csv
```

인자를 생략하면 `data/raw_data` 폴더에서 가장 최근 CSV를 자동으로 사용합니다. 전처리 결과는 `data/step1_data/step1_preprocessed_<원본파일명>_<타임스탬프>.csv` 형태로 저장됩니다.

> **결측치 처리 기준**: 아래 열만 필수로 판단하며, 이 중 하나라도 비어 있으면 해당 행이 제거됩니다. 그 외 열은 비어 있어도 유지됩니다.
> 
> - `DGM_NM`, `권역`, `자치구`, `법정동`, `후보지-지정고시`, `신속통합기획 후보지 선정일`, `사례유형`
> - `정비구역면적(㎡)`, `용도지역`, `택지면적(㎡)`
> - `Zone1`, `Zone1_면적`, `Zone1_기준용적률(%이하)`, `Zone1_법적상한용적률`
> - `높이(m)`, `지상층수`, `총 세대수`, `임대세대총수`

## Step 2: 파생변수 생성 실행

```powershell
python src\feature_engineering.py --input "data\step1_data\step1_preprocessed_*.csv"
```

입력을 생략하면 `data/step1_data`에서 가장 최신 CSV가 자동 선택됩니다. 결과 파일은 `data/step2_data/step2_features_<원본파일명>_<타임스탬프>.csv`로 저장되며 아래 변수 3개가 추가됩니다.

- `duration_months_initial`: `duration_days_initial ÷ 30` 값을 월 단위로 환산(소수 셋째 자리 반올림)
- `zone_weighted_far`: 각 Zone 면적 가중치(Zone1~3 면적)와 용적률(계획 → 기준 → 법정 순)로 계산한 가중 평균
- `unit_rent_ratio`: `임대세대총수 ÷ 총 세대수` (총 세대수 또는 임대가 0/결측이면 NA)

## Step 3: 기초 통계 실행

```powershell
python src\basic_statistics.py --input "data\step2_data\step2_features_*.csv"
```

`data/step2_data`에서 최신 파일을 자동으로 선택할 수 있으며, 수치형 컬럼만 골라 최소/최대/평균/표준편차를 계산합니다. 결과 표는 `data/basic_statistic/basic_statistic_<원본파일명>_<타임스탬프>.csv`로 저장됩니다.

## Step 4: 기간-지표 상관관계 분석

```powershell
python src\correlation_analysis.py --input "data\step2_data\step2_features_*.csv"
```

`duration_months_initial`을 기준으로 모든 수치형 지표와의 피어슨 상관계수, t-stat, p-value(양측 t-test)를 계산합니다. 결과는 `data/correlation_result/correlation_duration_<원본파일명>_<타임스탬프>.csv`로 저장됩니다.
