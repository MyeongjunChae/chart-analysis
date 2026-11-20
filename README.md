# 차트 분석

OHLCV 가격 CSV를 업로드해 캔들 차트, 자동 패턴 감지, 과거 패턴 빈도, 근사 거래량 프로파일, 피보나치 되돌림, 몬테카를로 시나리오를 확인하는 로컬 Streamlit 앱입니다.

## 실행 방법 (Windows PowerShell)

    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    streamlit run app.py

## CSV 형식

시간 열은 선택 사항입니다. 시가(Open), 고가(High), 저가(Low), 종가(Close)가 필요하고 거래량(Volume)은 매물대 계산에 사용합니다. TradingView에서 내보낸 CSV를 올리거나 왼쪽 메뉴에서 열 이름을 직접 연결하세요.

## 감지 패턴

- 캔들: 도지, 슈팅스타, 망치형, 상승/하락 장악형
- 차트 구조: 상승 삼각형, 상승/하락 쐐기, 베어 페넌트, 헤드앤숄더와 역헤드앤숄더
- 근사 POC/VAH/VAL, 스윙 피보나치, 과거 패턴 방향 빈도와 95% Wilson 구간
- 최근 로그수익률 부트스트랩 몬테카를로 경로와 분위수

패턴은 규칙 기반 근사라 오탐과 미탐이 생깁니다. 표본 빈도와 시뮬레이션은 미래 가격이나 수익을 보장하지 않으며 매매 권유가 아닙니다.
