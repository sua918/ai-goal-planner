# AI Goal Planner

자연어로 목표, 마감, 고정 일정, 사용 가능한 시간을 입력하면  
LLM이 조건을 해석하고 실행 가능한 일정으로 구성하는 LangChain 기반 AI 플래너입니다.

## Demo

https://ai-goal-planner-lswpkuu5jaztfuhtddnhd8.streamlit.app/

## 주요 기능

- 자연어 목표 및 일정 조건 해석
- 목표 우선순위 및 일정 생성
- 고정 일정 / 가용 시간 / 충돌 검증
- 검증 실패 시 LLM 기반 일정 수정
- 주·일·목록 캘린더 보기
- `.ics` 캘린더 파일 다운로드

## 처리 흐름

사용자 입력  
→ Interpretation LLM  
→ `PlanSpec`  
→ Planner LLM  
→ `DraftSchedule`  
→ Python Validator  
→ 필요 시 Repair LLM  
→ 최종 일정

## LangChain

- `ChatPromptTemplate`
- Structured Output + Pydantic
- LCEL
- `RunnableLambda`

## Tech Stack

- Python
- LangChain
- Gemini
- Streamlit
- Pydantic
- streamlit-calendar

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py