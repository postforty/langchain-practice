"""
====================================================================
동적 모델 사용 사례: 에러 복구 및 툴 사용 실패 대응을 위한 동적 모델 전환
====================================================================
모델이 툴을 잘못 호출하거나, 
툴 사용 후 받은 관찰(Observation)을 통해 결론을 내리지 못하고 루프에 빠질 때 (Hallucination 또는 Reasoning Failure),
더 강력한 고급 모델로 전환하여 문제를 해결하고 정확한 답변을 유도합니다.
"""

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


# 1. 커스텀 상태 정의: 복구 시도 횟수를 추적
class RecoveryState(AgentState):
    # AgentState는 messages를 자동으로 포함합니다.
    recovery_attempts: int = 0  # 복구 시도 횟수


# 2. 모델 인스턴스 정의
basic_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
advanced_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
MAX_ATTEMPTS = 2  # 복구 모델 전환 전 최대 시도 횟수


@wrap_model_call
def recovery_router(request: ModelRequest, handler) -> ModelResponse:
    """에이전트가 툴 사용 루프에 빠졌을 때 모델을 전환하여 복구합니다."""

    # 3. 현재 상태 가져오기
    state: RecoveryState = request.state

    # 4. 루프 또는 추론 실패 감지
    # 마지막 두 메시지가 ToolMessage와 AIMessage이고, ToolMessage의 내용이 비어있지 않다면 (성공적인 관찰),
    # 그 이전 메시지를 분석하여 모델이 '헤매고' 있는지 확인해야 합니다.
    # 간단화를 위해, 현재 예제에서는 복구 시도 횟수가 MAX_ATTEMPTS를 초과할 경우를 '실패'로 가정합니다.
    if state["recovery_attempts"] >= MAX_ATTEMPTS:
        # 5. 복구 로직: 고급 모델로 전환하여 문제 해결을 시도
        request.model = advanced_model
        print(
            f"--- ⚠️ WARNING: {MAX_ATTEMPTS}회 이상 실패 감지! '{advanced_model.model}'로 긴급 복구 전환합니다. ---")
    else:
        # 일반적인 상황에서는 기본 모델을 사용
        request.model = basic_model
        print(f"--- INFO: 정상 작동 중. '{basic_model.model}'을 사용합니다. ---")

    # 6. 복구 시도 횟수 업데이트 (미들웨어의 핵심 역할)
    # 여기에서는 실습의 단순화를 위해 이 로직을 생략하지만
    # 에이전트가 툴 호출에 실패하거나 추론 루프에 빠지는 상황이 발생할 때마다 이 값을 증가시켜야 합니다.
    # state["recovery_attempts"] += 1

    return handler(request)

# 7. 더미 도구 및 에이전트 설정
@tool
def check_status(item: str) -> str:
    """상품의 재고 상태를 확인합니다. 특정 입력에 대해 에러를 발생시킵니다."""
    if "에러" in item:
        raise ValueError(f"상품 '{item}'의 재고 조회 중 연결 오류가 발생했습니다.")
    return f"상품 '{item}': 재고 10개 있음."


tools = [check_status]

# 8. 에이전트 생성 (커스텀 상태 스키마 포함)
agent = create_agent(
    model=basic_model,
    tools=tools,
    middleware=[recovery_router],
    state_schema=RecoveryState  # 커스텀 상태 스키마 연결
)

# 9. 실습 실행: 초기 시도 (기본 모델 예상)
print("\n=== 실습 1: 초기 질문 (복구 시도 0회) ===")
# recovery_attempts가 0이므로 gpt-4o-mini 사용 예상
initial_state = {"messages": [
    {"role": "user", "content": "상품 'A' 재고를 확인해줘."}], "recovery_attempts": 0} # recovery_attempts 0으로 하드코딩
result_initial = agent.invoke(initial_state)

# 10. 실습 실행: 실패 시뮬레이션 (고급 모델 전환 예상)
print("\n=== 실습 2: 실패 시뮬레이션 (복구 시도 2회 이상) ===")
# 툴 사용 실패 루프가 발생했다고 가정하고, recovery_attempts를 2로 설정합니다.
failure_state = {"messages": [
    {"role": "user", "content": "상품 'B' 재고를 확인해줘."}], "recovery_attempts": 2} # recovery_attempts 2로 하드코딩
result_failure = agent.invoke(failure_state)

print("\n초기 시도 응답:", result_initial["messages"][-1].content)
print("실패 시뮬레이션 응답:", result_failure["messages"][-1].content)
