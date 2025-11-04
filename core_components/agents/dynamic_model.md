# Agents
에이전트

#### Dynamic model
동적 모델

Dynamic models are selected at <Tooltip tip="The execution environment of your agent, containing immutable configuration and contextual data that persists throughout the agent's execution (e.g., user IDs, session details, or application-specific configuration).">runtime</Tooltip> based on the current <Tooltip tip="The data that flows through your agent's execution, including messages, custom fields, and any information that needs to be tracked and potentially modified during processing (e.g., user preferences or tool usage stats).">state</Tooltip> and context. This enables sophisticated routing logic and cost optimization.</br>
동적 모델은 현재 <Tooltip tip="에이전트의 실행 환경으로, 에이전트 실행 전반에 걸쳐 유지되는 변경 불가능한 구성 및 컨텍스트 데이터(예: 사용자 ID, 세션 세부 정보 또는 애플리케이션별 구성)를 포함합니다.">런타임</Tooltip>에 현재 <Tooltip tip="메시지, 사용자 정의 필드, 처리 중에 추적 및 잠재적으로 수정해야 하는 모든 정보(예: 사용자 기본 설정 또는 도구 사용 통계)를 포함하여 에이전트 실행을 통해 흐르는 데이터입니다.">상태</Tooltip> 및 컨텍스트를 기반으로 선택됩니다. 이를 통해 정교한 라우팅 로직과 비용 최적화가 가능합니다.

To use a dynamic model, create middleware using the [`@wrap_model_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call) decorator that modifies the model in the request:</br>
동적 모델을 사용하려면 요청에서 모델을 수정하는 [`@wrap_model_call`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call) 데코레이터를 사용하여 미들웨어를 생성합니다.

```python  theme={null}
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    # 대화 복잡성에 따라 모델을 선택합니다.
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        # 긴 대화에는 고급 모델을 사용합니다.
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # Default model
    # 기본 모델
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

<Warning>
  Pre-bound models (models with [`bind_tools`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.bind_tools) already called) are not supported when using structured output. If you need dynamic model selection with structured output, ensure the models passed to the middleware are not pre-bound.
</Warning>
</br>
<Warning>
  사전 바인딩된 모델([`bind_tools`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel.bind_tools)가 이미 호출된 모델)은 구조화된 출력을 사용할 때 지원되지 않습니다. 구조화된 출력으로 동적 모델 선택이 필요한 경우, 미들웨어에 전달되는 모델이 사전 바인딩되지 않았는지 확인하십시오.
</Warning>
</br>
<Tip>
  For model configuration details, see [Models](/oss/python/langchain/models). For dynamic model selection patterns, see [Dynamic model in middleware](/oss/python/langchain/middleware#dynamic-model).
</Tip>
</br>
<Tip>
  모델 구성 세부 정보는 [모델](/oss/python/langchain/models)을 참조하십시오. 동적 모델 선택 패턴은 [미들웨어의 동적 모델](/oss/python/langchain/middleware#dynamic-model)을 참조하십시오.
</Tip>