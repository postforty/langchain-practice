# What's new in v1
# v1의 새로운 기능

**LangChain v1 is a focused, production-ready foundation for building agents.**<br>
**LangChain v1은 에이전트 구축을 위한 집중적이고 프로덕션에 바로 사용할 수 있는 기반입니다.**

We've streamlined the framework around three core improvements:
<br>저희는 세 가지 핵심 개선 사항을 중심으로 프레임워크를 간소화했습니다.

<CardGroup cols={1}>
  <Card title="create_agent" icon="robot" href="#create-agent" arrow>
    The new standard for building agents in LangChain, replacing `langgraph.prebuilt.create_react_agent`.<br>
    LangChain에서 에이전트를 구축하기 위한 새로운 표준으로, `langgraph.prebuilt.create_react_agent`를 대체합니다.
  </Card>

  <Card title="Standard content blocks" icon="cube" href="#standard-content-blocks" arrow>
    A new `content_blocks` property that provides unified access to modern LLM features across providers.<br>
    공급자에 관계없이 최신 LLM 기능에 대한 통합된 액세스를 제공하는 새로운 `content_blocks` 속성입니다.
  </Card>

  <Card title="Simplified namespace" icon="sitemap" href="#simplified-package" arrow>
    The `langchain` namespace has been streamlined to focus on essential building blocks for agents, with legacy functionality moved to `langchain-classic`.<br>
    `langchain` 네임스페이스는 에이전트의 필수 구성 요소에 집중하도록 간소화되었으며, 레거시 기능은 `langchain-classic`으로 이전되었습니다.
  </Card>
</CardGroup>

To upgrade,<br>
업그레이드하려면,

<CodeGroup>
  ```bash pip theme={null}
  pip install -U langchain
  ```

  ```bash uv theme={null}
  uv add langchain
  ```
</CodeGroup>

For a complete list of changes, see the [migration guide](/oss/python/migrate/langchain-v1).<br>
전체 변경 사항 목록은 [마이그레이션 가이드](/oss/python/migrate/langchain-v1)를 참조하세요.

## `create_agent`

[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) is the standard way to build agents in LangChain 1.0. It provides a simpler interface than [`langgraph.prebuilt.create_react_agent`](https://reference.langchain.com/python/langgraph/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent) while offering greater customization potential by using [middleware](#middleware).<br>
[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)는 LangChain 1.0에서 에이전트를 구축하는 표준 방법입니다. [`langgraph.prebuilt.create_react_agent`](https://reference.langchain.com/python/langgraph/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)보다 간단한 인터페이스를 제공하면서도 [미들웨어](#middleware)를 사용하여 더 큰 사용자 정의 잠재력을 제공합니다.

```python  theme={null}
from langchain.agents import create_agent

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[search_web, analyze_data, send_email],
    system_prompt="You are a helpful research assistant."
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Research AI safety trends"}
    ]
})
```

Under the hood, [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) is built on the basic agent loop -- calling a model, letting it choose tools to execute, and then finishing when it calls no more tools:<br>
내부적으로 [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)는 기본 에이전트 루프를 기반으로 구축됩니다. 즉, 모델을 호출하고, 실행할 도구를 선택하게 한 다음, 더 이상 도구를 호출하지 않을 때 완료됩니다.

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src="https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=ac72e48317a9ced68fd1be64e89ec063" alt="Core agent loop diagram" className="rounded-lg" data-og-width="300" width="300" data-og-height="268" height="268" data-path="oss/images/core_agent_loop.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=280&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=a4c4b766b6678ef52a6ed556b1a0b032 280w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=560&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=111869e6e99a52c0eff60a1ef7ddc49c 560w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=840&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=6c1e21de7b53bd0a29683aca09c6f86e 840w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=1100&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=88bef556edba9869b759551c610c60f4 1100w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=1650&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=9b0bdd138e9548eeb5056dc0ed2d4a4b 1650w, https://mintcdn.com/langchain-5e9cc07a/Tazq8zGc0yYUYrDl/oss/images/core_agent_loop.png?w=2500&fit=max&auto=format&n=Tazq8zGc0yYUYrDl&q=85&s=41eb4f053ed5e6b0ba5bad2badf6d755 2500w" />
</div>

For more information, see [Agents](/oss/python/langchain/agents).<br>
자세한 내용은 [에이전트](/oss/python/langchain/agents)를 참조하세요.

### Middleware
### 미들웨어

Middleware is the defining feature of [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent). It offers a highly customizable entry-point, raising the ceiling for what you can build.<br>
미들웨어는 [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)의 결정적인 기능입니다. 매우 사용자 정의 가능한 진입점을 제공하여 구축할 수 있는 것의 한계를 높입니다.

Great agents require [context engineering](/oss/python/langchain/context-engineering): getting the right information to the model at the right time. Middleware helps you control dynamic prompts, conversation summarization, selective tool access, state management, and guardrails through a composable abstraction.<br>
훌륭한 에이전트는 [컨텍스트 엔지니어링](/oss/python/langchain/context-engineering)이 필요합니다. 즉, 적시에 적절한 정보를 모델에 전달하는 것입니다. 미들웨어는 구성 가능한 추상화를 통해 동적 프롬프트, 대화 요약, 선택적 도구 액세스, 상태 관리 및 가드레일을 제어하는 데 도움이 됩니다.

#### Prebuilt middleware
#### 사전 빌드된 미들웨어

LangChain provides a few [prebuilt middlewares](/oss/python/langchain/middleware#built-in-middleware) for common patterns, including:<br>
LangChain은 다음을 포함하여 일반적인 패턴을 위한 몇 가지 [사전 빌드된 미들웨어](/oss/python/langchain/middleware#built-in-middleware)를 제공합니다.

* [`PIIMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.PIIMiddleware): Redact sensitive information before sending to the model<br>모델로 보내기 전에 민감한 정보를 수정합니다.
* [`SummarizationMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.SummarizationMiddleware): Condense conversation history when it gets too long<br>대화 기록이 너무 길어지면 압축합니다.
* [`HumanInTheLoopMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.HumanInTheLoopMiddleware): Require approval for sensitive tool calls<br>민감한 도구 호출에 대한 승인이 필요합니다.

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.middleware import (
    PIIMiddleware,
    SummarizationMiddleware,
    HumanInTheLoopMiddleware
)


agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[read_email, send_email],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware(
            "phone_number",
            detector=(
                r"(?:\+?\d{1,3}[\s.-]?)?"
                r"(?:\(?\d{2,4}\)?[\s.-]?)?"
                r"\d{3,4}[\s.-]?\d{4}"
			),
			strategy="block"
        ),
        SummarizationMiddleware(
            model="claude-sonnet-4-5-20250929",
            max_tokens_before_summary=500
        ),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                }
            }
        ),
    ]
)
```

#### Custom middleware
#### 사용자 정의 미들웨어

You can also build custom middleware to fit your needs.<br>
필요에 맞게 사용자 정의 미들웨어를 빌드할 수도 있습니다.
Middleware exposes hooks at each step in an agent's execution:<br>
미들웨어는 에이전트 실행의 각 단계에서 후크를 노출합니다.

<div style={{ display: "flex", justifyContent: "center" }}>
  <img src="https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=eb4404b137edec6f6f0c8ccb8323eaf1" alt="Middleware flow diagram" className="rounded-lg" data-og-width="500" width="500" data-og-height="560" height="560" data-path="oss/images/middleware_final.png" data-optimize="true" data-opv="3" srcset="https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=280&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=483413aa87cf93323b0f47c0dd5528e8 280w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=560&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=41b7dd647447978ff776edafe5f42499 560w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=840&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=e9b14e264f68345de08ae76f032c52d4 840w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=1100&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=ec45e1932d1279b1beee4a4b016b473f 1100w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=1650&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=3bca5ebf8aa56632b8a9826f7f112e57 1650w, https://mintcdn.com/langchain-5e9cc07a/RAP6mjwE5G00xYsA/oss/images/middleware_final.png?w=2500&fit=max&auto=format&n=RAP6mjwE5G00xYsA&q=85&s=437f141d1266f08a95f030c2804691d9 2500w" />
</div>

Build custom middleware by implementing any of these hooks on a subclass of the [`AgentMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware) class:<br>
[`AgentMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.AgentMiddleware) 클래스의 서브클래스에서 이러한 후크 중 하나를 구현하여 사용자 정의 미들웨어를 빌드합니다.

| Hook              | When it runs             | Use cases                               |
| ----------------- | ------------------------ | --------------------------------------- |
| `before_agent`    | Before calling the agent | Load memory, validate input             |
| `before_model`    | Before each LLM call     | Update prompts, trim messages           |
| `wrap_model_call` | Around each LLM call     | Intercept and modify requests/responses |
| `wrap_tool_call`  | Around each tool call    | Intercept and modify tool execution     |
| `after_model`     | After each LLM response  | Validate output, apply guardrails       |
| `after_agent`     | After agent completes    | Save results, cleanup                   |

<br>

| 후크              | 실행 시점                | 사용 사례                               |
| ----------------- | ------------------------ | --------------------------------------- |
| `before_agent`    | 에이전트 호출 전         | 메모리 로드, 입력 유효성 검사           |
| `before_model`    | 각 LLM 호출 전           | 프롬프트 업데이트, 메시지 자르기        |
| `wrap_model_call` | 각 LLM 호출 주변         | 요청/응답 가로채기 및 수정              |
| `wrap_tool_call`  | 각 도구 호출 주변        | 도구 실행 가로채기 및 수정              |
| `after_model`     | 각 LLM 응답 후           | 출력 유효성 검사, 가드레일 적용         |
| `after_agent`     | 에이전트 완료 후         | 결과 저장, 정리                         |

Example custom middleware:<br>
사용자 정의 미들웨어 예시:

```python expandable theme={null}
from dataclasses import dataclass
from typing import Callable

from langchain_openai import ChatOpenAI

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest
)
from langchain.agents.middleware.types import ModelResponse

@dataclass
class Context:
    user_expertise: str = "beginner"

class ExpertiseBasedToolMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        user_level = request.runtime.context.user_expertise

        if user_level == "expert":
            # More powerful model
            model = ChatOpenAI(model="gpt-5")
            tools = [advanced_search, data_analysis]
        else:
            # Less powerful model
            model = ChatOpenAI(model="gpt-5-nano")
            tools = [simple_search, basic_calculator]

        request.model = model
        request.tools = tools
        return handler(request)

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[
        simple_search,
        advanced_search,
        basic_calculator,
        data_analysis
    ],
    middleware=[ExpertiseBasedToolMiddleware()],
    context_schema=Context
)
```

For more information, see [the complete middleware guide](/oss/python/langchain/middleware).<br>
자세한 내용은 [전체 미들웨어 가이드](/oss/python/langchain/middleware)를 참조하세요.

### Built on LangGraph
### LangGraph 기반으로 구축

Because [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) is built on [LangGraph](/oss/python/langgraph), you automatically get built in support for long running and reliable agents via:<br>
[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)는 [LangGraph](/oss/python/langgraph)를 기반으로 구축되었기 때문에 다음과 같은 장기 실행 및 신뢰할 수 있는 에이전트에 대한 지원이 자동으로 내장됩니다.

<CardGroup cols={2}>
  <Card title="Persistence" icon="database">
    Conversations automatically persist across sessions with built-in checkpointing<br>
    내장된 체크포인팅을 통해 세션 간에 대화가 자동으로 유지됩니다.
  </Card>

  <Card title="Streaming" icon="water">
    Stream tokens, tool calls, and reasoning traces in real-time<br>
    토큰, 도구 호출 및 추론 추적을 실시간으로 스트리밍합니다.
  </Card>

  <Card title="Human-in-the-loop" icon="hand">
    Pause agent execution for human approval before sensitive actions<br>
    민감한 조치를 취하기 전에 사람의 승인을 위해 에이전트 실행을 일시 중지합니다.
  </Card>

  <Card title="Time travel" icon="clock-rotate-left">
    Rewind conversations to any point and explore alternate paths and prompts<br>
    대화를 어느 시점으로든 되감고 대체 경로와 프롬프트를 탐색합니다.
  </Card>
</CardGroup>

You don't need to learn LangGraph to use these features—they work out of the box.<br>
이러한 기능을 사용하기 위해 LangGraph를 배울 필요가 없습니다. 즉시 작동합니다.

### Structured output
### 구조화된 출력

[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent) has improved structured output generation:<br>
[`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent)는 구조화된 출력 생성을 개선했습니다.

* **Main loop integration**: Structured output is now generated in the main loop instead of requiring an additional LLM call<br>
* **메인 루프 통합**: 이제 추가 LLM 호출 없이 메인 루프에서 구조화된 출력이 생성됩니다.
* **Structured output strategy**: Models can choose between calling tools or using provider-side structured output generation<br>
* **구조화된 출력 전략**: 모델은 도구를 호출하거나 공급자 측 구조화된 출력 생성을 사용하는 것 중에서 선택할 수 있습니다.
* **Cost reduction**: Eliminates extra expense from additional LLM calls<br>
* **비용 절감**: 추가 LLM 호출로 인한 추가 비용을 제거합니다.

```python  theme={null}
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel


class Weather(BaseModel):
    temperature: float
    condition: str

def weather_tool(city: str) -> str:
    """Get the weather for a city."""
    return f"it's sunny and 70 degrees in {city}"

agent = create_agent(
    "gpt-4o-mini",
    tools=[weather_tool],
    response_format=ToolStrategy(Weather)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in SF?"}]
})

print(repr(result["structured_response"]))
# results in `Weather(temperature=70.0, condition='sunny')`
```

**Error handling**: Control error handling via the `handle_errors` parameter to `ToolStrategy`:<br>
**오류 처리**: `ToolStrategy`의 `handle_errors` 매개변수를 통해 오류 처리를 제어합니다.

* **Parsing errors**: Model generates data that doesn't match desired structure<br>
* **파싱 오류**: 모델이 원하는 구조와 일치하지 않는 데이터를 생성합니다.
* **Multiple tool calls**: Model generates 2+ tool calls for structured output schemas<br>
* **여러 도구 호출**: 모델이 구조화된 출력 스키마에 대해 2개 이상의 도구 호출을 생성합니다.

***

## Standard content blocks
## 표준 콘텐츠 블록

<Note>
  Content block support is currently only available for the following integrations:<br>
  콘텐츠 블록 지원은 현재 다음 통합에서만 사용할 수 있습니다.

  * [`langchain-anthropic`](https://pypi.org/project/langchain-anthropic/)
  * [`langchain-aws`](https://pypi.org/project/langchain-aws/)
  * [`langchain-openai`](https://pypi.org/project/langchain-openai/)
  * [`langchain-google-genai`](https://pypi.org/project/langchain-google-genai/)
  * [`langchain-ollama`](https://pypi.org/project/langchain-ollama/)

  Broader support for content blocks will be rolled out gradually across more providers.<br>
  콘텐츠 블록에 대한 광범위한 지원은 더 많은 공급자에 걸쳐 점진적으로 출시될 예정입니다.
</Note>

The new [`content_blocks`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.messages.BaseMessage.content_blocks) property introduces a standard representation for message content that works across providers:<br>
새로운 [`content_blocks`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.messages.BaseMessage.content_blocks) 속성은 공급자 간에 작동하는 메시지 콘텐츠에 대한 표준 표현을 도입합니다.

```python  theme={null}
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
response = model.invoke("What's the capital of France?")

# Unified access to content blocks
for block in response.content_blocks:
    if block["type"] == "reasoning":
        print(f"Model reasoning: {block['reasoning']}")
    elif block["type"] == "text":
        print(f"Response: {block['text']}")
    elif block["type"] == "tool_call":
        print(f"Tool call: {block['name']}({block['args']})")
```

### Benefits
### 이점

* **Provider agnostic**: Access reasoning traces, citations, built-in tools (web search, code interpreters, etc.), and other features using the same API regardless of provider<br>
* **공급자 독립적**: 공급자에 관계없이 동일한 API를 사용하여 추론 추적, 인용, 내장 도구(웹 검색, 코드 인터프리터 등) 및 기타 기능에 액세스합니다.
* **Type safe**: Full type hints for all content block types<br>
* **타입 안전**: 모든 콘텐츠 블록 유형에 대한 전체 타입 힌트.
* **Backward compatible**: Standard content can be [loaded lazily](/oss/python/langchain/messages#standard-content-blocks), so there are no associated breaking changes<br>
* **하위 호환성**: 표준 콘텐츠는 [지연 로드](/oss/python/langchain/messages#standard-content-blocks)될 수 있으므로 관련된 주요 변경 사항이 없습니다.

For more information, see our guide on [content blocks](/oss/python/langchain/messages#standard-content-blocks).<br>
자세한 내용은 [콘텐츠 블록](/oss/python/langchain/messages#standard-content-blocks)에 대한 가이드를 참조하세요.

***

## Simplified package
## 단순화된 패키지

LangChain v1 streamlines the [`langchain`](https://pypi.org/project/langchain/) package namespace to focus on essential building blocks for agents.<br>
LangChain v1은 에이전트의 필수 구성 요소에 집중하기 위해 [`langchain`](https://pypi.org/project/langchain/) 패키지 네임스페이스를 간소화합니다.
The refined namespace exposes the most useful and relevant functionality:<br>
세련된 네임스페이스는 가장 유용하고 관련성 있는 기능을 노출합니다.

### Namespace
### 네임스페이스

| Module                                                                                | What's available                                                                                                                                                                                                                                                          | Notes                                 |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| [`langchain.agents`](https://reference.langchain.com/python/langchain/agents)         | [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent), [`AgentState`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.AgentState)                                                            | Core agent creation functionality     |
| [`langchain.messages`](https://reference.langchain.com/python/langchain/messages)     | Message types, [content blocks](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ContentBlock), [`trim_messages`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.trim_messages)                               | Re-exported from @\[`langchain-core`] |
| [`langchain.tools`](https://reference.langchain.com/python/langchain/tools)           | [`@tool`](https://reference.langchain.com/python/langchain/tools/#langchain.tools.tool), [`BaseTool`](https://reference.langchain.com/python/langchain/tools/#langchain.tools.BaseTool), injection helpers                                                                | Re-exported from @\[`langchain-core`] |
| [`langchain.chat_models`](https://reference.langchain.com/python/langchain/models)    | [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model), [`BaseChatModel`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel)   | Unified model initialization          |
| [`langchain.embeddings`](https://reference.langchain.com/python/langchain/embeddings) | [`Embeddings`](https://reference.langchain.com/python/langchain_core/embeddings/#langchain_core.embeddings.embeddings.Embeddings), [`init_embeddings`](https://reference.langchain.com/python/langchain_core/embeddings/#langchain_core.embeddings.embeddings.Embeddings) | Embedding models                      |

<br>

| 모듈                                                                                  | 사용 가능한 항목                                                                                                                                                                                                                                                        | 참고                                  |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| [`langchain.agents`](https://reference.langchain.com/python/langchain/agents)         | [`create_agent`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.create_agent), [`AgentState`](https://reference.langchain.com/python/langchain/agents/#langchain.agents.AgentState)                                                            | 핵심 에이전트 생성 기능               |
| [`langchain.messages`](https://reference.langchain.com/python/langchain/messages)     | 메시지 유형, [콘텐츠 블록](https://reference.langchain.com/python/langchain/messages/#langchain.messages.ContentBlock), [`trim_messages`](https://reference.langchain.com/python/langchain/messages/#langchain.messages.trim_messages)                               | @\[`langchain-core`]에서 다시 내보냄 |
| [`langchain.tools`](https://reference.langchain.com/python/langchain/tools)           | [`@tool`](https://reference.langchain.com/python/langchain/tools/#langchain.tools.tool), [`BaseTool`](https://reference.langchain.com/python/langchain/tools/#langchain.tools.BaseTool), 주입 헬퍼                                                                | @\[`langchain-core`]에서 다시 내보냄 |
| [`langchain.chat_models`](https://reference.langchain.com/python/langchain/models)    | [`init_chat_model`](https://reference.langchain.com/python/langchain/models/#langchain.chat_models.init_chat_model), [`BaseChatModel`](https://reference.langchain.com/python/langchain_core/language_models/#langchain_core.language_models.chat_models.BaseChatModel)   | 통합 모델 초기화                      |
| [`langchain.embeddings`](https://reference.langchain.com/python/langchain/embeddings) | [`Embeddings`](https://reference.langchain.com/python/langchain_core/embeddings/#langchain_core.embeddings.embeddings.Embeddings), [`init_embeddings`](https://reference.langchain.com/python/langchain_core/embeddings/#langchain_core.embeddings.embeddings.Embeddings) | 임베딩 모델                           |

Most of these are re-exported from `langchain-core` for convenience, which gives you a focused API surface for building agents.<br>
이들 대부분은 편의를 위해 `langchain-core`에서 다시 내보내므로 에이전트 구축을 위한 집중된 API 표면을 제공합니다.

```python  theme={null}
# Agent building
from langchain.agents import create_agent

# Messages and content
from langchain.messages import AIMessage, HumanMessage

# Tools
from langchain.tools import tool

# Model initialization
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
```

### `langchain-classic`

Legacy functionality has moved to [`langchain-classic`](https://pypi.org/project/langchain-classic) to keep the core packages lean and focused.<br>
레거시 기능은 핵심 패키지를 간결하고 집중적으로 유지하기 위해 [`langchain-classic`](https://pypi.org/project/langchain-classic)으로 이전되었습니다.

**What's in `langchain-classic`:**<br>
**`langchain-classic`의 내용:**

* Legacy chains and chain implementations<br>
* 레거시 체인 및 체인 구현
* Retrievers (e.g. `MultiQueryRetriever` or anything from the previous `langchain.retrievers` module)<br>
* 리트리버 (예: `MultiQueryRetriever` 또는 이전 `langchain.retrievers` 모듈의 모든 것)
* The indexing API<br>
* 인덱싱 API
* The hub module (for managing prompts programmatically)<br>
* 허브 모듈 (프로그래밍 방식으로 프롬프트 관리용)
* [`langchain-community`](https://pypi.org/project/langchain-community) exports<br>
* [`langchain-community`](https://pypi.org/project/langchain-community) 내보내기
* Other deprecated functionality<br>
* 기타 사용되지 않는 기능

If you use any of this functionality, install [`langchain-classic`](https://pypi.org/project/langchain-classic):<br>
이 기능 중 하나라도 사용하는 경우 [`langchain-classic`](https://pypi.org/project/langchain-classic)을 설치하세요.

<CodeGroup>
  ```bash pip theme={null}
  pip install langchain-classic
  ```

  ```bash uv theme={null}
  uv add langchain-classic
  ```
</CodeGroup>

Then update your imports:<br>
그런 다음 가져오기를 업데이트하세요.

```python  theme={null}
from langchain import ...  # [!code --]
from langchain_classic import ...  # [!code ++]

from langchain.chains import ...  # [!code --]
from langchain_classic.chains import ...  # [!code ++]

from langchain.retrievers import ...  # [!code --]
from langchain_classic.retrievers import ...  # [!code ++]

from langchain import hub  # [!code --]
from langchain_classic import hub  # [!code ++]
```

## Migration guide
## 마이그레이션 가이드

See our [migration guide](/oss/python/migrate/langchain-v1) for help updating your code to LangChain v1.<br>
코드를 LangChain v1으로 업데이트하는 데 도움이 필요하면 [마이그레이션 가이드](/oss/python/migrate/langchain-v1)를 참조하세요.

## Reporting issues
## 문제 보고

Please report any issues discovered with 1.0 on [GitHub](https://github.com/langchain-ai/langchain/issues) using the `'v1'` [label](https://github.com/langchain-ai/langchain/issues?q=state%3Aopen%20label%3Av1).<br>
1.0에서 발견된 모든 문제는 `'v1'` [레이블](https://github.com/langchain-ai/langchain/issues?q=state%3Aopen%20label%3Av1)을 사용하여 [GitHub](https://github.com/langchain-ai/langchain/issues)에 보고해 주세요.

## Additional resources
## 추가 리소스

<CardGroup cols={3}>
  <Card title="LangChain 1.0" icon="rocket" href="https://blog.langchain.com/langchain-langchain-1-0-alpha-releases/">
    Read the announcement<br>
    발표문 읽기
  </Card>

  <Card title="Middleware Guide" icon="puzzle-piece" href="https://blog.langchain.com/agent-middleware/">
    Deep dive into middleware<br>
    미들웨어 심층 분석
  </Card>

  <Card title="Agents Documentation" icon="book" href="/oss/python/langchain/agents" arrow>
    Full agent documentation<br>
    전체 에이전트 문서
  </Card>

  <Card title="Message Content" icon="message" href="/oss/python/langchain/messages#message-content" arrow>
    New content blocks API<br>
    새로운 콘텐츠 블록 API
  </Card>

  <Card title="Migration guide" icon="arrow-right-arrow-left" href="/oss/python/migrate/langchain-v1" arrow>
    How to migrate to LangChain v1<br>
    LangChain v1으로 마이그레이션하는 방법
  </Card>

  <Card title="GitHub" icon="github" href="https://github.com/langchain-ai/langchain">
    Report issues or contribute<br>
    문제 보고 또는 기여
  </Card>
</CardGroup>

## See also
## 참조

* [Versioning](/oss/python/versioning) - Understanding version numbers<br>
* [버전 관리](/oss/python/versioning) - 버전 번호 이해
* [Release policy](/oss/python/release-policy) - Detailed release policies<br>
* [릴리스 정책](/oss/python/release-policy) - 자세한 릴리스 정책

***

<Callout icon="pen-to-square" iconType="regular">
  [Edit the source of this page on GitHub.](https://github.com/langchain-ai/docs/edit/main/src/oss/python/releases/langchain-v1.mdx)<br>
  [GitHub에서 이 페이지의 소스 편집하기.](https://github.com/langchain-ai/docs/edit/main/src/oss/python/releases/langchain-v1.mdx)
</Callout>

<Tip icon="terminal" iconType="regular">
  [Connect these docs programmatically](/use-these-docs) to Claude, VSCode, and more via MCP for    real-time answers.<br>
  [실시간 답변을 위해 MCP를 통해 Claude, VSCode 등에 이 문서를 프로그래밍 방식으로 연결하세요.](/use-these-docs)
</Tip>