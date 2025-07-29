import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGenerationChunk, ChatGeneration
from typing import List, Iterator, Any, Optional, Dict
from dashscope import Generation
from langchain_core.messages import AIMessageChunk


class ChatDashScope(BaseChatModel):
    model: str = "qwen-max"
    temperature: float = 0.7
    top_p: float = 0.8
    api_key: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "dashscope-chat"

    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        """非流式生成"""
        ds_messages = self._convert_messages(messages)

        response = Generation.call(
            model=self.model,
            messages=ds_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY")
        )

        if response.status_code == 200:
            content = self._extract_content(response)
            generation = ChatGeneration(message=AIMessage(content=content))
            return ChatResult(generations=[generation])
        else:
            raise Exception(f"DashScope API error: {response}")

    def _stream(self, messages: List[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """流式生成，只输出增量文本"""
        ds_messages = self._convert_messages(messages)

        response = Generation.call(
            model=self.model,
            messages=ds_messages,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True,
            api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY")
        )

        last_len = 0

        for chunk in response:
            if chunk.status_code != 200:
                raise Exception(f"Stream error: {chunk}")

            current_text = chunk.output.text
            if current_text is None:
                continue

            if last_len < len(current_text):
                delta = current_text[last_len:]
                last_len = len(current_text)

                yield ChatGenerationChunk(message=AIMessageChunk(content=delta))

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        """转换 LangChain 消息为 DashScope 格式"""
        ds_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                role = "user"
            ds_messages.append({"role": role, "content": msg.content})
        return ds_messages

    def _extract_content(self, response) -> str:
        """从响应中提取文本内容，兼容 text 和 choices 两种格式"""
        output = getattr(response, 'output', None)
        if not output:
            raise Exception("Empty output")

        if hasattr(output, 'text') and output.text:
            return output.text
        elif hasattr(output, 'choices') and output.choices:
            return output.choices[0].message.content
        else:
            raise Exception("No valid content in response")

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p
        }