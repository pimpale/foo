from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from anthropic import Anthropic
from History import History
from openai import OpenAI
from together import Together


class ModelName(str, Enum):
    gpt_4o_mini = "gpt-4o-mini"
    gpt_4o = "gpt-4o-2024-08-06"
    claude3_haiku = "claude-3-haiku-20240307"
    claude3_sonnet = "claude-3-sonnet-20240229"
    claude3_opus = "claude-3-opus-20240229"
    claude3_5_sonnet = "claude-3-5-sonnet-20240620"
    llama3_1_405b = "accounts/fireworks/models/llama-v3p1-405b-instruct"
    llama3_1_70b = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    llama3_1_8b = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    llama2_7b = "meta-llama/Llama-2-7b-chat-hf"
    llama2_13b = "meta-llama/Llama-2-13b-chat-hf"
    llama2_70b = "meta-llama/Llama-2-70b-chat-hf"
    llama3_1_405b_together = (
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"  # Deprecated, small context window.
    )
    gemma2_27b = "google/gemma-2-27b-it"
    gemma2_13b = "google/gemma-2-13b-it"
    gemma_7b = "google/gemma-7b-it"
    qwen1_5_72b = "Qwen/Qwen1.5-72B-Chat"
    qwen1_5_32b = "Qwen/Qwen1.5-32B-Chat"
    qwen1_5_14b = "Qwen/Qwen1.5-14B-Chat"
    qwen1_5_7b = "Qwen/Qwen1.5-7B-Chat"
    qwen1_5_4b = "Qwen/Qwen1.5-4B-Chat"


@dataclass(frozen=True)
class ModelArguments:
    model: ModelName = ModelName.gpt_4o_mini  # The LLM model to use.
    temperature: int = 1  # The temperature for sampling.
    top_p: int = 1  # The top_p for sampling.
    max_tokens: int = 1024  # The maximum number of tokens to generate.
    openai_api_key: str | None = None  # The OpenAI API key.
    together_api_key: str | None = None  # The Together.ai API key.
    fireworks_api_key: str | None = None  # The Fireworks API key.
    anthropic_api_key: str | None = None  # The Anthropic API key.
    api_timeout: int = 60  # The timeout for API requests.
    api_max_retries: int = 1  # The maximum number of retries for API requests.


class APIModel(ABC):
    def __init__(self, args: ModelArguments):
        self.args = args
        self.model = args.model
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_tokens = args.max_tokens

    @abstractmethod
    def query(self, history: History) -> str:
        pass

    @abstractmethod
    def get_client(self, args: ModelArguments) -> OpenAI | Together | Anthropic:
        pass


class OpenAIModel(APIModel):
    def get_client(self, args: ModelArguments) -> OpenAI:
        return OpenAI(
            api_key=args.openai_api_key, max_retries=args.api_max_retries, timeout=args.api_timeout
        )

    def query(self, history: History) -> str:
        response = self.get_client(self.args).chat.completions.create(
            model=self.model,
            messages=list(history),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class TogetherAIModel(APIModel):
    def get_client(self, args: ModelArguments) -> Together:
        return Together(
            api_key=args.together_api_key,
            max_retries=args.api_max_retries,
            timeout=args.api_timeout,
        )

    def query(self, history: History) -> str:
        response = self.get_client(self.args).chat.completions.create(
            model=self.model,
            messages=list(history),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class FireworksAIModel(APIModel):
    def get_client(self, args: ModelArguments) -> OpenAI:
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=args.fireworks_api_key,
            timeout=args.api_timeout,
        )

    def query(self, history: History) -> str:
        response = self.get_client(self.args).chat.completions.create(
            model=self.model,
            messages=list(history),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content


class AnthropicModel(APIModel):
    def get_client(self, args: ModelArguments) -> Anthropic:
        return Anthropic(
            api_key=args.anthropic_api_key,
            max_retries=args.api_max_retries,
            timeout=args.api_timeout,
        )

    def query(self, history: History) -> str:
        response = self.get_client(self.args).messages.create(
            model=self.model,
            system=history[0]["content"],
            messages=list(history[1:]),
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        return response.content[0].text


def get_model(args: ModelArguments) -> APIModel:
    model_registry = {
        ModelName.gpt_4o_mini: OpenAIModel,
        ModelName.gpt_4o: OpenAIModel,
        ModelName.claude3_haiku: AnthropicModel,
        ModelName.claude3_sonnet: AnthropicModel,
        ModelName.claude3_opus: AnthropicModel,
        ModelName.claude3_5_sonnet: AnthropicModel,
        ModelName.llama3_1_405b_together: TogetherAIModel,
        ModelName.llama3_1_405b: FireworksAIModel,
        ModelName.llama3_1_70b: TogetherAIModel,
        ModelName.llama3_1_8b: TogetherAIModel,
        ModelName.llama2_7b: TogetherAIModel,
        ModelName.llama2_13b: TogetherAIModel,
        ModelName.llama2_70b: TogetherAIModel,
        ModelName.gemma2_27b: TogetherAIModel,
        ModelName.gemma2_13b: TogetherAIModel,
        ModelName.gemma_7b: TogetherAIModel,
        ModelName.qwen1_5_72b: TogetherAIModel,
        ModelName.qwen1_5_32b: TogetherAIModel,
        ModelName.qwen1_5_14b: TogetherAIModel,
        ModelName.qwen1_5_7b: TogetherAIModel,
        ModelName.qwen1_5_4b: TogetherAIModel,
    }
    model_class = model_registry[args.model]
    return model_class(args)
