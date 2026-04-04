from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage


class FakeLLM(BaseLanguageModel):
    def invoke(self, input, config=None, **kwargs):
        return AIMessage(content="fake answer")

    def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        pass

    def predict(self, text, *, stop=None, **kwargs):
        return "fake answer"

    def predict_messages(self, messages, *, stop=None, **kwargs):
        return AIMessage(content="fake answer")

    async def agenerate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
        pass

    def apredict(self, text, *, stop=None, **kwargs):
        pass

    def apredict_messages(self, messages, *, stop=None, **kwargs):
        pass

    @property
    def _llm_type(self):
        return "fake"


class MockLLMStrategy:
    def create(self) -> BaseLanguageModel:
        return FakeLLM()


def test_llm_strategy_contract():
    """
    Test that a custom class implementing the LLMStrategy
    protocol successfully fulfills the required contract.
    """
    strategy = MockLLMStrategy()
    llm = strategy.create()

    assert isinstance(llm, BaseLanguageModel), "Should return a LangChain BaseLanguageModel instance"
