from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage

from zerorag.core.llms import generate_answer, get_llm


@patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
@patch("langchain_openai.ChatOpenAI")
class TestGetLLMEngine:
    """Test suite for the main registry-based LLM engine."""

    def test_returns_llm_with_default_strategy(self, mock_cls):
        """Test that the engine returns a BaseLanguageModel instance using the default strategy."""
        mock_cls.return_value = MagicMock(spec=BaseLanguageModel)
        llm = get_llm()

        assert isinstance(llm, BaseLanguageModel)
        mock_cls.assert_called_once_with(model="gpt-4.1-mini")

    def test_unknown_strategy_falls_back_to_openai_mini(self, mock_cls):
        """Test that an unknown strategy name falls back to openai-mini."""
        mock_cls.return_value = MagicMock(spec=BaseLanguageModel)
        llm = get_llm(strategy="nonexistent")

        assert isinstance(llm, BaseLanguageModel)
        mock_cls.assert_called_once_with(model="gpt-4.1-mini")

    def test_explicit_openai_mini_strategy(self, mock_cls):
        """Test that explicitly requesting openai-mini works."""
        mock_cls.return_value = MagicMock(spec=BaseLanguageModel)
        llm = get_llm(strategy="openai-mini")

        assert isinstance(llm, BaseLanguageModel)
        mock_cls.assert_called_once_with(model="gpt-4.1-mini")

    def test_explicit_openai_strategy(self, mock_cls):
        """Test that explicitly requesting openai works."""
        mock_cls.return_value = MagicMock(spec=BaseLanguageModel)
        llm = get_llm(strategy="openai")

        assert isinstance(llm, BaseLanguageModel)
        mock_cls.assert_called_once_with(model="gpt-4.1")


class TestGenerateAnswer:
    """Test suite for the generate_answer function."""

    def test_returns_answer_string(self):
        """Test that generate_answer returns the LLM's content as a string."""
        mock_llm = MagicMock()
        mock_llm.__or__ = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = AIMessage(content="The answer is 42.")
        mock_llm.__or__.return_value = mock_chain

        documents = [Document(page_content="Some context", metadata={"source": "doc.txt"})]

        with patch("zerorag.core.llms.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            answer = generate_answer("What is the answer?", documents, mock_llm)

        assert answer == "The answer is 42."

    def test_joins_multiple_documents_as_context(self):
        """Test that multiple documents are joined with double newlines."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = AIMessage(content="combined answer")

        documents = [
            Document(page_content="First chunk", metadata={"source": "a.txt"}),
            Document(page_content="Second chunk", metadata={"source": "b.txt"}),
        ]

        with patch("zerorag.core.llms.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            generate_answer("question", documents, MagicMock())

        mock_chain.invoke.assert_called_once_with(
            {
                "question": "question",
                "context": "First chunk\n\nSecond chunk",
            }
        )
