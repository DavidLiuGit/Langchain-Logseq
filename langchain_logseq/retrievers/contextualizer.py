from logging import getLogger
from textwrap import dedent
from typing import Annotated, Any, Optional, Type

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


logger = getLogger(__name__)

RETRIEVER_CONTEXTUALIZER_PROMPT_TEMPLATE = dedent(
    """\
    {prompt}

    Latest user input: {user_input}

    {format_instructions}
    """
)
RETRIEVER_CONTEXTUALIZER_PROMPT_TEMPLATE_WITH_CHAT_HISTORY = dedent(
    """\
    {prompt}

    Chat History:
    {chat_history}

    Latest user input: {user_input}

    {format_instructions}
    """
)


class RetrieverContextualizerProps(BaseModel):
    """
    Contextualizers are a component within Langchain `Retriever`s, that transform a natural-language
    input (and history) into an actionable query, which can in turn be used to fetch relevant
    `Document`s to answer address the input. The actionable query output can be structured, or
    simply another string that can be used to query a Vectorstore.

    To do this, the core of the Contextualizer is an LLM. The `prompt` is used by the LLM to perform
    the transformation task.
    """

    llm: Annotated[
        BaseLanguageModel,
        Field(
            "The LLM that will be used to transform the input into an actionable query.",
        ),
    ]

    prompt: Annotated[
        str,
        Field(
            description="The prompt to use for the LLM to transform the input into an actionable query.",
            examples=[
                "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {user_input}\nStandalone question:"
            ],
            default="Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {user_input}\nStandalone question:",
        ),
    ]

    # TODO impl validation on this schema
    output_schema: Annotated[
        Optional[Type[BaseModel]],
        Field(
            description="(Optional) Structured output schema, as a Pydantic `BaseModel`. If provided, will be added to the end of the prompt.",
            default=None,
        ),
    ] = None

    enable_chat_history: Annotated[
        bool,
        Field(
            description="Whether to enable chat history in the prompt.",
            default=True,
        ),
    ] = True


class RetrieverContextualizer(Runnable):
    """
    A Runnable that transforms natural language input into an actionable query
    for retrievers, based on the provided configuration.
    """

    def __init__(self, props: RetrieverContextualizerProps):
        """Initialize with validated props."""
        self.props = props
        self.chain = self._generate_chain()
        self._parser_type = self.parser._type
        self._output_type = self.parser.OutputType if not self.props.output_schema else self.props.output_schema

    def _generate_chain(self) -> Runnable:
        """
        Generate and return the appropriate chain based on props.

        Returns:
            A Runnable chain that processes inputs according to the configuration.
        """
        # If output schema is provided, use PydanticOutputParser
        if self.props.output_schema:
            self.parser = PydanticOutputParser(pydantic_object=self.props.output_schema)
            # create a PromptTemplate with partials
            self.prompt_template = PromptTemplate(
                input_variables=["chat_history", "user_input"] if self.props.enable_chat_history else ["user_input"],
                partial_variables={
                    "prompt": self.props.prompt,
                    "format_instructions": self.parser.get_format_instructions(),
                },
                template=(
                    RETRIEVER_CONTEXTUALIZER_PROMPT_TEMPLATE_WITH_CHAT_HISTORY
                    if self.props.enable_chat_history
                    else RETRIEVER_CONTEXTUALIZER_PROMPT_TEMPLATE
                ),
            )

        else:
            # Otherwise, use the LLM and extract the string content
            # This ensures we get a clean string output rather than an LLM result object
            from langchain_core.output_parsers import StrOutputParser
            self.parser = StrOutputParser()
            self.prompt_template = PromptTemplate.from_template(self.props.prompt)

        return (
            self.prompt_template
            | (lambda x: logger.debug(f"Contextualizer prompt: {x}") or x)  # can enable for debugging, will not fail
            | self.props.llm
            | (lambda x: logger.debug(f"Contextualizer LLM output: {x}") or x)
            | self.parser
            | (lambda x: logger.info(f"OutputParser output: {x}") or x)
        )

    def invoke(self, input: dict[str, Any], config: Optional[dict[str, Any]] = None) -> Any:
        """
        Process the input through the chain.

        Args:
            input: The input to process, typically containing 'question' and 'chat_history'.
            config: Optional configuration for the chain.

        Returns:
            The processed output, either a string or a structured object based on the output_schema.
        """
        return self.chain.invoke(input, config=config)
