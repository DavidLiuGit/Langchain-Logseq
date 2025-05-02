from typing import Annotated, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


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
                "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:",
                "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:",
            ],
        ),
    ]

    output_schema: Annotated[
        Optional[BaseModel],
        Field(
            description="(Optional) Structured output schema, as a Pydantic `BaseModel`. If provided, will be added to the end of the prompt.",
        )
    ] = None



class RetrieverContextualizer(Runnable):
    """
    A Runnable that transforms natural language input into an actionable query
    for retrievers, based on the provided configuration.
    """
    
    def __init__(self, props: RetrieverContextualizerProps):
        """Initialize with validated props."""
        self.props = props
        self.chain = self._generate_chain()


    def _generate_chain(self) -> Runnable:
        """
        Generate and return the appropriate chain based on props.
        
        Returns:
            A Runnable chain that processes inputs according to the configuration.
        """
        prompt_template = PromptTemplate.from_template(self.props.prompt)
        
        if self.props.output_schema:
            # If output schema is provided, use PydanticOutputParser
            parser = PydanticOutputParser(pydantic_object=self.props.output_schema)
            format_instructions = parser.get_format_instructions()
            # Append format instructions to the prompt
            prompt_with_format = self.props.prompt + "\n\n" + format_instructions
            prompt_template = PromptTemplate.from_template(prompt_with_format)
            return prompt_template | self.props.llm | parser
        
        else:
            # Otherwise, use the LLM and extract the string content
            # This ensures we get a clean string output rather than an LLM result object
            from langchain_core.output_parsers import StrOutputParser
            return prompt_template | self.props.llm | StrOutputParser()
