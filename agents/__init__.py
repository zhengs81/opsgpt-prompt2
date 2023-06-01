from langchain.agents.mrkl.base import (
    ZeroShotAgent,
    PromptTemplate,
    Sequence,
    BaseTool,
    PREFIX,
    SUFFIX,
    FORMAT_INSTRUCTIONS
)

from typing import (Optional, List)


class CustomizedZeroShotAgent(ZeroShotAgent):
    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        prefix: str = PREFIX,
        suffix: str = SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ) -> PromptTemplate:
        """Create prompt in the style of the zero shot agent.

        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        suffix += "The final answer MUST be translated to Chinese."
        return super().create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            format_instructions=format_instructions,
            input_variables=input_variables
        )
