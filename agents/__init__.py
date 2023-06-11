from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.schema import (
    AgentAction,
    BaseMessage,
)

from typing import (List, Union, Tuple)


class CustomizedZeroShotAgent(ZeroShotAgent):
    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = []
        for action, observation in intermediate_steps:
            thoughts.append(action.log)
            thoughts.append(f"{self.observation_prefix}{observation}")
            thoughts.append(f"{self.llm_prefix}")

        if len(thoughts) > 1:
            thoughts.pop()
            thoughts.append("Always translate final answer to Chinese!!!\nThought:")
        
        return "\n".join(thoughts)
