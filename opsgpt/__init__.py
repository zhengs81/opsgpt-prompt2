from __future__ import annotations

from agents.toolkits import CustomizedNLAToolkit
from agents import CustomizedZeroShotAgent

from langchain.base_language import BaseLanguageModel
from langchain.tools import OpenAPISpec
from langchain.requests import Requests
from langchain.agents.agent import AgentExecutor
from langchain.callbacks.base import BaseCallbackManager
from langchain.tools.base import BaseTool

from pathlib import Path
from typing import Optional, Any, Sequence


ROOT_PATH = Path(__file__).parent.parent
RESOURECE_PATH = ROOT_PATH.joinpath("resources")
OPENAPI_FORMAT_INSTRUCTION = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: what to instruct the AI Action representative.
Observation: The Agent's response
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools. And I shoudld translate my answer to Chinese.
Final Answer: 具有适当细节的原始输入问题的最终答案

When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response. 

Important!!! when you think you know the final answer, please always translate your answer into Chinese"""


class BizseerToolkit(CustomizedNLAToolkit):
    """bizseer demo api工具库模板类"""
    
    @staticmethod
    def spec_path() -> Path | str:
        """the path to specific openapi spec file"""
        raise NotImplementedError("`spec_path` for `BizseerToolkit` is not implemented")

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        **kwargs: Any,
        ) -> BizseerToolkit:
        spec = OpenAPISpec.from_file(cls.spec_path())
        return cls.from_llm_and_spec(
            llm,
            spec=spec,
            requests=requests,
            verbose=verbose,
            **kwargs
        )


class MetacubeToolkit(BizseerToolkit):
    """metacube demo api工具库"""

    @staticmethod
    def spec_path() -> Path | str:
        return RESOURECE_PATH.joinpath("opsgpt_metacube_selected.yml")


class SearchToolkit(BizseerToolkit):
    """bizseer产品内部数据搜索 API工具库"""

    @staticmethod
    def spec_path() -> Path | str:
        return RESOURECE_PATH.joinpath("opsgpt_searcher.yml")


class RiskseerToolkit(BizseerToolkit):
    """riskseer demo api工具库"""

    @staticmethod
    def spec_path() -> Path | str:
        return RESOURECE_PATH.joinpath("opsgpt_riskseer_selected.yml")


class DataseerToolkit(BizseerToolkit):
    """dataseer demo api工具库"""

    @staticmethod
    def spec_path() -> Path | str:
        return RESOURECE_PATH.joinpath("opsgpt_dataseer_selected.yml")


class AlertseerToolkit(BizseerToolkit):
    """alertseer demo api工具库"""

    @staticmethod
    def spec_path() -> Path | str:
        return RESOURECE_PATH.joinpath("opsgpt_alertseer_selected.yml")


class TicketseerToolkit(BizseerToolkit):
    """ticketseer demo api工具库"""

    @staticmethod
    def spec_path() -> Path | str:
        return RESOURECE_PATH.joinpath("opsgpt_ticketseer_selected.yml")


class OpsGPTAgent(AgentExecutor):
    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ):
        zero_shot_agent = CustomizedZeroShotAgent.from_llm_and_tools(
            llm,
            tools,
            callback_manager=callback_manager,
            format_instructions=OPENAPI_FORMAT_INSTRUCTION
        )
        
        return cls.from_agent_and_tools(
            agent=zero_shot_agent,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
