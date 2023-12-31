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
import yaml
import jsonref
import json


ROOT_PATH = Path(__file__).parent.parent
RESOURECE_PATH = ROOT_PATH.joinpath("resources")
OPENAPI_FORMAT_INSTRUCTION = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: what to instruct the AI Action to do with the required data or information and context for tool to bettwer understanding as well.
Observation: The Agent's response
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer. User can't see any of my observations, API responses, links, or tools. And I shoudld translate my answer to Chinese.
Final Answer: 具有适当细节的原始输入问题的最终答案

The "Action Input" must include information about your intention instead of simple data either short instruction.
The "Action Input" must mentioned "Thought" and "Question" may want to know or to do with the following "Action".
When responding with your Final Answer, remember that the person you are responding to CANNOT see any of your Thought/Action/Action Input/Observations, so if there is any relevant information there you need to include it explicitly in your response. 

Important!!! when you think you know the final answer, please always translate your answer into Chinese.
If you think you don't know the final answer, but get at least one successful API response, then return some [key : value] pairs from the API response that you believe is useful.
"""


class BizseerToolkit(CustomizedNLAToolkit):
    """bizseer demo api工具库模板类"""
    
    @staticmethod
    def spec_path() -> Path | str:
        """the path to specific openapi spec file"""
        raise NotImplementedError("`spec_path` for `BizseerToolkit` is not implemented")

    def _load_spec(filename) -> dict:
        with open(filename, 'r') as f:
            raw_spec = yaml.safe_load(f)

        spec = jsonref.loads(json.dumps(raw_spec))
        return spec

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        **kwargs: Any,
        ) -> BizseerToolkit:
        spec = OpenAPISpec.from_spec_dict(cls._load_spec(cls.spec_path()))
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
