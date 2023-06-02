import pytest

from opsgpt import (
    RiskseerToolkit,
    OpsGPTAgent
)


@pytest.fixture(
    name="riskseer_toolkit",
    scope="session"
)
def __riskseer_toolkit(llm, authorized_requests):
    """riskseer风险感知CRUD工具库

    `resources/opsgpt_riskseer_selected.yaml`对应的NLA工具库
    """
    return RiskseerToolkit.from_llm(llm, authorized_requests)


@pytest.fixture(
    name="riskseer_agent"
)
def __riskseer_agent(
    llm,
    search_toolkit,
    riskseer_toolkit
):
    """具备实体、数据搜索功能的riskseer agent"""
    tools = riskseer_toolkit.get_tools() + search_toolkit.get_tools()
    return OpsGPTAgent.from_llm_and_tools(
        llm, tools
    )