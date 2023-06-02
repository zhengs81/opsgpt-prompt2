import pytest

from opsgpt import (
    MetacubeToolkit,
    OpsGPTAgent
)


@pytest.fixture(
    name="metacube_toolkit",
    scope="session"
)
def __metacube_toolkit(llm, authorized_requests):
    """metacube知识图谱CRUD工具库

    `resources/opsgpt_metacube_selected.yaml`对应的NLA工具库
    """
    return MetacubeToolkit.from_llm(llm, authorized_requests)


@pytest.fixture(
    name="metacube_agent",
    scope="session"
)
def __metacube_agent(
    llm,
    search_toolkit,
    metacube_toolkit
):
    """具备实体、数据搜索功能的metacube agent"""
    tools = metacube_toolkit.get_tools() + search_toolkit.get_tools()
    return OpsGPTAgent.from_llm_and_tools(
        llm, tools
    )