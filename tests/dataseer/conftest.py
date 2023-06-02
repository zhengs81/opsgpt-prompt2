import pytest

from opsgpt import (
    DataseerToolkit,
    OpsGPTAgent
)


@pytest.fixture(
    name="dataseer_toolkit",
    scope="session"
)
def __dataseer_toolkit(llm, authorized_requests):
    """dataseer数据中心CRUD工具库

    `resources/opsgpt_dataseer_selected.yaml`对应的NLA工具库
    """
    return DataseerToolkit.from_llm(llm, authorized_requests)


@pytest.fixture(
    name="dataseer_agent"
)
def __dataseer_agent(
    llm,
    search_toolkit,
    dataseer_toolkit
):
    """具备实体、数据搜索功能的dataseer agent"""
    tools = dataseer_toolkit.get_tools() + search_toolkit.get_tools()
    return OpsGPTAgent.from_llm_and_tools(
        llm, tools
    )