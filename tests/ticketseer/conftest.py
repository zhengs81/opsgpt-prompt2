import pytest

from opsgpt import (
    TicketseerToolkit,
    OpsGPTAgent
)


@pytest.fixture(
    name="ticketseer_toolkit",
    scope="session"
)
def __ticketseer_toolkit(llm, authorized_requests):
    """ticketseer排障树CRUD工具库

    `resources/opsgpt_ticketseer_selected.yaml`对应的NLA工具库
    """
    return TicketseerToolkit.from_llm(llm, authorized_requests)


@pytest.fixture(
    name="ticketseer_agent"
)
def __ticketseer_agent(
    llm,
    search_toolkit,
    ticketseer_toolkit
):
    """具备实体、数据搜索功能的ticketseer agent"""
    tools = ticketseer_toolkit.get_tools() + search_toolkit.get_tools()
    return OpsGPTAgent.from_llm_and_tools(
        llm, tools
    )