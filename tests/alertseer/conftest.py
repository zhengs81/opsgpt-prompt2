import pytest

from opsgpt import (
    AlertseerToolkit,
    OpsGPTAgent
)


@pytest.fixture(
    name="alertseer_toolkit",
    scope="session"
)
def __alertseer_toolkit(llm, authorized_requests):
    """alertseer统一告警CRUD工具库

    `resources/opsgpt_alertseer_selected.yaml`对应的NLA工具库
    """
    return AlertseerToolkit.from_llm(llm, authorized_requests)


@pytest.fixture(
    name="alertseer_agent",
    scope="session"
)
def __alertseer_agent(
    llm,
    search_toolkit,
    alertseer_toolkit
):
    """具备实体、数据搜索功能的alertseer agent"""
    tools = alertseer_toolkit.get_tools() + search_toolkit.get_tools()
    return OpsGPTAgent.from_llm_and_tools(
        llm, tools
    )