import pytest

from agents import get_nla_agent_executor
from agents.metacube import (
    SearchToolKit,
    MetaCubeToolKit
)


@pytest.fixture(
    name="search_toolkit"
)
def __search_toolkit():
    """metacube数据搜索工具库

    `resources/opsgpt_metacube_search_apis.yaml`对应的NLA工具库
    """
    return SearchToolKit()


@pytest.fixture(
    name="metacube_toolkit"
)
def __metacube_toolkit():
    """metacube知识图谱CRUD工具库

    `resources/opsgpt_metacube_kg_apis.yaml`对应的NLA工具库
    """
    return MetaCubeToolKit()


@pytest.fixture(
    name="search_agent_executor"
)
def __search_agent_executor(search_toolkit):
    """metacube数据搜索Agent"""
    tools = search_toolkit.get_tools()
    return get_nla_agent_executor(tools)


@pytest.fixture(
    name="metacube_agent_executor"
)
def __metacube_agent_executor(metacube_toolkit):
    """metacube知识图谱操作Agent"""
    tools = metacube_toolkit.get_tools()
    return get_nla_agent_executor(tools)


@pytest.fixture(
    name="all_agent_executor"
)
def __all_agent_executor(
    search_toolkit,
    metacube_toolkit
):
    """metacube知识综合Agent"""
    tools = metacube_toolkit.get_tools() + search_toolkit.get_tools()
    return get_nla_agent_executor(tools)