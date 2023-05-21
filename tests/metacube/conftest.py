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
    return SearchToolKit()


@pytest.fixture(
    name="metacube_toolkit"
)
def __metacube_toolkit():
    return MetaCubeToolKit()


@pytest.fixture(
    name="search_agent_executor"
)
def __search_agent_executor(search_toolkit):
    tools = search_toolkit.get_tools()
    return get_nla_agent_executor(tools)


@pytest.fixture(
    name="metacube_agent_executor"
)
def __metacube_agent_executor(metacube_toolkit):
    tools = metacube_toolkit.get_tools()
    return get_nla_agent_executor(tools)


@pytest.fixture(
    name="all_agent_executor"
)
def __all_agent_executor(
    search_toolkit,
    metacube_toolkit
):
    tools = metacube_toolkit.get_tools() + search_toolkit.get_tools()
    return get_nla_agent_executor(tools)