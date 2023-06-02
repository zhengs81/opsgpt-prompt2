from opsgpt import (
    ROOT_PATH,
    SearchToolkit
)

from langchain.llms import OpenAI
from langchain.requests import Requests
import pytest

import dotenv
import os

@pytest.fixture(
    name="llm",
    scope="session"
)
def __llm():
    """langchain基础语言模型"""
    dotenv.load_dotenv(ROOT_PATH.joinpath(".env"))

    return OpenAI(model_name="text-davinci-003")


@pytest.fixture(
    scope="session"
)
def __auth_token():
    """从环境变量中获取登录鉴权token"""
    
    return os.environ["BIZSEER_TOKEN"]


@pytest.fixture(
    name="authorized_requests",
    scope="session"
)
def __requests(__auth_token):
    """经过鉴权的langchain请求实例
    
    header增加了相关鉴权
    """
    
    return Requests(
        headers={
            "Content-Type": "application/json",
            "Authorization": __auth_token
        }
    )


@pytest.fixture(
    name="search_toolkit",
    scope="session"
)
def __search_toolkit(llm, authorized_requests):
    """产品数据搜索工具库

    `resources/opsgpt_searcher.yml`对应的NLA工具库
    TODO: 增加metacube以外其他产品可能需要用到的模糊搜索、匹配接口?
    """
    return SearchToolkit.from_llm(llm, authorized_requests)