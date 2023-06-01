from chains import (
    CustomizedAPIOperation,
    CustomizedOpenAPIEndpointChain
)

from langchain.agents.agent_toolkits import NLAToolkit
from langchain.agents.agent_toolkits.nla.tool import NLATool
from langchain.base_language import BaseLanguageModel
from langchain.requests import Requests
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec

from typing import Any, Optional, List


class CustomizedNLATool(NLATool):
    @classmethod
    def from_llm_and_method(
        cls,
        llm: BaseLanguageModel,
        path: str,
        method: str,
        spec: OpenAPISpec,
        requests: Requests | None = None,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        **kwargs: Any
    ) -> NLATool:
        api_operation = CustomizedAPIOperation.from_openapi_spec(spec, path, method)
        chain = CustomizedOpenAPIEndpointChain.from_api_operation(
            api_operation,
            llm,
            requests=requests,
            verbose=verbose,
            return_intermediate_steps=return_intermediate_steps,
            **kwargs,
        )
        return cls.from_open_api_endpoint_chain(chain, spec.info.title)


class CustomizedNLAToolkit(NLAToolkit):
    @staticmethod
    def _get_http_operation_tools(
        llm: BaseLanguageModel,
        spec: OpenAPISpec,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> List[CustomizedNLATool]:
        """Get the tools for all the API operations."""
        if not spec.paths:
            return []
        http_operation_tools = []
        for path in spec.paths:
            for method in spec.get_methods_for_path(path):
                endpoint_tool = CustomizedNLATool.from_llm_and_method(
                    llm=llm,
                    path=path,
                    method=method,
                    spec=spec,
                    requests=requests,
                    verbose=verbose,
                    **kwargs,
                )
                http_operation_tools.append(endpoint_tool)
        return http_operation_tools
