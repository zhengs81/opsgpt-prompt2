from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import Callbacks
from langchain.chains.llm import LLMChain
from langchain.requests import Requests
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain.chains.api.openapi.response_chain import (
    APIResponderChain,
    PromptTemplate,
    APIResponderOutputParser
)
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain
from pydantic import Field

from typing import Any, Optional


CUSTOMIZED_RESPONSE_TEMPLATE = """You are a helpful AI assistant trained to answer user queries from API responses.
You attempted to call an API, which resulted in:
API_RESPONSE: {response}
RESPONSE_SCHEMA: {schema}
USER_COMMENT: "{instructions}"


If the API_RESPONSE and RESPONSE_SCHEMA can answer the USER_COMMENT respond with the following markdown json block:
Response: ```json
{{"response": "Human-understandable synthesis of the API_RESPONSE following RESPONSE_SCHEMA"}}
```

Otherwise respond with the following markdown json block:
Response Error: ```json
{{"response": "What you did and a concise statement of the resulting error. If it can be easily fixed, provide a suggestion."}}
```
If USER_COMMENT is to execute a task, and API_RESPONSE includes code : 200 or message: OK/success, then simply tell user the task is 
successfully executed.

You MUST respond as a markdown json code block. The person you are responding to CANNOT see the API_RESPONSE and RESPONSE_SCHEMA, so if there is any relevant information there you must include it in your response.

Begin:
---
"""


class CustomizedAPIOperation(APIOperation):
    response_body: Optional[dict] = Field(alias="response_body")
    """response_schema of API call"""
    
    def _format_response_properties(self, properties: dict, indent: int = 2) -> str:
        """Format nested response properties."""
        formatted_props = []

        for name, prop in properties.items():
            prop_name = name
            prop_type = prop.type
            prop_required = ""
            prop_desc = f"/* {prop.description} */" if prop.description else ""

            if prop.properties:
                nested_props = self.format_nested_properties(
                    prop.properties, indent + 2
                )
                prop_type = f"{{\n{nested_props}\n{' ' * indent}}}"
            
            if prop.items and prop.items.properties:
                nested_props = self.format_nested_properties(
                    prop.items.properties, indent + 2
                )
                prop_type = f"[\n{nested_props}\n{' ' * indent}]"
                

            formatted_props.append(
                f"{prop_desc}\n{' ' * indent}{prop_name}{prop_required}: {prop_type},"
            )

        return "\n".join(formatted_props)

    
    @classmethod
    def from_openapi_spec(cls, spec: OpenAPISpec, path: str, method: str) -> APIOperation:
        fields_and_values = super().from_openapi_spec(spec, path, method).dict()
        
        operation = spec.get_operation(path, method)
        operation.responses["path"] = path
        operation.responses["method"] = method
        fields_and_values["response_body"] = operation.responses
        
        return cls(**fields_and_values)
    
    def fetch_response_body(self) -> str:
        """Get response schema from API call"""
        try:
            response_props = list(self.response_body["200"].content.values())[0].media_type_schema.properties
            return self._format_response_properties(response_props)
        except:
            # KeyError, properties not found, no schema
            return "No Schema Found"



class CustomizedAPIResponderChain(APIResponderChain):

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, 
        typescript_definition: str, 
        verbose: bool = True, 
        **kwargs: Any
    ) -> LLMChain:
        output_parser = APIResponderOutputParser()
        prompt = PromptTemplate(
            template=CUSTOMIZED_RESPONSE_TEMPLATE,
            output_parser=output_parser,
            partial_variables={"schema": typescript_definition},
            input_variables=["response", "instructions"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose, **kwargs)


class CustomizedOpenAPIEndpointChain(OpenAPIEndpointChain):
    @classmethod
    def from_api_operation(
        cls,
        operation: CustomizedAPIOperation,
        llm: BaseLanguageModel,
        requests: Requests | None = None,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        raw_response: bool = False,
        callbacks: Callbacks = None,
        **kwargs: Any
    ) -> OpenAPIEndpointChain:
        
        dont_create_response = True
        fields_and_values = super().from_api_operation(
            operation,
            llm,
            requests,
            verbose,
            return_intermediate_steps,
            dont_create_response,
            callbacks,
            **kwargs
        )
        
        if not raw_response:
            cus_responder_chain = CustomizedAPIResponderChain.from_llm(
                llm,
                typescript_definition=operation.fetch_response_body(), 
                verbose=verbose,
                callbacks=callbacks,
            )
            setattr(fields_and_values, "api_response_chain", cus_responder_chain)
        
        return fields_and_values
        
