from pydantic import Field
from requests import Response
from typing import Any, Dict, Optional, cast

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
from langchain.callbacks.manager import CallbackManagerForChainRun, Callbacks

from agents.toolkits.jsontookit import (
    CustomizedJsonSpec,
    CustomizedJsonToolkit
)
from agents.json_agent import create_customized_json_agent


CUSTOMIZED_RESPONSE_TEMPLATE = """You are a helpful AI assistant trained to answer user queries from API responses.
You attempted to call an API, which resulted in:

API_RESPONSE: {response}

RESPONSE_SCHEMA: ```typescript
/* 请求成功的响应体 */
type SuccessResponse = {{
{schema}
}};
```

USER_COMMENT: {instructions}({api_description})

If the API_RESPONSE can answer the USER_COMMENT, please respond with the following markdown json block:
Response: ```json
{{"response": "Human-understandable synthesis of the API_RESPONSE, always trying to lookup definition in RESPONSE_SCHEMA. Since API_RESPONSE is a plain valid json object conforming to RESPONSE_SCHEMA"}}
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


JSON_AGENT_RESPONSE_TEMPLATE = """You have an api response which can help me to answer the below user query, which is symboled as USER_QUERY?
USER_QUERY: "{instructions}"

If you think you can answer it, please answer it with a Human-understandable synthesis of the information in your JSON.
Otherwise respond with what you did and a concise statement of the possible reason that why there is nothing related to the user query. If it can be easily fixed, provide a suggestion as well.
Please try to provide the information that user may want to know as much as possible, no matter the information about the value or the keys, instead of just provide some suggestion or general summarization.
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
                nested_props = self._format_response_properties(
                    prop.properties, indent + 2
                )
                prop_type = f"{{\n{nested_props}\n{' ' * indent}}}"
            
            if prop.items and prop.items.properties:
                nested_props = self._format_response_properties(
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
            return "/* Unknown */"
        
    def fetch_response_dict(self) -> dict:
        """Dict version of above function"""
        try:
            return list(self.response_body["200"].content.values())[0].media_type_schema.dict()
        except:
            # KeyError, properties not found, no schema
            return {}


class CustomizedAPIResponderChain(APIResponderChain):

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, 
        typescript_definition: str, 
        description: str,
        verbose: bool = True, 
        **kwargs: Any
    ) -> LLMChain:
        output_parser = APIResponderOutputParser()
        prompt = PromptTemplate(
            template=CUSTOMIZED_RESPONSE_TEMPLATE,
            output_parser=output_parser,
            partial_variables={
                "schema": typescript_definition,
                "api_description": description
            },
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
                description=operation.description,
                verbose=verbose,
                callbacks=callbacks,
            )
            setattr(fields_and_values, "api_response_chain", cus_responder_chain)
        
        return fields_and_values
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        intermediate_steps = {}
        instructions = inputs[self.instructions_key]
        instructions = instructions[: self.max_text_length]
        _api_arguments = self.api_request_chain.predict_and_parse(
            instructions=instructions, callbacks=_run_manager.get_child()
        )
        api_arguments = cast(str, _api_arguments)
        intermediate_steps["request_args"] = api_arguments
        _run_manager.on_text(
            api_arguments, color="green", end="\n", verbose=self.verbose
        )
        if api_arguments.startswith("ERROR"):
            return self._get_output(api_arguments, intermediate_steps)
        elif api_arguments.startswith("MESSAGE:"):
            return self._get_output(
                api_arguments[len("MESSAGE:") :], intermediate_steps
            )
        api_response = {}
        try:
            request_args = self.deserialize_json_input(api_arguments)
            method = getattr(self.requests, self.api_operation.method.value)
            api_response: Response = method(**request_args)
            if api_response.status_code != 200:
                method_str = str(self.api_operation.method.value)
                response_text = (
                    f"{api_response.status_code}: {api_response.reason}"
                    + f"\nFor {method_str.upper()}  {request_args['url']}\n"
                    + f"Called with args: {request_args['params']}"
                )
            else:
                response_text = api_response.text
        except Exception as e:
            response_text = f"Error with message {str(e)}"
        response_text = response_text[: self.max_text_length]
        intermediate_steps["response_text"] = response_text
        _run_manager.on_text(
            response_text, color="blue", end="\n", verbose=self.verbose
        )
        
        is_valid_response = isinstance(api_response, Response)
        status_is_200 = is_valid_response and api_response.status_code == 200
        without_other_error = not response_text.startswith("Error with message")

        if is_valid_response and status_is_200 and without_other_error:
            json_spec = CustomizedJsonSpec(
                dict_ = api_response.json() if isinstance(api_response, Response) else {},
                schema_ = self.api_operation.fetch_response_dict(),
                max_value_length=1000
            )
            json_toolkit = CustomizedJsonToolkit(spec=json_spec)
            json_agent_executor = create_customized_json_agent(
                llm=self.api_response_chain.llm,
                toolkit=json_toolkit,
                verbose=True,
                callback_manager=_run_manager.get_child()
            )
            
            formatted_template = JSON_AGENT_RESPONSE_TEMPLATE.format(
                instructions=instructions
            )
            formatted_template = formatted_template[: self.max_text_length]
            _answer = json_agent_executor.run(formatted_template)
        else:
            _answer = response_text

        return self._get_output(_answer, intermediate_steps)

        if self.api_response_chain is not None:
            _answer = self.api_response_chain.predict_and_parse(
                response=response_text,
                instructions=instructions,
                callbacks=_run_manager.get_child(),
            )
            answer = cast(str, _answer)
            _run_manager.on_text(answer, color="yellow", end="\n", verbose=self.verbose)
            return self._get_output(answer, intermediate_steps)
        else:
            return self._get_output(response_text, intermediate_steps)
