from langchain.tools.json.tool import (
    JsonSpec,
    _parse_input
)
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit

from typing import Dict

CUSTOMIZED_JSON_PREFIX = """You are an agent designed to interact with JSON.
Your goal is to return a final answer by interacting with the JSON.
You have access to the following tools which help you learn more about the JSON you are interacting with.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
Do not make up any information that is not contained in the JSON.
Your input to the tools should be in the form of `data["key"][0]` where `data` is the JSON blob you are interacting with, and the syntax used is Python. 
You should only use keys that you know for a fact exist. You must validate that a key exists by seeing it previously when calling `json_spec_list_keys`. 
If you have not seen a key in one of those responses, you cannot use it.
You should only add one key at a time to the path. You cannot add multiple keys at once.
If you encounter a "KeyError", go back to the previous key, look at the available keys, and try again.

If the question does not seem to be related to the JSON, just return "I don't know" as the answer.
Always begin your interaction with the `json_spec_list_keys` tool with input "data" to see what keys exist in the JSON.

Note that sometimes the value at a given path is large. In this case, you will get an error "Value is a large dictionary, should explore its keys directly".
In this case, you should ALWAYS follow up by using the `json_spec_list_keys` tool to see what keys exist at that path.
Do not simply refer the user to the JSON or a section of the JSON, as this is not a valid answer. Keep digging until you find the answer and explicitly return it.
"""

class CorrectJsonSpec(JsonSpec):

    """json schema of corresponding dict_ field"""
    schema_: Dict
    
    @staticmethod
    def _get_fields_definition(json_object_schema):
        formatted_props = []

        props = json_object_schema["properties"]
        for name in props:
            prop = props[name]
            prop_name = name
            
            desc = prop["description"]
            prop_desc = f"{desc}" if desc else "Unknown Definition Field" 
        
            formatted_props.append(
                f"  {prop_name}: {prop_desc},"
            )
        
        fields_def = "\n".join(formatted_props)
        return fields_def

    def keys(self, text: str) -> str:
        try:
            items = _parse_input(text)
            val = self.schema_
            for item in items:
                if item is None or val["type"] not in ["object", "array"]:
                    raise KeyError(
                        f"invalid path `{text}`"
                    )
                
                if isinstance(item, str):
                    val = val["properties"][item]
                    continue
                
                if isinstance(item, int):
                    val = val["items"]
                    
            if val["type"] != "object":
                raise ValueError(
                    f"Value at path `{text}` is not a dict, get the value directly."
                )

            fields_def =  self._get_fields_definition(val)
            observation = f"the keys and their definitions:\n{fields_def}"        
            return observation
        except Exception as e:
            return repr(e)
