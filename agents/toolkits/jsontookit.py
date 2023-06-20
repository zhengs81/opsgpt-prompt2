from langchain.tools.json.tool import (
    JsonSpec,
    _parse_input
)
from langchain.tools.base import BaseTool
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from typing import Dict, List, Optional
from functools import reduce

CUSTOMIZED_JSON_PREFIX = """You are an agent designed to interact with JSON.
Your goal is to return a final answer by interacting with the JSON.
You have access to the following tools which help you learn more about the JSON you are interacting with.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
Do not make up any information that is not contained in the JSON.
Your input to the tools should be in the form of `data[key1][key2]` where `data` is the JSON dictionary you are interacting with, and the syntax used is Python. If `data` is an array, the keys should be integer.
You should only use keys that you know for a fact exist. You must validate that a key exists by seeing it previously when calling `json_spec_brief_view`. 
If you have not seen a key in one of those responses, you cannot use it.
You MUST start from `data` and MUST only add one key at a time to the path. You MUST NOT add multiple keys at once nor decrease the key in case of repeatation.

If the question does not seem to be related to the JSON after trials, just return "I don't know" as the answer.
Always begin your interaction with the `json_spec_brief_view` tool with input "data" to see the root information of the JSON.

If the final answer is from an array, then you MUST walk through ALL elements of the array to give final answer

Do not simply refer the user to the JSON or a section of the JSON, as this is not a valid answer.
You MUST NOT return `{{...}}` or `[...]` in the final answer, instead, you MUST keep digging until you find the answer and explicitly return it, but do not repeat the same action input.
"""

CUSTOMIZED_JSON_SUFFIX = """Begin!"

Question: {input}
Thought: I should take a brief view of data to see what I have access to
{agent_scratchpad}"""

class CorrectJsonSpec(JsonSpec):

    """json schema of corresponding dict_ field"""
    schema_: Dict
    
    @staticmethod
    def _get_fields_definition(json_object_schema: Dict, json_object: Dict, access_path: str):
        formatted_props = []

        props = json_object_schema["properties"]
        for name in props:
            if name not in json_object:
                continue

            prop = props[name]
            prop_name = name
            prop_type = prop["type"]
            prop_hint = f" # a folded {prop_type} can be accesss at `{access_path}[\"{name}\"]`" if prop_type in ["object", "array"] else ""

            desc = prop["description"]
            prop_desc = f"{desc}" if desc else "Unknown Definition Field" 
        
            formatted_props.append(
                f"  {prop_name}: {prop_desc},{prop_hint}"
            )
        
        fields_def = "\n".join(formatted_props) if formatted_props else "json schema and value not match at all"
        return fields_def
    
    def _truncate_redundant_dict(self, json_object: Dict, access_path: str, include_tips=True):
        truncate_dict = {}
        tips = []
        for k, v in json_object.items():
            if isinstance(v, list):
                truncate_dict[k] = "[...]"
                tips.append(f"* value of field {k} is a folded list, you can get a value of it at path `{access_path}[\"{k}\"][0]`")
                continue
            
            if isinstance(v, dict):
                truncate_dict[k] = "{...}"
                tips.append(f"* value of field {k} is a folded dictionary, you can get the value of it at path `{access_path}[\"{k}\"]`")
                continue
            
            truncate_dict[k] = v

        tips = "\n".join(tips)
        
        maximum_length = self.max_value_length - len(tips)
        
        str_rep = str(truncate_dict)

        if len(str_rep) > maximum_length:
            str_rep = str_rep[:maximum_length - 3] + "..."
            tips.append("* Value is a large dictionary, should explore its keys directly")

        if include_tips:
            str_rep = f"{str_rep}\nTips:\n{tips}"

        return str_rep

    def _truncate_redundant_list(self, json_array: List, access_path: str, include_tips=True):
        truncate_list = []
        tips = []
        if isinstance(json_array[0], dict):
            truncate_list = [self._truncate_redundant_dict(item, access_path, include_tips=False) for item in json_array]

        str_val = str(truncate_list)
        maximum_length = self.max_value_length
        
        possible_tip = "* Value is a list with large dictionary elements, should explore the keys of its element directly"
        if len(str_val) <= maximum_length - len(possible_tip):
            tips.append(possible_tip)
            maximum_length -= len(possible_tip)
        
        an_element = truncate_list[0]
        list_len = len(truncate_list)
        changeline_char_len = 1
        possible_tip = f"* Value is a list that is too long and has a length of {list_len}, should try explore its element direcly, with path as `{access_path}[i]`, where `i` is an integer in range [0, {list_len})"
        if len(str_val) > maximum_length - len(possible_tip) - changeline_char_len:
            str_val = f"[{an_element}...]"
            tips.append(possible_tip)
            maximum_length -= len(possible_tip) + changeline_char_len

        if len(str_val) > maximum_length:
            str_val = str_val[:maximum_length - 3] + "..."

        tips = "\n".join(tips)
        str_rep = f"{str_val}\nTips:\n{tips}"
        return str_rep

    def keys(self, text: str) -> str:
        try:
            items = _parse_input(text)
            schema = self.schema_
            json_val = self.dict_
            prefix_items = "data"
            for item in items:
                if item is None:
                    raise KeyError(
                        f"invalid path `{text}`"
                    )

                if "type" not in schema:
                    raise KeyError(
                        f"schema with path `{text}` come into invalid json schema, should stop exploration and ask user to fix the error direclty"
                    )

                schema_type = schema["type"]
                
                prefix_items += f"[\"{item}\"]" if isinstance(item, str) else f"[{item}]"
                
                if schema_type == "object":
                    if not isinstance(json_val, dict):
                        raise TypeError(f"schema type {schema_type} and real type {type(json_val)} is not match at `{prefix_items}`, should stop exploration and ask user to fix the error direclty")
                    if not isinstance(item, str):
                        raise KeyError(
                            f"invalid prefix path of `{prefix_items}`, which cannot access object type"
                        )
                    schema = schema["properties"][item]
                elif schema_type == "array":
                    if not isinstance(json_val, list):
                        raise TypeError(f"schema type {schema_type} and real type {type(json_val)} is not match at `{prefix_items}`, should stop exploration and ask user to fix the error direclty")
                    if not isinstance(item, int):
                        raise KeyError(
                            f"invalid prefix path of `{prefix_items}`, which cannot access array type"
                        )
                    schema = schema["items"]
                else:
                    raise KeyError(
                        f"invalid prefix path of `{prefix_items}`, which cannot subscript to primitive type"
                    )

                json_val = json_val[item]
                    
            if schema["type"] != "object" or not isinstance(json_val, dict):
                raise ValueError(
                    f"Value at path `{text}` is not a dict, get the value directly."
                )

            fields_def =  self._get_fields_definition(json_object_schema=schema, json_object=json_val, access_path=text)
            observation = f"the keys and their definitions:\n{fields_def}"        
            return observation
        except Exception as e:
            return repr(e)

    def value(self, text: str) -> str:
        """Return the value of the dict at the given path.

        Args:
            text: Python representation of the path to the dict (e.g. data["key1"][0]["key2"]).
        """
        try:
            items = _parse_input(text)
            val = self.dict_
            for i in items:
                val = val[i]

            str_val = str(val)
            
            if len(str_val) <= self.max_value_length:
                return str_val

            if isinstance(val, list):
                str_val = self._truncate_redundant_list(val, access_path=text)
                return str_val

            if isinstance(val, dict):
                str_val = self._truncate_redundant_dict(val, access_path=text)
                return str_val

            if len(str_val) > self.max_value_length:
                str_val = str_val[: self.max_value_length] + "..."

            return str_val
        except Exception as e:
            return repr(e)


class CustomizedJsonSpec(JsonSpec):

    """json schema of corresponding dict_ field"""
    schema_: Dict

    """maximum description text chapter length"""
    max_desc_len: int = 150
    
    """maximum display columns for json array in table view"""
    max_cols: int = 15
    
    """maximum display rows for json array in table view"""
    max_rows: int = 10

    
    def _get_object_brief_view(self, json_object_schema: Dict, json_object: Dict, access_path: str, indent: int = 0):
        
        single_indent = " " * 2
        indent_str = single_indent * indent

        if not json_object:
            return f"{indent_str}{str(json_object)} # empty object"

        formatted_props = []

        props = json_object_schema["properties"]
        for name in props:
            if name not in json_object:
                continue

            prop = props[name]
            prop_val = json_object[name]
            prop_name = name
            prop_type = prop["type"]

            prop_hint = ""
            if prop_val:
                prop_hint = f", a folded {prop_type} can be accesss at `{access_path}[\"{name}\"]`" if prop_type in ["object", "array"] else ""
                prop_val = "[...]" if prop_type == "array" else prop_val
                prop_val = "{...}" if prop_type == "object" else prop_val
            else:
                prop_hint = ", an empty value does not need to be digged into"

            prop_desc = prop["description"]
            prop_desc = f"{prop_desc}" if prop_desc else "Unknown Definition Field" 
            prop_desc = prop_desc[:self.max_desc_len] if len(prop_desc) > self.max_desc_len else prop_desc
        
            formatted_props.append(
                f"{indent_str + single_indent}{prop_name}: {prop_val}, # {prop_desc}{prop_hint}"
            )
        
        if len(formatted_props) == 0:
            raise TypeError("json schema and value not match at all")            

        fields_def = "\n".join(formatted_props)
        return fields_def
    

    def _get_list_brief_view(self, json_schema: Dict, json_array: List, access_path: str, indent: int = 0):
        
        if "items" not in json_schema or not json_schema["items"]:
            raise TypeError("json schema and value not match at all")            

        single_indent = " " * 2
        indent_str = single_indent * indent

        list_len = len(json_array)
        if list_len == 0:
            return f"{indent_str}[], empty array"

        tip = f"* Value is a list that has a length of {list_len}, you can dig into its element direcly, with path as `{access_path}[i]`, where i is a integer in range [0, {list_len})"
        element_schema = json_schema["items"]
        
        view_rep = ""
        
        if isinstance(json_array[0], dict):
            if element_schema["type"] != "object":
                raise TypeError("json schema and value not match at all")            
            
            element_example = json_array[0]
            # example_and_schema = self._get_object_brief_view(
            #     json_object_schema=element_schema,
            #     json_object=json_array[0],
            #     access_path=access_path,
            #     indent=indent + 2
            # )
            
            fields = reduce(lambda x, y: x & y, (set(v.keys()) for v in json_array[:10]))
            fields &= set(element_schema["properties"].keys())
            fields = list(fields)
            
            if not fields:
                raise TypeError(f"the element of `{access_path}` has invalida schema")
            
            column_padding = ", ..." if len(fields) > self.max_cols else ""
            row_padding = "\n..." if len(json_array) > self.max_rows else ""

            column_rep = ", ".join(fields) + column_padding
            
            table_values = []
            for row in json_array[:self.max_rows]:
                row_values = []
                for k in fields[:self.max_cols]:
                    rep = ""
                    v = row[k]
                    if isinstance(v, dict):
                        rep = "{...}"
                    elif isinstance(v, list):
                        rep = "[...]"
                    elif isinstance(v, str):
                        rep = f"\"{v}\""
                    else:
                        rep = str(v)

                    row_values.append(rep)
                table_values.append(row_values)

            rows_rep = f"\n{indent_str}".join(
                [
                    ", ".join([str(v) for v in row]) + column_padding
                    for row in table_values
                ]
            ) + row_padding
            table_view = f"{column_rep}\n{indent_str}{rows_rep}"

            view_segements = [
                table_view,
                # "* Above table is formated into csv here, key definition and value example for above table:",
                # example_and_schema
            ]

            view_rep = f"\n{indent_str}".join(view_segements)

        elif len(str(json_array)) > self.max_value_length:
            val_str = str(json_array)[:self.max_value_length - 5]
            view_rep = f"{val_str}...]"
        else:
            view_rep = str(json_array)

        str_rep = f"{indent_str}{view_rep}\n{indent_str}{tip}"
        return str_rep


    def brief_view(self, text: str) -> str:
        try:
            items = _parse_input(text)
            schema = self.schema_
            json_val = self.dict_
            prefix_items = "data"
            for item in items:
                if item is None:
                    raise KeyError(
                        f"invalid path `{text}`"
                    )

                if "type" not in schema:
                    raise KeyError(
                        f"schema with path `{text}` come into invalid json schema, should stop exploration and ask user to fix the error direclty"
                    )

                schema_type = schema["type"]
                
                prefix_items += f"[\"{item}\"]" if isinstance(item, str) else f"[{item}]"
                
                if schema_type == "object":
                    if not isinstance(json_val, dict):
                        raise TypeError(f"schema type {schema_type} and real type {type(json_val)} is not match at `{prefix_items}`, should stop exploration and ask user to fix the error direclty")
                    if not isinstance(item, str):
                        raise KeyError(
                            f"invalid prefix path of `{prefix_items}`, which cannot access object type"
                        )
                    schema = schema["properties"][item]
                elif schema_type == "array":
                    if not isinstance(json_val, list):
                        raise TypeError(f"schema type {schema_type} and real type {type(json_val)} is not match at `{prefix_items}`, should stop exploration and ask user to fix the error direclty")
                    if not isinstance(item, int):
                        raise KeyError(
                            f"invalid prefix path of `{prefix_items}`, which cannot access array type"
                        )
                    schema = schema["items"]
                else:
                    raise KeyError(
                        f"invalid prefix path of `{prefix_items}`, which cannot subscript to primitive type"
                    )

                json_val = json_val[item]
            
            schema_type = schema["type"]
            if schema_type == "object":
                brief_rep = self._get_object_brief_view(
                    json_object_schema=schema,
                    json_object=json_val,
                    access_path=text
                )
                return f"value at path `{text}` is an object, the brief view of it is as below, each line follow the format as ${{key}}: ${{value}}, # ${{description}} \n{brief_rep}"
            
            if schema_type == "array":
                brief_rep = self._get_list_brief_view(
                    json_schema=schema,
                    json_array=json_val,
                    access_path=text
                )
                return f"value at path `{text}` is an array, the brief view of it is as below,\n{brief_rep}"
            
            return f"value at path `{text}` is just a simple {schema_type} with value of `{json_val}`, you may not need to dig such deep and can step back to explore more general infomation"

        except Exception as e:
            return f"fetch `{prefix_items}` get error: {repr(e)}"
        

class JsonBriefViewTool(BaseTool):
    """Tool for take a brief view of JSON spec."""

    name = "json_spec_brief_view"
    description = """
    Can be used to take a brief view of the value at a given path. 
    Before calling this you should be SURE that the path to this exists.
    The input is a text representation of the path to the dict in Python syntax (e.g. data["key1"][0]["key2"]).
    """
    spec: CustomizedJsonSpec

    def _run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return self.spec.brief_view(tool_input)

    async def _arun(
        self,
        tool_input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return self._run(tool_input)


class CustomizedJsonToolkit(BaseToolkit):
    """Toolkit for interacting with a JSON spec."""

    spec: CustomizedJsonSpec

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            JsonBriefViewTool(spec=self.spec),
        ]
