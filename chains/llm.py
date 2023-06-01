from langchain.llms.base import LLM

from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import (
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList
)
from transformers import AutoTokenizer, AutoModel

import torch

from typing import Optional, List, Any


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class StopWordStoppingCreteria(StoppingCriteria):
    def __init__(self, stops = []):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if input_ids[0][-1] in self.stops:
            return True
        return False


class Glm6B(LLM):
    temperature: float = 0.95,
    max_length: int = 2048,
    num_beams: int = 1,
    do_sample: bool = True,
    top_p: float = 0.75,
    tokenizer: Any
    model: Any
    
    def __init__(self,
        temperature: float = 0.95,
        max_length: int = 2048,
        num_beams: int = 1,
        do_sample: bool = True,
        top_p: float = 0.75,
    ):
        super().__init__()
        self.temperature = temperature
        self.max_length = max_length
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True).half().cuda()
        
    
    @torch.inference_mode()
    def _call(self,
        prompt: str,
        stop: Optional[List[str]] = None
    ):
        
        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        
        gen_kwargs = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "logits_processor": logits_processor
        }
        
        if stop is not None and len(stop) > 0:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    StopWordStoppingCreteria(
                        stops=[
                            self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze()[1:-2]
                            for stop_word in stop
                        ]
                    )
                ]
            )
        
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(outputs)
        response = self.model.process_response(response)
        
        return response
    
    def _llm_type(self):
        return "glm-6b"
            
