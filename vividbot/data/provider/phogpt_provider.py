import sys
from typing import Union, List
sys.path.insert(0, r'/')
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .base import Provider

# turn off warning
import warnings
warnings.filterwarnings("ignore")
class PhoGPTProvider(Provider):
    def __init__(self, model_path="vinai/PhoGPT-4B-Chat", device="cuda"):
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
        self.config.init_device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, config=self.config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) 
        self.PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:" 
    def extract_texts(self, obj):
        '''
        Extract .text attribute from Translator object
        '''

        if isinstance(obj, list):
            return [self.extract_texts(item) for item in obj]
        else:
            try:
                return obj.text
            except AttributeError:
                return obj

    def _do_translate(self, input_data: Union[str, List[str]],
                      **kwargs) -> Union[str, List[str]]:
        
        input_prompt= self.PROMPT_TEMPLATE.format_map({"instruction": f"dịch {input_data} sang tiếng Việt"}) 

        input_ids = self.tokenizer(input_prompt, return_tensors="pt")  

        outputs = self.model.generate(  
            inputs=input_ids["input_ids"].to(self.config.init_device),  
            attention_mask=input_ids["attention_mask"].to(self.config.init_device),  
            do_sample=True,  
            temperature=1.0,  
            top_k=50,  
            top_p=0.9,  
            max_new_tokens=1024,  
            eos_token_id=self.tokenizer.eos_token_id,  
            pad_token_id=self.tokenizer.pad_token_id  
        )  

        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
        response = response.split("### Trả lời:")[1]
        return response

if __name__ == '__main__':
    test = PhoGPTProvider()
    #print(test.translate(["As long as I did, I really dont. Oh, I got a shotgun.", "How are you today ?"], src="en", dest="vi"))
    print(test.translate("As long as I did, I really dont. Oh, I got a shotgun."))
    print(test.translate("A man is talking through a field of wildflowers, talking to the cameraman, and advising to always get permission before harvesting on any land."))
