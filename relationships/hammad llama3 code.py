import os
import torch
import transformers
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login

cache_dir = '/dvmm-filer3a/users/nrdyava/hf_home'
os.environ['HF_HOME'] = cache_dir
hf_token = !cat $HF_HOME/huggingface/token

login(token=hf_token[0])

model_id = 'llama3-8b-inst'

if model_id == 'llama3-8b-inst':
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
elif model_id == 'llama3-70b-inst':
    model_name = 'meta-llama/Meta-Llama-3-70B-Instruct'
else:
    raise Exception("Unknown model")

# prepare model
sampling_params = SamplingParams(top_p=0.9, temperature=0.7, max_tokens=1024)
llm = LLM(
    model=model_name, 
    tensor_parallel_size=1, 
    download_dir=cache_dir
)
tokenizer = llm.get_tokenizer()

# prepare data

def format_instruction(question, document):
    instruction = "Answer the following question from the given document. Your answer should contain only sentences from the document that best answer the given question. Do not output anything else apart from document sentences."
    user_message = f"""{instruction}
    
    Question: {question}
    Document: {document}
    """
    
    messages = [
        {"role": "user", "content": user_message},
    ]
    instruction = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(instruction)
    prompt_token_ids = tokenizer.encode(instruction, return_tensors="pt")
    return prompt_token_ids.tolist()

question = 'After how many hour does the link expire?'
document = 'Remember that the links expire after 24 hours and a certain amount of downloads. You can always re-request a link if you start seeing errors such as 403: Forbidden'

prompt_ids = format_instruction(question, document)
ans = llm.generate(
        prompt_token_ids=prompt_ids, 
        sampling_params=sampling_params
    )
answer = ans[0].outputs[0].text