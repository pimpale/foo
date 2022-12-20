from transformers import GPTJForCausalLM, AutoTokenizer
import torch

model = GPTJForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B", revision="float16", low_cpu_mem_usage=True
)
model.cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.cuda()

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
