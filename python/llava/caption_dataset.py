from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import os
from PIL import Image
import argparse
import json


def main(data_dir, output_file):
    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    loader = torch.utils.data.DataLoader(files, batch_size=32)

    d = {}
    for i, batch in enumerate(loader):
        prompt = "USER: <image>\nGive a very concise caption for the image.\nASSISTANT:"
        inputs = processor([prompt for _  in batch], images=[Image.open(file) for file in batch], padding=True, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=200)
        completed_prompts = processor.batch_decode(output, skip_special_tokens=True)

        captions = ['sks a photo of ' + completed_prompt.split('ASSISTANT: ')[1].lower() for completed_prompt in completed_prompts]
        
        for f, c in zip(batch, captions):
            print(f)
            print(c)
            d[f] = c

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=4)            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_file)