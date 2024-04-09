from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import os
import pathlib
from PIL import Image
import argparse

def process_caption(completed_prompt):
    caption = completed_prompt.split('ASSISTANT: ')[1].lower()
    caption = caption.replace('"', '')
    return caption

def main(data_dir, device):
    model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

    processor = LlavaNextProcessor.from_pretrained(model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    # Load images
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    files = [f for f in files if not f.endswith(".txt")]

    loader = torch.utils.data.DataLoader(files, batch_size=1)

    for i, batch in enumerate(loader):
        prompt = "USER: <image>\nGive a very concise caption for the image.\nASSISTANT:"
        inputs = processor([prompt for _  in batch], images=[Image.open(file) for file in batch], padding=True, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=100)
        completed_prompts = processor.batch_decode(output, skip_special_tokens=True)

        captions = [process_caption(completed_prompt) for completed_prompt in completed_prompts]

        for f, c, cp in zip(batch, captions, completed_prompts):
            print(f)
            print(c)
            print(cp)
            caption_path = pathlib.Path(f).with_suffix(".txt")
            with open(caption_path, "w") as f:
                f.write(c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)

    args = parser.parse_args()

    with torch.no_grad():
        main(args.input_dir, args.device)
