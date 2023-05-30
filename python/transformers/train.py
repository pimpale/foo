import os
import json
import random

# @title Download model and tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

model_name = "finetuned_model"
num_train_epochs = 1
test_frac = 0.2

# open corpus
corpus = open("corpus.txt", "r").read().split("===")
random.shuffle(corpus)

# construct datasets
datasets = DatasetDict(
    {
        "train": Dataset.from_dict({"text": corpus[: int(len(corpus) * (1 - test_frac))]}),
        "test": Dataset.from_dict({"text": corpus[int(len(corpus) * (1 - test_frac)) :]}),
    }
)


tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(
    tokenize_function, batched=True, num_proc=1, remove_columns=["text"]
)

# block_size = tokenizer.model_max_length
block_size = int(tokenizer.model_max_length / 4)

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=1,
)

# @title Set up the Trainer

trainer_state_path = f"{model_name}/trainer_state.json"
if os.path.isfile(trainer_state_path):
    f = open(trainer_state_path, "r")
    trainer_state = json.loads(f.read())
    f.close()
    epoch = trainer_state["epoch"]
    num_train_epochs += epoch

seed_data = random.randint(0, 2**32 - 1)

training_args = TrainingArguments(
    f"output/{model_name}",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=1.372e-4,
    weight_decay=0.01,
    num_train_epochs=num_train_epochs,
    save_total_limit=10,
    save_strategy="epoch",
    save_steps=1,
    report_to=None,
    seed=seed_data,
    logging_steps=5,
    do_eval=True,
    eval_steps=1,
    load_best_model_at_end=True
    # disable_tqdm=True
    # load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    # tokenizer=tokenizer,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
)

from transformers import get_cosine_schedule_with_warmup

train_dataloader = trainer.get_train_dataloader()
num_train_steps = len(train_dataloader)
trainer.create_optimizer_and_scheduler(num_train_steps)
trainer.lr_scheduler = get_cosine_schedule_with_warmup(
    trainer.optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)

trainer.model.config.task_specific_params["text-generation"] = {
    "do_sample": True,
    "min_length": 100,
    "max_length": 200,
    "temperature": 1.0,
    "top_p": 0.95,
    # 'prefix': '<|endoftext|>',
}

trainer.train()


model.save_pretrained(model_name)
tokenizer.save_pretrained(model_name)