from pathlib import Path

from transformers.utils import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

#DEVICE = torch.device("cuda:0")
DEVICE = torch.device("cpu")


def load_model():
    #cache_dir = './model_data'
    cache_dir = './model_data'
    Path(cache_dir).mkdir(parents=True, exist_ok=True)


    # Загружаем модель ruGPT от сбера
    model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
    print('loading')
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, local_files_only=True)

    model_pretrained_path = 'model_data_finetuned' # model_name_or_path 'model_data_finetuned'
    model = GPT2LMHeadModel.from_pretrained(model_pretrained_path).to(DEVICE)
    return tokenizer, model

def train(model):
    print('training')
    train_file = 'train.txt'
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=64,
                                overwrite_cache=True)

    # специальный класс который будет подавать в модель данные в нужном ей виде
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./finetuned",
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        optimizers=(torch.optim.AdamW(model.parameters(), lr=1e-5), None)  # Optimizer and lr scheduler
    )
    model.train()
    trainer.train()

    savedir = 'model_data_finetuned'
    Path(savedir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(savedir)


def execute(tokenizer, model):
    model.eval()
    print('executing')

    text = 'Любить незнакомые города легко – мы принимаем их такими, какие они есть, и не требуем ничего, кроме новых впечатлений.'
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)

    out = model.generate(input_ids, do_sample=True,
                         top_k=10,
                         max_new_tokens=100,
                         repetition_penalty=2.5,
                         temperature=5.0,
                         num_beams=20,
                        )

    generated_text = list(map(tokenizer.decode, out))[0]
    generated_text = generated_text.replace('<s>', ' ')
    print()
    print(generated_text)


if __name__ == '__main__':
    tokenizer, model = load_model()
    #train(model)
    execute(tokenizer, model)