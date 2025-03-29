import re
import os
import copy
import json
import random
import pathlib
import traceback
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from torch.utils.data import Dataset

import transformers
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import sys
sys.path.append('./')
from utils import tokenizer_token
from trainer import (VideoLLaMA2Trainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486   
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None


import os
import subprocess
import shutil
import torch
from typing import List
import uuid
import pandas as pd

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ModelArguments:
    # LLM Arguments
    model_type: Optional[str] = field(default="SMES_llama", metadata={"help": "Model type selected in the list: "})
    model_path: Optional[str] = field(default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV", metadata={"help": "This is the videollama2 model path"})
    version: Optional[str] = field(default="v1", metadata={"help": "Version of the conversation template."})
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the LLM backbone."})

@dataclass
class DataArguments:
    # Path Arguments
    data_path: str = field(default=None, metadata={"help": "Path to the training data, file .jsonl , raw samples."})
    # Loading Arguments
    lazy_preprocess: bool = False
 

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"




def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    modal_token: str = None,
) -> Dict:
    # Apply prompt templates
    conversations = []
    input_ids = []
    targets = []
    # pprint(sources)
    for i, source in enumerate(sources):
        source = source[1:]# not add sys prompt in this step
        if source[0]["role"] != "user":
            # Skip the first one if it is not from human
            source = source[1:]

        message = source
        conversation = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        input_ids.append(tokenizer_token(conversation, tokenizer, return_tensors='pt'))
        targets.append(copy.deepcopy(input_ids[-1]))
        assert len(source) % 2 == 0, f"Invalid conversation length {len(source)}."

        cur = 0
        message = []
        for idx, sentence in enumerate(source):
            if idx % 2 == 1:
                tmp_message = [
                    source[idx-1],
                    sentence
                ]

                instruction = tokenizer.apply_chat_template(message + tmp_message[:1], tokenize=False, add_generation_prompt=True)
                conversation = tokenizer.apply_chat_template(message + tmp_message, tokenize=False, add_generation_prompt=False)

                instruction_len = len(tokenizer_token(instruction, tokenizer, return_tensors='pt'))
                conversation_len = len(tokenizer_token(conversation, tokenizer, return_tensors='pt'))

                targets[-1][cur:instruction_len] = IGNORE_INDEX

                cur = conversation_len
                message += tmp_message
    return dict(input_ids=input_ids, labels=targets)


def get_conversation(components) -> List[Dict[str,str]] :

    label = (f"Answer: {components['answer']}")


    total_prompt = (
        "[CONTEXT] \n"
        "[CONTEXT] \n"
        "question: {problem_type} \n"
        "Situation: {situation} \n"
        "History chat informations above \n"
        "[CURRENT CONTEXT] \n"
        "Client's video: <video> \n"
        "Client utterance: {user_question} \n"

        "[GUIDELINE] \n"
        "Understand the Client's emotion, follow Client's point of view and intention, express sympathy for Client's negative situation or approval of Client's positive situation. The response should not imply negative emotions or triggers toward anyone or anything, such as disgust, resentment, discrimation, hatred, etc while supporting user well-being. Keep the information in the response truthful, avoid misinformation. The response should open and honest, foster understanding, connection, and provide comfort or support. The response should safeguard human autonomy, identity and data dignity. Ensuring AI behaviors guide emotional health responsibly, avoiding manipulation or harm. "
        "You must follow the output format in [OUTPUT FORMAT] below, just print what is listed in [OUTPUT FORMAT], do not print anything more even your step thought. \n"
        "The [CONTEXT] is history of current conversation between 'Client' and 'Therapist'. And [CURRENT CONTEXT] is the current 'Client' turn in the conversation \n"
        "Now you are the 'Therapist' and you need to make an empathy response to the 'Client' based on the context. Let's think about it step by step: \n"
        "Step 1: Describe and understanding the context and content of the conversation \n"
        "Step 2: Predict the following and explain why for each components: \n"
            "Client's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n"
            "Therapist's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n"
            "Therapist's strategy: Choose only one from (Open questions, Approval, Self-disclosure, Restatement, Interpretation, Advisement, Communication Skills, \n"
            "Structuring the therapy, Guiding the pace and depth of the conversation, Others). \n"
        "Step 3: You are the 'Therapist', think about how to reply to 'Client' in empathy. \n"
        "Step 4: You need to consider the potential impact of your reply, you can express a different posotion or opinion, but you should not hurt Client's feelings \n"

        "[OUTPUT FORMAT] \n"
        "Client's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n"
        "Therapist's emotion: Choose only one from (anger, sadness, disgust, depression, neutral, joy, fear). \n"
        "Therapist's strategy: Choose only one from (Open questions, Approval, Self-disclosure, Restatement, Interpretation, Advisement, Communication Skills, Structuring the therapy, Guiding the pace and depth of the conversation, Others). \n"
        "Therapist's response: [Generated response text]"

    ).format( 
        question=components["question"],
        choices=components["choices"],
    )
    conversation = [
        {"role": "system", "content": ("You are an expert in emotional psychology. Your task is to analyze the client's emotional state, predict the therapist's emotional response,"
                                        "determine the therapist's strategy, and generate an appropriate response based on the given inputs and historical context.")
        },
        {"role": "user", "content": total_prompt},
        {"role": "assistant", "content": label}
    ]

    return conversation

def load_data(file_path):
  data = []
  data_frame = pd.read_csv(file_path)
  for index, row in data_frame.iterrows():
    data.append(row.to_dict())
  return data


# Dataset class
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path: str,
            tokenizer: transformers.PreTrainedTokenizer,
            data_args: DataArguments
        ):
        super(LazySupervisedDataset, self).__init__()
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.mix_sampler_tag = False
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.raw_data_samples = load_data(data_path)

        for idx, item in enumerate(self.raw_data_samples):
            if item is None:
                print(f"None found at index {idx}")

    def __len__(self):
        return len(self.raw_data_samples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.raw_data_samples[i]
        # print(sources)
        # Extract raw data
        question = sources['question']
        choices = sources['choices']

        answer = sources['answer']

        components = {
            "question" : question,
            "choices" : choices ,
            "answer" : answer,
        }
        conversation = get_conversation(components)
     
        data_dict = preprocess([conversation], self.tokenizer, modal_token='<video>')

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        return data_dict



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(instances)
        # instances is one batch
        IGNORE_INDEX = -100
  
        total_prompt_ids = []
        label_ids = []
        for instance in instances:
            total_prompt_ids.append(instance['total_prompt_ids']['input_ids'].view(-1))
            label_ids.append(instance['label_ids']['input_ids'].view(-1))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            total_prompt_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(label_ids,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        return batch

# TODO: create dataloader here
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_args=data_args
    )
    # nn.Dataset load csv + video + audio
    # -> nn.Dataloader
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)





def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # BUG: High version transformers report error:
            # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time
            # load_in_4bit=training_args.bits == 4, # Use these two lines when training gemma2 and comment the config
            # load_in_8bit=training_args.bits == 8
            quantization_config=BitsAndBytesConfig( # Use this config when training other models and comment the above two lines
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model_args.hidden_size = config.hidden_size
    # Parameters for mamba

    model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_path,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False


    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    with open("model_arch.txt", "w") as f:
        f.write("Trainable parameters:\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"{name}: {param.size()}\n")
        f.write("\nFull architecture:\n")
        f.write(str(model))


    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # select a Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer = Trainer(model=model,
                          tokenizer=tokenizer,
                          args=training_args,
                          **data_module,
                          callbacks=[ClearCacheCallback()])


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()