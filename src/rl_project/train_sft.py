import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch

from dataset_sft import CountDownDatasetBuilder

MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DATA_PATH = "./data/train1_b4_t100_n500000_random.json"
OUTPUT_DIR = "./Qwen3-0.6-Countdown-SFT"


class CountDownSFTTrainer:
    def __init__(self, model_id, output_dir="./Qwen3-0.6B-Countdown-SFT"):
        self.model_id = model_id
        self.output_dir = output_dir
                
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,  
            device_map="auto",
            # Flash Attention 2 활성화
            # attn_implementation="flash_attention_2", 
            dtype=torch.bfloat16
        )
        return model

    def train(self, train_dataset, max_seq_length=3000):
        """학습 시작"""
        
        training_args = SFTConfig(
            output_dir=self.output_dir,
            logging_dir="./out/logs",
            num_train_epochs=2,
            learning_rate=2e-5,              
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            report_to="none",
            
            per_device_train_batch_size=2,   
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,     
            
            optim="paged_adamw_8bit",            
            bf16=True,
            
            max_length=max_seq_length,
            save_strategy="epoch",
            packing=False
        )

        print("Initializing SFTTrainer ...")
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            processing_class=self.tokenizer,
            args=training_args,
        )

        print("Starting Training...")
        trainer.train()

        print("Saving Model...")
        trainer.save_model(os.path.join(self.output_dir, "final_model"))


if __name__ == "__main__":
    
    trainer_wrapper = CountDownSFTTrainer(
        model_id=MODEL_ID, 
        output_dir=OUTPUT_DIR
    )
    
    data_builder = CountDownDatasetBuilder(
        file_path=DATA_PATH, 
        tokenizer=trainer_wrapper.tokenizer
    )

    train_dataset = data_builder.get_dataset()
    
    trainer_wrapper.train(train_dataset)