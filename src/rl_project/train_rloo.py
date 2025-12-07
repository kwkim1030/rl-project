import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import RLOOTrainer, RLOOConfig
import wandb

from dataset_rloo import build_dataset

SFT_MODEL_PATH = "kwkim1030/Qwen3-0.6-Countdwon-SoS-SFT"
DATA_PATH = "Jiayi-Pan/Countdown-Tasks-3to4"
OUTPUT_DIR = "./Qwen3-0.6-Countdown-SoS-RLOO"

wandb.init(
    project="Countdown-SoS",  # 프로젝트 이름 (폴더 같은 개념)
    name="sos-rloo-experiment-1",         # 이번 실행의 이름
    config={"model": "Qwen3-0.6-Countdwon-SoS-SFT", "lr": 1e-6} # (선택) 추가로 기록할 정보
)


def rule_based_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        reward = 0.0
        
        try:
            match = re.search(r"target\s*:\s*(\d+)", prompt[1]['content'])
            
            if not match:
                rewards.append(0.0) # 프롬프트 파싱 실패 시 0점
                continue
            
            target_val = int(match.group(1))
        except:
            rewards.append(0.0)
            continue
        # 2. 포맷 보상 (Format Reward)
        # <think>...</think> 정답 패턴 이있는지 확인
        pattern = r"^<think>([\s\S]*?)</think>\s*([\s\S]*?)$"
        completion_content = completion[0]["content"]
        
        has_think = re.match(pattern, completion_content)
        if not has_think:
            rewards.append(-5.0) # 생각 태그가 없으면 큰 
            continue
        
        try:
            think_content = has_think.group(1)
            answer_content = has_think.group(2)
            
        except:
            rewards.append(-0.5)
            return rewards

        if "<|im_end|>" in answer_content:
            answer_content = answer_content.split("<|im_end|>")[0].strip()
            
        clean = re.sub(r"[^0-9+\-*/()]", "", answer_content)
        if not clean:
            rewards.append(-5.0) # 태그 뒤에 수식이 없으면 페널티
            print(f"4 : {rewards}")
            continue
        
        # 여기까지 왔으면 기본 포맷 점수 부여
        reward +=  10.0
        
        # "숫자 연산자 숫자 = 숫자" 패턴 찾기
        equations = re.findall(r"(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)", think_content)
        
        for n1, op, n2, res in equations:
            try:
                # 실제 계산 수행
                # eval은 보안상 위험할 수 있으나, 여기서는 숫자만 파싱했으므로 안전
                correct_res = eval(f"{n1}{op}{n2}")
                if abs(correct_res - int(res)) < 1e-5: # 정수 비교지만 float 안전장치
                    reward += 0.05  # 올바른 계산 스텝마다 가산점
                else:
                    reward -= 0.1  # 틀린 계산은 감점 (Hallucination 방지)
            except:
                reward -= 0.1
        # 4. 최종 정답 보상 (Outcome Reward)
        try:
            final_val = eval(clean)
            
            if abs(final_val - target_val) < 1e-5:
                reward += 20.0  # 정답 맞히면 큰 점수
            else:
                reward -= 0.5  # 틀리면 감점
        except:
            # 수식 파싱 에러 (괄호 짝 안맞음 등)
            reward -= 3.0
        
        rewards.append(reward)
        
    return rewards


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)

    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # RL 학습 대상 모델
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2", 
    )

    # 3. RLOO 설정 (TrainingArguments 역할)
    rloo_config = RLOOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,          # RL은 보통 1 epoch만 해도 충분 (또는 step 기준)
        learning_rate=1e-6,          
        num_generations=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        bf16=True,
        report_to="wandb",
        run_name="sos-rloo-experiment-3",
        logging_strategy="steps",
        logging_steps=10,
        
        use_vllm=True, 
        vllm_mode="colocate",

        max_prompt_length=512,       # 입력(질문) 길이 제한
        max_completion_length=4096,  # 생성(답변) 길이 제한 (<think> 포함 넉넉히)
    )

    # 2. 데이터셋 준비
    train_dataset = build_dataset(DATA_PATH)

    trainer = RLOOTrainer(
        reward_funcs=rule_based_reward_fn,
        model=model,
        processing_class=tokenizer,
        args=rloo_config,
        train_dataset=train_dataset,
    )

    print("Starting RLOO Training...")
    trainer.train()
    
    print("Saving RLOO Aligned Model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))