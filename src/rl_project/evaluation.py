from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

# 1. 평가할 모델 경로들
models_to_test = {
    "Baseline": "Qwen/Qwen3-0.6B-Base",
    "SFT": "kwkim1030/Qwen3-0.6-Countdwon-SoS-SFT",
    "RLOO": "kwkim1030/Qwen3-0.6-Countdwon-SoS-RLOO"
}

# 2. 테스트 데이터 로드 (학습에 안 쓴 뒷부분 5000개)
dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train[-1000:]')
filtered_dataset = dataset.filter(lambda example: len(example['nums']) == 4)

def evaluate_model(model_path, dataset, llm, sampling_parms):    
    correct_count = 0
    format_error_count = 0
    total = len(dataset)

    for example in tqdm(dataset):
        # 프롬프트 생성
        prompt = (
            f"<|im_start|>system\n당신은 숫자를 조합하여 목표값을 만드는 수학 전문가입니다.<|im_end|>\n"
            f"<|im_start|>user\nnums: {example['nums']}, target: {example['target']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
        prompts = [prompt] 
        outputs = llm.generate(prompts, sampling_params)
        completions = outputs[0].outputs[0].text

        # --- 채점 로직 (간소화) ---
        try:
            # 수식 추출 (Format 체크)
            if "</think>" in completions:
                answer_part = completions.split("</think>")[1].strip()
            else:
                answer_part = completions # 태그 없으면 전체를 봄 (SFT의 경우)
                format_error_count += 1            
            
            clean = re.sub(r"[^0-9+\-*/()]", "", answer_part)
            # 정답 확인
            pred_val = eval(clean)
            print(pred_val)

            if abs(pred_val - example['target']) < 1e-5:
                correct_count += 1
        except:
            pass # 파싱 에러나 계산 에러는 오답 처리

    accuracy = (correct_count / total) * 100
    format_error_rate = (format_error_count / total) * 100
    
    return accuracy, format_error_rate

# 실행 및 결과 출력
results = {}
for name, path in models_to_test.items():
    print(f"Evaluating {path}...")
    llm = LLM(
        model=path, 
        max_model_len=4096,
        dtype="bfloat16",           
        gpu_memory_utilization=0.6
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=4096,           # 생성 최대 길이
        stop=["<|im_end|>"]        # 생성 중단 토큰 (필수)
    )
    acc, err = evaluate_model(path, dataset, llm, sampling_params)
    results[name] = {"Accuracy": acc, "Format Error": err}

print("\n=== 실험 결과표 ===")
print(results)

df = pd.DataFrame(results).T
df = df.round(2)

df.to_csv("experiment_table.csv")

def draw_comparison_chart(df):
    # 스타일 설정 (논문용 깔끔한 스타일)
    sns.set_theme(style="whitegrid")
    
    # 데이터 형태 변환 (Plotting을 위해 Long-form으로 변경)
    # 인덱스(모델명)를 컬럼으로 빼냄
    df_reset = df.reset_index().rename(columns={"index": "Model"})
    
    # 'Model'을 기준으로 나머지 컬럼들을 'Metric', 'Value'로 녹임(Melt)
    df_melted = df_reset.melt(
        id_vars="Model", 
        var_name="Metric", 
        value_name="Score"
    )

    # 차트 그리기
    plt.figure(figsize=(10, 6))
    
    # Bar plot 생성
    chart = sns.barplot(
        data=df_melted, 
        x="Metric", 
        y="Score", 
        hue="Model", 
        palette="viridis" # 색상 테마 (muted, deep, viridis 등)
    )

    # 값 표시 (Bar 위에 숫자 적기)
    for container in chart.containers:
        chart.bar_label(container, fmt='%.1f', padding=3)

    plt.title("Comparison of SFT vs RLOO Performance", fontsize=15, pad=20)
    plt.ylabel("Score / Percentage (%)", fontsize=12)
    plt.xlabel("")
    plt.legend(title="Model Architecture")
    
    # 저장
    plt.tight_layout()
    plt.savefig("experiment_figure.png", dpi=300) # 고해상도 저장
    print("\n[Figure 1] 차트가 'experiment_figure.png'로 저장되었습니다.")
    plt.show()

# 실행
draw_comparison_chart(df)