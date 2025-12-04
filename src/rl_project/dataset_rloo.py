from datasets import load_dataset


def build_dataset(name):
    # 1. 데이터 로드
    ds = load_dataset(name, split="train[:20000]")
    filtered_ds = ds.filter(lambda example: len(example['nums']) == 4)

    def format_prompt(example):
        # 메시지 리스트 구조로 정의
        messages = [
            {"role": "system", "content": "당신은 숫자를 조합하여 목표값을 만드는 수학 전문가입니다."},
            {"role": "user", "content": f"nums: {example['nums']}, target: {example['target']}"}
        ]
        
        return {"prompt": messages}

    # 병렬 처리로 속도 향상
    return filtered_ds.map(format_prompt, num_proc=4)
