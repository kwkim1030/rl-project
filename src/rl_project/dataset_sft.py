import re
from datasets import load_dataset

class CountDownDatasetBuilder:
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

    def _convert_solution_to_expression(self, solution_list):
        expressions = {}
        last_result = ""
        for step in solution_list:
            match = re.match(r"(\d+)([\+\-\*\/])(\d+)=(\d+)", step)
            if match:
                left, op, right, res = match.groups()
                left_expr = expressions.get(left, left)
                right_expr = expressions.get(right, right)
                current_expr = f"({left_expr}{op}{right_expr})"
                expressions[res] = current_expr
                last_result = res
                
        return expressions.get(last_result, "")

    def _formatting_func(self, example):
        final_equation = self._convert_solution_to_expression(example['solution'])
        text = (
            f"<|im_start|>system\n당신은 숫자를 조합하여 목표값을 만드는 수학 전문가입니다.<|im_end|>\n"
            f"<|im_start|>user\nnums: {example['nums']}, target: {example['target']}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n{example['search_path']}\n</think>\n\n"
            f"{final_equation}<|im_end|>"
        )

        return {"text": text}

    def get_dataset(self):
        print(f"Loading data from {self.file_path}...")
        raw_dataset = load_dataset("json", data_files=self.file_path)
        print("Formatting dataset...")
        hf_datatset = raw_dataset['train'].select(range(int(1e5)))
        processed_dataset = hf_datatset.map(self._formatting_func, num_proc=4)

        return processed_dataset
    