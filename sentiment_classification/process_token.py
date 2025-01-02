import os
import pandas as pd
from transformers import BertTokenizerFast
from tqdm import tqdm
import argparse

# BertTokenizerFast 초기화
tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")

def tokenize_and_save(input_file, output_dir="./results/token/"):
    """
    TSV 파일을 읽어 문장을 토큰화하고, 결과를 CSV 파일로 저장.
    
    Args:
        input_file (str): 입력 TSV 파일 경로.
        output_dir (str): 결과 CSV 파일을 저장할 디렉토리 경로.
    """
    # 저장 경로가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TSV 파일 읽기
    data = pd.read_csv(input_file, sep="\t", header=None, names=["label", "text"])
    
    # 토큰화 결과 저장 리스트
    tokenized_data = []

    # 토큰화 진행
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Tokenizing documents"):
        label = row["label"]
        text = row["text"]
        
        # 토큰화 수행
        token_ids = tokenizer.encode(text, truncation=False, add_special_tokens=True)
        token_strings = tokenizer.convert_ids_to_tokens(token_ids)  # ID를 토큰으로 변환
        
        # 결과 저장
        tokenized_data.append({
            "label": label,
            "text": text,
            "tokens": " ".join(token_strings)  # 토큰을 공백으로 구분된 문자열로 저장
        })

    # 토큰화된 데이터프레임 생성
    tokenized_df = pd.DataFrame(tokenized_data)

    # 저장할 파일명 생성
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("token_") and f.endswith(".csv")]
    next_file_number = len(existing_files) + 1
    output_file = os.path.join(output_dir, f"token_{next_file_number}.csv")

    # 결과 저장
    tokenized_df.to_csv(output_file, index=False)
    print(f"Tokenized data saved to: {output_file}")


def parse_arguments():
    """
    명령줄 인자 파싱
    """
    parser = argparse.ArgumentParser(description="Tokenize and save TSV data.")
    parser.add_argument("--input_file", type=str, required=True, help="Input TSV file path.")
    parser.add_argument("--output_dir", type=str, default="./results/token/", help="Directory to save the output files.")
    return parser.parse_args()


if __name__ == "__main__":
    # 명령줄 인자 파싱
    args = parse_arguments()
    
    # 입력 파일과 출력 디렉토리를 인자로 전달
    tokenize_and_save(input_file=args.input_file, output_dir=args.output_dir)
