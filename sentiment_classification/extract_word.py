import pandas as pd
from collections import Counter
import os

# 파일 경로 설정
input_file = "/home/jinwon/workspace/5-plm/nsmc/train.tsv"
output_dir = "/home/jinwon/workspace/5-plm/sentiment_classification/results/EDA"
os.makedirs(output_dir, exist_ok=True)

# 실행될 때마다 파일 이름에 숫자 추가
def get_next_filename(base_dir, base_name, ext):
    counter = 1
    while True:
        file_name = f"{base_name}_{counter}{ext}"
        full_path = os.path.join(base_dir, file_name)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

positive_output_file = get_next_filename(output_dir, "positive_top_100", ".tsv")
negative_output_file = get_next_filename(output_dir, "negative_top_100", ".tsv")

# TSV 파일 읽기
data = pd.read_csv(input_file, sep="\t", header=None, names=["label", "text"])

# 라벨별 단어 추출 및 빈도 계산
def get_top_words(data, label, top_n=100):
    # 특정 라벨에 해당하는 텍스트만 필터링
    filtered_texts = data[data["label"] == label]["text"].tolist()
    
    # 단어 토큰화
    words = []
    for text in filtered_texts:
        words.extend(text.split())

    # 단어 빈도 계산
    word_counts = Counter(words)

    # 상위 N개 단어 반환
    return word_counts.most_common(top_n)

# 긍정, 부정 라벨별로 단어 추출
top_positive_words = get_top_words(data, "positive", top_n=100)
top_negative_words = get_top_words(data, "negative", top_n=100)

# 결과를 데이터프레임으로 변환
positive_df = pd.DataFrame(top_positive_words, columns=["word", "count"])
negative_df = pd.DataFrame(top_negative_words, columns=["word", "count"])

# 결과를 TSV 파일로 저장
positive_df.to_csv(positive_output_file, sep="\t", index=False)
negative_df.to_csv(negative_output_file, sep="\t", index=False)

print(f"Top 100 words for 'positive' label saved to {positive_output_file}")
print(f"Top 100 words for 'negative' label saved to {negative_output_file}")
