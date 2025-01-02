import pandas as pd

def remove_empty_rows(input_file, output_file):
    # TSV 파일 읽기
    data = pd.read_csv(input_file, sep='\t', header=None, names=['label', 'text'])
    
    # 빈 텍스트 제거
    cleaned_data = data.dropna(subset=['text'])  # NaN 제거
    cleaned_data = cleaned_data[cleaned_data['text'].str.strip() != '']  # 공백만 있는 텍스트 제거
    
    # 결과를 TSV 파일로 저장
    cleaned_data.to_csv(output_file, sep='\t', index=False, header=False)

# 사용 예시
input_file = "/home/jinwon/workspace/5-plm/nsmc/train15.tsv"  # 입력 파일 경로
output_file = "/home/jinwon/workspace/5-plm/nsmc/train15_clean.tsv"  # 출력 파일 경로
remove_empty_rows(input_file, output_file)