import pandas as pd
import argparse
import re
from soynlp.normalizer import repeat_normalize
import emoji

def clean_text(text):
    """
    텍스트 데이터를 정제합니다.
    Args:
        text (str): 원본 텍스트.
        max_length (int): 최대 토큰 길이.
    Returns:
        str: 정제된 텍스트.
    """
    # 이모지와 특수문자 필터링 패턴
    emojis = ''.join(emoji.EMOJI_DATA.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    )

    # # repeat_normalize 임의로 정의
    # doublespace_pattern = re.compile('\s+')
    # repeatchars_pattern = re.compile('(\w)\\1{2,}')

    # def repeat_normalize(sent, num_repeats=2):
    #     if num_repeats > 0:
    #         sent = repeatchars_pattern.sub('\\1' * num_repeats, sent)
    #     sent = doublespace_pattern.sub(' ', sent)
    #     return sent.strip()

    # 특수문자 제거
    text = pattern.sub(' ', text)
    text = url_pattern.sub('', text)
    text = text.strip()
    text = repeat_normalize(text, num_repeats=2)  # 2회 이상 반복되는 문자 처리
    # repeat_pattern = re.compile(r'(ㅋ|ㅎ|ㅠ|ㅜ|!)\1{2}')  # 'ㅋㅋㅋ', 'ㅎㅎㅎ', 'ㅠㅠㅠ' 등을 'ㅋㅋ', 'ㅎㅎ', 'ㅠㅠ'로 변환
    # text = repeat_pattern.sub(r'\1\1', text)

    return text


def convert_to_tsv(input_file, output_file):
    """
    데이터 파일을 읽고 TSV 파일로 변환하며 정제 과정을 추가합니다.
    Args:
        input_file (str): 원본 데이터 파일 경로 (.txt 파일).
        output_file (str): 변환된 데이터 파일 경로 (.tsv 파일).
    """
    # 데이터 파일 로드
    df = pd.read_csv(input_file, delimiter='\t', names=['id', 'document', 'label'], header=0)

    # 중복값 제거 
    df = df.drop_duplicates(subset=['document'])
    
    # 결측값 제거
    df = df.dropna(subset=['document', 'label'])

    # 레이블 값을 1 -> positive, 0 -> negative로 변환
    df['label'] = df['label'].apply(lambda x: 'positive' if int(x) == 1 else 'negative')

    # 텍스트 데이터 전처리
    df['document'] = df['document'].map(lambda x: clean_text(str(x)))

    # 빈 텍스트 제거
    df = df[df['document'].str.strip() != '']

    # 필요한 열만 추출
    df = df[['label', 'document']]

    # TSV 파일로 저장 (헤더 제거)
    df.to_csv(output_file, sep='\t', index=False, header=False)

    print(f"파일이 변환되었습니다: {output_file}")


def main(args):
    # 데이터 변환 함수 호출
    convert_to_tsv(args.input_file, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ratings data to TSV format with preprocessing.")
    parser.add_argument('--input_file', type=str, help='Path to the input .txt file')
    parser.add_argument('--output_file', type=str, help='Path to the output .tsv file')

    args = parser.parse_args()
    main(args)


# 형태소분석기 사용할때만 제거
    
# import pandas as pd
# import argparse
# import re
# from soynlp.normalizer import repeat_normalize  # soynlp 패키지 사용
# from konlpy.tag import Komoran, Okt
# import emoji

# # komoran = Komoran()
# okt = Okt()

# # Komoran 에서 제거할 품사태그
# # REMOVE_POS = {'JX', 'JKB', 'JKO', 'JKG', 'JKC', 'JKS', 'JC', 'SF', 'SP', 'SE', 'SSO', 'SSC', 'SC'}

# # Otk 에서 제거할 품사태그
# REMOVE_POS = {'Josa', 'Punctuation', 'Conjunction'}


# def clean_text(text):
#     """
#     텍스트 데이터를 정제합니다.
#     Args:
#         text (str): 원본 텍스트.
#     Returns:
#         str: 정제된 텍스트.
#     """
#     # 이모지와 특수문자 필터링 패턴
#     emojis = ''.join(emoji.EMOJI_DATA.keys())
#     pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
#     url_pattern = re.compile(
#         r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
#     )

#     # 반복 문자 처리 (추가 정규식 사용)
#     text = repeat_normalize(text, num_repeats=2)  # 4회 이상 반복되는 문자 처리
#     repeat_pattern = re.compile(r'(ㅋ|ㅎ|ㅠ|ㅜ|!)\1{2}')  # 'ㅋㅋㅋ', 'ㅎㅎㅎ', 'ㅠㅠㅠ' 등을 'ㅋㅋ', 'ㅎㅎ', 'ㅠㅠ'로 변환
#     text = repeat_pattern.sub(r'\1\1', text)

#     # 특수문자 제거
#     text = pattern.sub(' ', text)

#     # URL 제거
#     text = url_pattern.sub('', text)

#     # 앞뒤 공백 제거
#     text = text.strip()

#     # 품사 태깅 (빈 문자열 체크)
#     if not text:
#         return ''  # 빈 문자열 반환
    
#     tagged_tokens = okt.pos(text, norm=True, stem=True)

#     # 형태소 분석 결과 확인 및 처리
#     if not tagged_tokens:
#         return ''  # 형태소 분석 결과가 비어 있으면 빈 문자열 반환

#     # 불필요한 품사 제거 및 어간 추출
#     filtered_tokens = [
#         word for word, pos in tagged_tokens
#         if pos not in REMOVE_POS
#     ]

#     # 필터링 결과 확인
#     if not filtered_tokens:
#         return ''  # 필터링 후 빈 결과일 경우 빈 문자열 반환

#     # 결과를 문자열로 결합하여 반환
#     return ' '.join(filtered_tokens)



# def convert_to_tsv(input_file, output_file):
#     """
#     데이터 파일을 읽고 TSV 파일로 변환하며 정제 과정을 추가합니다.
#     Args:
#         input_file (str): 원본 데이터 파일 경로 (.txt 파일).
#         output_file (str): 변환된 데이터 파일 경로 (.tsv 파일).
#     """
#     # 데이터 파일 로드
#     df = pd.read_csv(input_file, delimiter='\t', names=['id', 'document', 'label'], header=0)

#     # 중복값 제거 
#     df = df.drop_duplicates(subset=['document'])
    
#     # 결측값 제거
#     df = df.dropna(subset=['document', 'label'])

#     # 레이블 값을 1 -> positive, 0 -> negative로 변환
#     df['label'] = df['label'].apply(lambda x: 'positive' if int(x) == 1 else 'negative')

#     # 텍스트 데이터 전처리
#     df['document'] = df['document'].map(lambda x: clean_text(str(x)))

#     # 빈 텍스트 제거
#     df = df[df['document'].str.strip() != '']

#     # 필요한 열만 추출
#     df = df[['label', 'document']]

#     # TSV 파일로 저장 (헤더 제거)
#     df.to_csv(output_file, sep='\t', index=False, header=False)

#     print(f"파일이 변환되었습니다: {output_file}")


# def main(args):
#     # 데이터 변환 함수 호출
#     convert_to_tsv(args.input_file, args.output_file)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Convert ratings data to TSV format with preprocessing.")
#     parser.add_argument('--input_file', type=str, help='Path to the input .txt file')
#     parser.add_argument('--output_file', type=str, help='Path to the output .tsv file')

#     args = parser.parse_args()
#     main(args)


# # 바른 형태소 분석기 사용할때만 사용
# import pandas as pd
# import argparse
# import re
# from soynlp.normalizer import repeat_normalize
# from bareunpy import Tagger
# import emoji

# # Bareun API 초기화
# API_KEY = "koba-RRCBU3I-Q76EJ3I-T65LKUI-NUKMWCQ"  # 발급받은 API KEY를 입력하세요.
# bareun_tagger = Tagger(API_KEY, "localhost", 5757)

# # # 태그 번호와 문자열 매핑
# # TAG_MAP = {
# #     13: 'JKS', 9: 'JKC', 10: 'JKG', 11: 'JKO', 8: 'JKB', 14: 'JKV', 12: 'JKQ', 15: 'JX', 7: 'JC',
# #     30: 'SF', 35: 'SP', 36: 'SS', 29: 'SE', 34: 'SO'
# # }

# # 조사만 제거할때
# TAG_MAP = {
#     13: 'JKS', 9: 'JKC', 10: 'JKG', 11: 'JKO', 8: 'JKB', 14: 'JKV', 12: 'JKQ', 15: 'JX', 7: 'JC'
# }

# # 구두점만 제거할때
# # TAG_MAP = {
# #     30: 'SF', 35: 'SP', 36: 'SS', 29: 'SE', 34: 'SO'
# # }

# def clean_text_with_bareun(text):
#     """
#     Bareun 형태소 분석기를 사용하여 텍스트를 정제합니다.
#     Args:
#         text (str): 원본 텍스트.
#     Returns:
#         str: 정제된 텍스트.
#     """
#     # 빈 문자열 체크
#     if not text.strip():
#         return ''
    
#     # 이모지와 특수문자 필터링 패턴
#     emojis = ''.join(emoji.EMOJI_DATA.keys())
#     pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
#     url_pattern = re.compile(
#         r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
#     )

#     text = pattern.sub(' ', text)
#     text = url_pattern.sub('', text)
#     text = text.strip()
#     text = repeat_normalize(text, num_repeats=2)  # 반복 문자 처리
    
#     # 공백을 "A"로 대체
#     text = text.replace(' ', 'A')
    
#     # Bareun 형태소 분석기 호출
#     res = bareun_tagger.tags([text])  # Bareun은 리스트 형태로 입력받음
#     processed_tokens = []

#     for sentence in res.msg().sentences:
#         for token in sentence.tokens:
#             for morph in token.morphemes:
#                 # 태그 번호를 문자열로 변환
#                 morph_tag_str = TAG_MAP.get(morph.tag, None)
                
#                 # TAG_MAP에 매핑된 태그일 경우 제거
#                 if morph_tag_str is None:  # TAG_MAP에 없는 경우만 추가
#                     processed_tokens.append(morph.text.content)
    
#     # 형태소 분석 결과를 공백으로 결합
#     processed_text = ''.join(processed_tokens)

#     # "A"를 다시 빈칸으로 복원
#     processed_text = processed_text.replace('A', ' ')
    
#     # 다중 공백 제거
#     processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
#     return processed_text

# def convert_to_tsv(input_file, output_file):
#     """
#     데이터 파일을 읽고 TSV 파일로 변환하며 정제 과정을 추가합니다.
#     Args:
#         input_file (str): 원본 데이터 파일 경로 (.txt 파일).
#         output_file (str): 변환된 데이터 파일 경로 (.tsv 파일).
#     """
#     # 데이터 파일 로드
#     df = pd.read_csv(input_file, delimiter='\t', names=['id', 'document', 'label'], header=0)

#     # 중복값 제거 
#     df = df.drop_duplicates(subset=['document'])
    
#     # 결측값 제거
#     df = df.dropna(subset=['document', 'label'])

#     # 레이블 값을 1 -> positive, 0 -> negative로 변환
#     df['label'] = df['label'].apply(lambda x: 'positive' if int(x) == 1 else 'negative')

#     # 텍스트 데이터 전처리
#     df['document'] = df['document'].map(lambda x: clean_text_with_bareun(str(x)))

#     # 빈 텍스트 제거
#     df = df[df['document'].str.strip() != '']

#     # 필요한 열만 추출
#     df = df[['label', 'document']]

#     # TSV 파일로 저장 (헤더 제거)
#     df.to_csv(output_file, sep='\t', index=False, header=False)

#     print(f"파일이 변환되었습니다: {output_file}")


# def main(args):
#     # 데이터 변환 함수 호출
#     convert_to_tsv(args.input_file, args.output_file)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Convert ratings data to TSV format with preprocessing.")
#     parser.add_argument('--input_file', type=str, help='Path to the input .txt file')
#     parser.add_argument('--output_file', type=str, help='Path to the output .tsv file')

#     args = parser.parse_args()
#     main(args)


# # 형태소 분석 후, 불용어사전을 통해서 불용어 제거
# import pandas as pd
# import argparse
# import re
# from soynlp.normalizer import repeat_normalize
# from bareunpy import Tagger
# import emoji

# # Bareun API 초기화
# API_KEY = "koba-RRCBU3I-Q76EJ3I-T65LKUI-NUKMWCQ" 
# bareun_tagger = Tagger(API_KEY, "localhost", 5757)

# def load_stopwords(filepath):
#     """
#     불용어 리스트를 파일에서 로드합니다.
#     Args:
#         filepath (str): 불용어 사전 파일 경로.
#     Returns:
#         set: 불용어 단어 집합.
#     """
#     with open(filepath, 'r', encoding='utf-8') as f:
#         stopwords = {line.split('\t')[0].strip() for line in f if line.strip()}
#     return stopwords

# def clean_text_with_stopwords(text, stopwords):
#     """
#     Bareun 형태소 분석기를 사용하여 텍스트를 정제하며 불용어를 제거합니다.
#     Args:
#         text (str): 원본 텍스트.
#         stopwords (set): 불용어 집합.
#     Returns:
#         str: 불용어 제거 후 텍스트.
#     """
#     # 빈 문자열 체크
#     if not text.strip():
#         return ''
    
#     # 이모지와 특수문자 필터링 패턴
#     emojis = ''.join(emoji.EMOJI_DATA.keys())
#     pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
#     url_pattern = re.compile(
#         r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
#     )

#     text = pattern.sub(' ', text)
#     text = url_pattern.sub('', text)
#     text = text.strip()
#     text = repeat_normalize(text, num_repeats=2)  # 반복 문자 처리
    
#     # 공백을 "A"로 대체
#     text = text.replace(' ', 'A')
    
#     # Bareun 형태소 분석기 호출
#     res = bareun_tagger.tags([text])  # Bareun은 리스트 형태로 입력받음
#     processed_tokens = []

#     for sentence in res.msg().sentences:
#         for token in sentence.tokens:
#             for morph in token.morphemes:
#                 # 불용어 사전에 포함되지 않은 단어만 추가
#                 if morph.text.content not in stopwords:
#                     processed_tokens.append(morph.text.content)
    
#     # 형태소 분석 결과를 공백으로 결합
#     processed_text = ''.join(processed_tokens)

#     # "A"를 다시 빈칸으로 복원
#     processed_text = processed_text.replace('A', ' ')
    
#     # 다중 공백 제거
#     processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
#     return processed_text

# def convert_to_tsv(input_file, output_file, stopwords):
#     """
#     데이터 파일을 읽고 TSV 파일로 변환하며 정제 과정을 추가합니다.
#     Args:
#         input_file (str): 원본 데이터 파일 경로 (.txt 파일).
#         output_file (str): 변환된 데이터 파일 경로 (.tsv 파일).
#         stopwords (set): 불용어 집합.
#     """
#     # 데이터 파일 로드
#     df = pd.read_csv(input_file, delimiter='\t', names=['id', 'document', 'label'], header=0)

#     # 중복값 제거 
#     df = df.drop_duplicates(subset=['document'])
    
#     # 결측값 제거
#     df = df.dropna(subset=['document', 'label'])

#     # 레이블 값을 1 -> positive, 0 -> negative로 변환
#     df['label'] = df['label'].apply(lambda x: 'positive' if int(x) == 1 else 'negative')

#     # 텍스트 데이터 전처리
#     df['document'] = df['document'].map(lambda x: clean_text_with_stopwords(str(x), stopwords))

#     # 빈 텍스트 제거
#     df = df[df['document'].str.strip() != '']

#     # 필요한 열만 추출
#     df = df[['label', 'document']]

#     # TSV 파일로 저장 (헤더 제거)
#     df.to_csv(output_file, sep='\t', index=False, header=False)

#     print(f"파일이 변환되었습니다: {output_file}")

# def main(args):
#     # 불용어 파일 로드
#     stopwords = load_stopwords(args.stopwords_file)

#     # 데이터 변환 함수 호출
#     convert_to_tsv(args.input_file, args.output_file, stopwords)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Convert ratings data to TSV format with preprocessing.")
#     parser.add_argument('--input_file', type=str, help='Path to the input .txt file')
#     parser.add_argument('--output_file', type=str, help='Path to the output .tsv file')
#     parser.add_argument('--stopwords_file', type=str, help='Path to the stopwords file')

#     args = parser.parse_args()
#     main(args)


# # 영어, 숫자 데이터 지우는 코드 추가
# import pandas as pd
# import argparse
# import re
# import emoji


# def clean_text(text):
#     """
#     텍스트 데이터를 정제합니다.
#     Args:
#         text (str): 원본 텍스트.
#     Returns:
#         str: 정제된 텍스트.
#     """
#     # 이모지와 특수문자 필터링 패턴
#     emojis = ''.join(emoji.EMOJI_DATA.keys())
#     pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
#     url_pattern = re.compile(
#         r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
#     )

#     # 특수문자 제거
#     text = pattern.sub(' ', text)
#     text = url_pattern.sub('', text)
#     text = text.strip()
#     text = repeat_normalize(text, num_repeats=2)

#     return text


# def is_only_english(text):
#     """
#     텍스트가 영어만 포함하는지 확인합니다.
#     Args:
#         text (str): 입력 텍스트.
#     Returns:
#         bool: 영어만 포함 여부.
#     """
#     return bool(re.fullmatch(r'[a-zA-Z\s.,?!\'"]+', text))


# def is_only_numbers(text):
#     """
#     텍스트가 숫자만 포함하는지 확인합니다.
#     Args:
#         text (str): 입력 텍스트.
#     Returns:
#         bool: 숫자만 포함 여부.
#     """
#     return bool(re.fullmatch(r'\d+', text))


# def contains_english_and_special_chars(text):
#     """
#     텍스트에 영어와 특수문자가 섞여있는지 확인합니다.
#     Args:
#         text (str): 입력 텍스트.
#     Returns:
#         bool: 영어와 특수문자 포함 여부.
#     """
#     return bool(re.search(r'[a-zA-Z]', text)) and bool(re.search(r'[^a-zA-Z0-9\s]', text))


# def contains_numbers_and_special_chars(text):
#     """
#     텍스트에 숫자와 특수문자가 섞여있는지 확인합니다.
#     Args:
#         text (str): 입력 텍스트.
#     Returns:
#         bool: 숫자와 특수문자 포함 여부.
#     """
#     return bool(re.search(r'\d', text)) and bool(re.search(r'[^a-zA-Z0-9\s]', text))


# def convert_to_tsv(input_file, output_file):
#     """
#     데이터 파일을 읽고 TSV 파일로 변환하며 정제 과정을 추가합니다.
#     Args:
#         input_file (str): 원본 데이터 파일 경로 (.txt 파일).
#         output_file (str): 변환된 데이터 파일 경로 (.tsv 파일).
#     """
#     # 데이터 파일 로드
#     df = pd.read_csv(input_file, delimiter='\t', names=['id', 'document', 'label'], header=0)

#     # 중복값 제거
#     df = df.drop_duplicates(subset=['document'])

#     # 결측값 제거
#     df = df.dropna(subset=['document', 'label'])

#     # 레이블 값을 1 -> positive, 0 -> negative로 변환
#     df['label'] = df['label'].apply(lambda x: 'positive' if int(x) == 1 else 'negative')

#     # 텍스트 데이터 전처리
#     df['document'] = df['document'].map(lambda x: clean_text(str(x)))

#     # 빈 텍스트 제거
#     df = df[df['document'].str.strip() != '']

#     # 조건에 따라 데이터 제거
#     df = df[~df['document'].map(is_only_english)]  # 영어만 있는 데이터 제거
#     df = df[~df['document'].map(is_only_numbers)]  # 숫자만 있는 데이터 제거
#     df = df[~df['document'].map(contains_english_and_special_chars)]  # 영어 + 특수문자 제거
#     df = df[~df['document'].map(contains_numbers_and_special_chars)]  # 숫자 + 특수문자 제거

#     # 빈 텍스트 제거 (다시확인)
#     df = df[df['document'].str.strip() != '']

#     # 필요한 열만 추출
#     df = df[['label', 'document']]

#     # TSV 파일로 저장 (헤더 제거)
#     df.to_csv(output_file, sep='\t', index=False, header=False)

#     print(f"파일이 변환되었습니다: {output_file}")


# def main(args):
#     # 데이터 변환 함수 호출
#     convert_to_tsv(args.input_file, args.output_file)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Convert ratings data to TSV format with preprocessing.")
#     parser.add_argument('--input_file', type=str, help='Path to the input .txt file')
#     parser.add_argument('--output_file', type=str, help='Path to the output .tsv file')

#     args = parser.parse_args()
#     main(args)
