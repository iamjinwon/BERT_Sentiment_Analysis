import openai  # GPT API 사용
import pandas as pd
import argparse
import re
from soynlp.normalizer import repeat_normalize
import emoji
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# GPT API 키 로드
openai.api_key = os.getenv("OPENAI_API_KEY")

# 비용 계산을 위한 변수 초기화
input_token_count = 0
output_token_count = 0

prompt = """
Translate the input Korean into English and output only the translated sentence without adding any other words.
"""

# GPT API를 이용한 한국어 → 영어 번역 함수
def translate_to_english_with_gpt(korean_text):
    global input_token_count, output_token_count
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": korean_text}
            ]
        )
        english_text = response['choices'][0]['message']['content'].strip()
        input_tokens = response['usage']['prompt_tokens']
        output_tokens = response['usage']['completion_tokens']
        input_token_count += input_tokens
        output_token_count += output_tokens
        print(f"Translated by GPT (EN): {english_text}")
        return english_text
    except Exception as e:
        print(f"GPT Translation 실패: {korean_text}, 오류: {e}")
        return ''

# GPT API를 이용한 영어 → 한국어 번역 함수
def translate_to_korean_with_gpt(english_text):
    global input_token_count, output_token_count
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": f"Translate the following English text to Korean:\n{english_text}"}
            ]
        )
        korean_text = response['choices'][0]['message']['content'].strip()
        input_tokens = response['usage']['prompt_tokens']
        output_tokens = response['usage']['completion_tokens']
        input_token_count += input_tokens
        output_token_count += output_tokens
        print(f"Translated by GPT (KO): {korean_text}")
        return korean_text
    except Exception as e:
        print(f"GPT Translation 실패: {english_text}, 오류: {e}")
        return ''

def is_english(text):
    """
    텍스트가 영어로만 이루어져 있는지 확인하는 함수
    """
    return bool(re.fullmatch(r'[a-zA-Z\s.,?!\'"]+', text))

# Back Translation 수행 함수
def back_translate(text):
    try:
        # 텍스트가 영어로만 이루어진 경우 스킵
        if is_english(text):
            print(f"Text is only English, skipping: {text}")
            return text

        # 1. 한국어 → 영어 번역 (GPT API 이용)
        english_text = translate_to_english_with_gpt(text)

        # 2. 영어 → 한국어 번역 (GPT API 이용)
        back_translated_text = translate_to_korean_with_gpt(english_text)

        return back_translated_text

    except Exception as e:
        print(f"Back Translation 실패: {text}, 오류: {e}")
        return ''



# 기존 텍스트 정제 함수
def clean_text(text):
    emojis = ''.join(emoji.EMOJI_DATA.keys())
    pattern = re.compile(f'[^\uac00-\ud7a3 .,?!/@$%~％·∼()\x00-\x7F{emojis}]+')  # 한글만 허용
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    )

    text = pattern.sub(' ', text)  # 특수문자 제거
    text = url_pattern.sub('', text)  # URL 제거
    text = text.strip()  # 앞뒤 공백 제거
    text = repeat_normalize(text, num_repeats=2)  # 반복 문자 처리

    return text if text else ''  # 빈 문자열 반환


def augment_with_back_translation(df):
    augmented_data = []

    for idx, row in df.iterrows():
        label = row['label']
        document = row['document']

        try:
            # Back Translation 수행
            back_translated = back_translate(document)

            # Skip된 텍스트는 결과 데이터에 추가하지 않음
            if back_translated and back_translated != document:
                augmented_data.append({'label': label, 'document': back_translated})

        except Exception as e:
            print(f"Back Translation 실패: {document}, 오류: {e}")

    return pd.DataFrame(augmented_data)


def convert_to_tsv(input_file, output_file):
    global input_token_count, output_token_count

    # 파일 읽기
    df = pd.read_csv(input_file, delimiter='\t', names=['id', 'document', 'label'], header=0)
    df = df.drop_duplicates(subset=['document']).dropna(subset=['document', 'label'])

    # 레이블 변환 (1 -> positive, 0 -> negative)
    df['label'] = df['label'].apply(lambda x: 'positive' if int(x) == 1 else 'negative')

    # 필요 없는 'id' 열 제거
    df = df[['label', 'document']]

    # 텍스트 정제
    df['document'] = df['document'].map(clean_text)
    df = df[df['document'].str.strip() != '']

    print(f"원본 데이터 크기: {len(df)}")

    # Back Translation 증강 적용
    augmented_df = augment_with_back_translation(df)
    print(f"증강된 데이터 크기: {len(augmented_df)}")

    # 역번역 데이터만 저장
    augmented_df.to_csv(output_file, sep='\t', index=False, header=False)
    print(f"파일이 변환되었습니다: {output_file}")

    # 비용 계산 및 출력
    input_cost = (input_token_count / 1000) * 0.0015  # 입력 토큰 비용
    output_cost = (output_token_count / 1000) * 0.002  # 출력 토큰 비용
    total_cost = input_cost + output_cost
    print(f"총 입력 토큰 수: {input_token_count}")
    print(f"총 출력 토큰 수: {output_token_count}")
    print(f"예상 비용: ${total_cost:.6f}")



# 메인 함수
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ratings data to TSV format with back-translation.")
    parser.add_argument('--input_file', type=str, help='Path to the input .txt file')
    parser.add_argument('--output_file', type=str, help='Path to the output .tsv file')

    args = parser.parse_args()
    convert_to_tsv(args.input_file, args.output_file)



# # MarianMTModel
# import pandas as pd
# import argparse
# import re
# from transformers import MarianMTModel, MarianTokenizer
# from soynlp.normalizer import repeat_normalize
# import emoji

# # 모델 및 토크나이저 초기화
# ko_en_model_name = "Helsinki-NLP/opus-mt-ko-en"
# en_ko_model_name = "halee9/translation_en_ko"

# ko_en_model = MarianMTModel.from_pretrained(ko_en_model_name)
# ko_en_tokenizer = MarianTokenizer.from_pretrained(ko_en_model_name)

# en_ko_model = MarianMTModel.from_pretrained(en_ko_model_name)
# en_ko_tokenizer = MarianTokenizer.from_pretrained(en_ko_model_name)

# # 한국어 → 영어 번역 함수
# def translate_to_english(korean_text):
#     inputs = ko_en_tokenizer(korean_text, return_tensors="pt", padding=True, truncation=True)
#     outputs = ko_en_model.generate(**inputs)
#     translated_text = ko_en_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"Translated to English: {translated_text}")
#     return translated_text

# # 영어 → 한국어 번역 함수
# def translate_to_korean(english_text):
#     inputs = en_ko_tokenizer(english_text, return_tensors="pt", padding=True, truncation=True)
#     outputs = en_ko_model.generate(**inputs)
#     translated_text = en_ko_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"Translated to Korean: {translated_text}")
#     return translated_text

# # 텍스트가 영어로만 이루어져 있는지 확인하는 함수
# def is_english(text):
#     return bool(re.fullmatch(r'[a-zA-Z\s.,?!\'"]+', text))

# # Back Translation 수행 함수
# def back_translate(text):
#     try:
#         if is_english(text):
#             print(f"Text is only English, skipping: {text}")
#             return text

#         # 1. 한국어 → 영어 번역
#         english_text = translate_to_english(text)

#         # 2. 영어 → 한국어 번역
#         back_translated_text = translate_to_korean(english_text)

#         return back_translated_text

#     except Exception as e:
#         print(f"Back Translation 실패: {text}, 오류: {e}")
#         return ''

# # 기존 텍스트 정제 함수
# def clean_text(text):
#     emojis = ''.join(emoji.EMOJI_DATA.keys())
#     pattern = re.compile(f'[^가-힣 .,?!/@$%~％·∼()\x00-\x7F{emojis}]+')
#     url_pattern = re.compile(
#         r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
#     )

#     text = pattern.sub(' ', text)
#     text = url_pattern.sub('', text)
#     text = text.strip()
#     text = repeat_normalize(text, num_repeats=2)

#     return text if text else ''

# # 데이터 증강 함수
# def augment_with_back_translation(df):
#     augmented_data = []

#     for idx, row in df.iterrows():
#         label = row['label']
#         document = row['document']

#         try:
#             # 한국어 → 영어 번역
#             english_text = translate_to_english(document)

#             # 영어 → 한국어 번역
#             back_translated_text = translate_to_korean(english_text)

#             # 결과 추가
#             augmented_data.append({
#                 'label': label,
#                 'original': document,
#                 'english_translation': english_text,
#                 'back_translation': back_translated_text
#             })

#         except Exception as e:
#             print(f"Back Translation 실패: {document}, 오류: {e}")

#     return pd.DataFrame(augmented_data)

# def convert_to_tsv(input_file, output_file):
#     # 파일 읽기
#     df = pd.read_csv(input_file, delimiter='\t', names=['id', 'document', 'label'], header=0)
#     df = df.drop_duplicates(subset=['document']).dropna(subset=['document', 'label'])

#     # 레이블 변환 (1 -> positive, 0 -> negative)
#     df['label'] = df['label'].apply(lambda x: 'positive' if int(x) == 1 else 'negative')

#     # 필요 없는 'id' 열 제거
#     df = df[['label', 'document']]

#     # 텍스트 정제
#     df['document'] = df['document'].map(clean_text)
#     df = df[df['document'].str.strip() != '']

#     print(f"원본 데이터 크기: {len(df)}")

#     # Back Translation 증강 적용
#     augmented_df = augment_with_back_translation(df)
#     print(f"증강된 데이터 크기: {len(augmented_df)}")

#     # 최종 데이터를 TSV 형식으로 저장 (label, original, english_translation, back_translation 순서)
#     augmented_df.to_csv(output_file, sep='\t', index=False, columns=['label', 'original', 'english_translation', 'back_translation'])
#     print(f"파일이 변환되었습니다: {output_file}")


# # 메인 함수
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Convert ratings data to TSV format with back-translation.")
#     parser.add_argument('--input_file', type=str, help='Path to the input .txt file')
#     parser.add_argument('--output_file', type=str, help='Path to the output .tsv file')

#     args = parser.parse_args()
#     convert_to_tsv(args.input_file, args.output_file)
