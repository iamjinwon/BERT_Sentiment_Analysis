def read_text(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            if line.strip() != '':
                # The file should have tab delimited two columns.
                # First column indicates label field,
                # and second column indicates text field.
                label, text = line.strip().split('\t') # 문자열 양쪽 끝의 공백 문자(스페이스, 탭, 줄바꿈)제거하고, '\t'기준으로 나눔
                labels += [label] # positive, negative
                texts += [text]

    return labels, texts

# fn = '/home/jinwon/workspace/5-plm/nsmc/train.tsv'
# print(read_text(fn))

# 디버깅 코드 
# def read_text(file_path):
#     labels, texts = [], []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line_num, line in enumerate(f, start=1):
#             print(f"[DEBUG] Line {line_num}: {line.strip()}")  # 디버그 출력 추가
#             parts = line.strip().split('\t')
#             if len(parts) == 2:
#                 label, text = parts
#                 labels.append(label)
#                 texts.append(text)
#             else:
#                 print(f"[WARNING] Malformed or empty line at {line_num}: {line.strip()}")
#                 break
#     return labels, texts

def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm
