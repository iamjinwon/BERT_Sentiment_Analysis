import argparse

def check_tsv_format(file_path):
    """
    주어진 TSV 파일에서 '\t'로 구분되지 않은 줄을 확인합니다.
    Args:
        file_path (str): 체크할 TSV 파일 경로.
    """
    malformed_lines = []  # 잘못된 줄을 저장할 리스트

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            # '\t'로 제대로 구분되지 않은 줄 확인
            if line.count('\t') != 1:  # '\t'가 정확히 한 번 있어야 함
                malformed_lines.append((line_num, line.strip()))

    if malformed_lines:
        for line_num, line in malformed_lines:
            print(f"[ERROR] Line {line_num} is malformed: {line}")
    else:
        print("모든 줄이 올바릅니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check TSV file format for incorrect lines.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the TSV file to check')
    args = parser.parse_args()
    
    check_tsv_format(args.file_path)
