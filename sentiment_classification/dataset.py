# import torch
# from torch.utils.data import Dataset


# class TextClassificationCollator:
#     def __init__(self, tokenizer, max_length, with_text=True):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.with_text = with_text

#     def __call__(self, samples):
#         # 튜플 형식의 데이터를 분리
#         texts, labels = zip(*samples)

#         encoding = self.tokenizer(
#             list(texts),
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )

#         return_value = {
#             'input_ids': encoding['input_ids'],
#             'attention_mask': encoding['attention_mask'],
#             'labels': torch.tensor(labels, dtype=torch.long),
#         }
#         if self.with_text:
#             return_value['text'] = list(texts)

#         return return_value


# class TextClassificationDataset(Dataset): # torch.utils.data.Dataset 객체 상속받음

#     def __init__(self, texts, labels):
#         self.texts = texts
#         self.labels = labels
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
#         return text, label # 예상출력 : 튜플형식의 ('text 내용', label)

# # 랜덤 삭제 적용할때만
# import torch
# from torch.utils.data import Dataset
# import random


# class TextClassificationCollator:
#     def __init__(self, tokenizer, max_length, with_text=True):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.with_text = with_text

#     def __call__(self, samples):
#         # 튜플 형식의 데이터를 분리
#         texts, labels = zip(*samples)

#         encoding = self.tokenizer(
#             list(texts),
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )

#         return_value = {
#             'input_ids': encoding['input_ids'],
#             'attention_mask': encoding['attention_mask'],
#             'labels': torch.tensor(labels, dtype=torch.long),
#         }
#         if self.with_text:
#             return_value['text'] = list(texts)

#         return return_value


# class TextClassificationDataset(Dataset):  # torch.utils.data.Dataset 객체 상속받음
#     def __init__(self, texts, labels):
#         """
#         Args:
#             texts (list): 텍스트 리스트
#             labels (list): 라벨 리스트
#         """
#         self.texts = texts
#         self.labels = labels

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
#         random_deletion_prob = random.uniform(0.1, 0.2)  # 매 문장마다 확률 결정
#         augmented_text = self.random_deletion(text, random_deletion_prob)  # Random Deletion 적용
#         return augmented_text, label

#     def random_deletion(self, text, deletion_prob):
#         """
#         랜덤 삭제 기법을 적용하여 텍스트를 수정합니다.
#         Args:
#             text (str): 원본 텍스트
#             deletion_prob (float): 단어 삭제 확률
#         Returns:
#             str: 랜덤 삭제가 적용된 텍스트
#         """
#         words = text.split()
#         if len(words) == 1:  # 단어가 1개일 경우 삭제하지 않음
#             return text

#         remaining_words = [
#             word for word in words if random.random() > deletion_prob
#         ]

#         # 모든 단어가 삭제된 경우 최소 1개 단어를 유지
#         if len(remaining_words) == 0:
#             remaining_words = random.sample(words, 1)

#         return " ".join(remaining_words)


# 문자위치변경 할때만
import random
import torch
from torch.utils.data import Dataset

def shuffle_korean_word(word):
    if len(word) > 3:  # 길이가 3 이하인 단어는 변경하지 않음
        middle = list(word[1:-1])
        random.shuffle(middle)
        return word[0] + ''.join(middle) + word[-1]
    return word

def augment_text(text):
    words = text.split()
    shuffled_text = ' '.join([shuffle_korean_word(word) for word in words])
    return shuffled_text

class TextClassificationCollator:
    def __init__(self, tokenizer, max_length, augment_prob=0.0, with_text=True):
        """
        Args:
            tokenizer: Hugging Face tokenizer.
            max_length: Maximum sequence length for padding/truncation.
            augment_prob: 각 문장에 대해 증강을 적용할 확률 (0.0 ~ 1.0).
            with_text: Whether to include original text in the output.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_prob = augment_prob
        self.with_text = with_text

    def augment_text(self, text):
        """Apply character shuffling augmentation to a single sentence."""
        return augment_text(text)

    def __call__(self, samples):
        texts, labels = zip(*samples)

        # 각 문장에 대해 augment_prob 확률로 증강 적용
        augmented_texts = [
            self.augment_text(text) if random.random() < self.augment_prob else text
            for text in texts
        ]

        encoding = self.tokenizer(
            list(augmented_texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value['text'] = list(augmented_texts)

        return return_value

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, augment=False):
        self.texts = texts
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 데이터 증강 적용
        if self.augment:
            text = augment_text(text)
        
        return text, label


