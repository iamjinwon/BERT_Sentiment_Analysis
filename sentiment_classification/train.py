# import argparse 
# '''
# The argparse module makes it easy to write user-friendly command-line interfaces. 
# 파싱한다 : 주어진 데이터를 해석하고 분석하여 원하는 형식 또는 구조로 변환하는 작업
# '''
# import random

# from sklearn.metrics import accuracy_score

# import torch

# from transformers import BertTokenizerFast
# from transformers import BertForSequenceClassification, AlbertForSequenceClassification
# from transformers import Trainer
# from transformers import TrainingArguments

# from dataset import TextClassificationCollator
# from dataset import TextClassificationDataset
# from utils import read_text

# def define_argparser():
#     p = argparse.ArgumentParser() # 명령줄 인자를 정의하고 처리하기 위한 객체 생성

#     p.add_argument('--model_fn', required=True)
#     p.add_argument('--train_fn', required=True)
#     # Recommended model list:
#     # - kykim/bert-kor-base
#     # - kykim/albert-kor-base
#     # - beomi/kcbert-base
#     # - beomi/kcbert-large
#     p.add_argument('--pretrained_model_name', type=str, default='kykim/bert-kor-base')
#     p.add_argument('--use_albert', action='store_true')

#     p.add_argument('--valid_ratio', type=float, default=.2)
#     p.add_argument('--batch_size_per_device', type=int, default=32)
#     p.add_argument('--n_epochs', type=int, default=5)

#     p.add_argument('--warmup_ratio', type=float, default=.2)

#     p.add_argument('--max_length', type=int, default=100)

#     config = p.parse_args()

#     return config


# def get_datasets(fn, valid_ratio=.2):
#      # Get list of labels and list of texts.
#     labels, texts = read_text(fn)

#     unique_labels = list(set(labels))
#     label_to_index = {}
#     index_to_label = {}
#     for i, label in enumerate(unique_labels):
#         label_to_index[label] = i
#         index_to_label[i] = label

#     # Convert label text to integer value.
#     labels = list(map(label_to_index.get, labels)) # labels의 각 요소에 대해 map함수는 label_to_index.get을 적용함.

#     # Shuffle before split into train and validation set.
#     shuffled = list(zip(texts, labels))
#     random.shuffle(shuffled)
#     texts = [e[0] for e in shuffled]
#     labels = [e[1] for e in shuffled]
#     idx = int(len(texts) * (1 - valid_ratio))

#     train_dataset = TextClassificationDataset(texts[:idx], labels[:idx])
#     valid_dataset = TextClassificationDataset(texts[idx:], labels[idx:])

#     return train_dataset, valid_dataset, index_to_label


# def main(config):
#     # Get pretrained tokenizer.
#     tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)
#     # Get datasets and index to label map.
#     train_dataset, valid_dataset, index_to_label = get_datasets(
#         config.train_fn,
#         valid_ratio=config.valid_ratio
#     )
#     print("[DEBUG] Train dataset sample:", train_dataset[0])

#     print(
#         '|train| =', len(train_dataset),
#         '|valid| =', len(valid_dataset),
#     )

#     total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
#     n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
#     n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
#     print(
#         '#total_iters =', n_total_iterations,
#         '#warmup_iters =', n_warmup_steps,
#     )

#     # Get pretrained model with specified softmax layer.
#     model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
#     model = model_loader.from_pretrained(
#         config.pretrained_model_name,
#         num_labels=len(index_to_label)
#     )

#     training_args = TrainingArguments(
#         output_dir='./.checkpoints',
#         num_train_epochs=config.n_epochs,
#         per_device_train_batch_size=config.batch_size_per_device,
#         per_device_eval_batch_size=config.batch_size_per_device,
#         warmup_steps=n_warmup_steps,
#         weight_decay=0.01,
#         fp16=True,
#         eval_strategy='epoch',
#         save_strategy='epoch',
#         logging_steps=n_total_iterations // 100,
#         save_steps=n_total_iterations // config.n_epochs,
#         load_best_model_at_end=True,
#     )

#     def compute_metrics(pred):
#         labels = pred.label_ids
#         preds = pred.predictions.argmax(-1)

#         return {
#             'accuracy': accuracy_score(labels, preds)
#         }
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=TextClassificationCollator(tokenizer,
#                                        config.max_length,
#                                        with_text=False),
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()

#     torch.save({
#         'rnn': None,
#         'cnn': None,
#         'bert': trainer.model.state_dict(),
#         'config': config,
#         'vocab': None,
#         'classes': index_to_label,
#         'tokenizer': tokenizer,
#     }, config.model_fn)

# if __name__ == '__main__':
#     config = define_argparser()
#     main(config)


# # 랜덤삭제 사용할때만
# import argparse
# import random
# from sklearn.metrics import accuracy_score
# import torch
# from transformers import BertTokenizerFast
# from transformers import BertForSequenceClassification, AlbertForSequenceClassification
# from transformers import Trainer
# from transformers import TrainingArguments
# from dataset import TextClassificationCollator, TextClassificationDataset
# from utils import read_text


# def define_argparser():
#     p = argparse.ArgumentParser()

#     p.add_argument('--model_fn', required=True)
#     p.add_argument('--train_fn', required=True)
#     p.add_argument('--pretrained_model_name', type=str, default='kykim/bert-kor-base')
#     p.add_argument('--use_albert', action='store_true')

#     p.add_argument('--valid_ratio', type=float, default=0.2)
#     p.add_argument('--batch_size_per_device', type=int, default=32)
#     p.add_argument('--n_epochs', type=int, default=5)

#     p.add_argument('--warmup_ratio', type=float, default=0.2)
#     p.add_argument('--max_length', type=int, default=100)
#     p.add_argument('--random_deletion_prob', type=float, default=0.15, help='랜덤 삭제 확률')

#     config = p.parse_args()

#     return config


# def get_datasets(fn, valid_ratio=0.2):
#     labels, texts = read_text(fn)

#     unique_labels = list(set(labels))
#     label_to_index = {label: i for i, label in enumerate(unique_labels)}
#     index_to_label = {i: label for i, label in enumerate(unique_labels)}

#     # 라벨을 정수로 변환
#     labels = list(map(label_to_index.get, labels))

#     # 데이터 섞기
#     shuffled = list(zip(texts, labels))
#     random.shuffle(shuffled)
#     texts, labels = zip(*shuffled)

#     idx = int(len(texts) * (1 - valid_ratio))

#     # random_deletion_prob을 매 문장마다 내부에서 처리하도록 수정
#     train_dataset = TextClassificationDataset(texts[:idx], labels[:idx])
#     valid_dataset = TextClassificationDataset(texts[idx:], labels[idx:])  # Validation 데이터에는 Random Deletion을 적용하지 않음

#     return train_dataset, valid_dataset, index_to_label

# def main(config):
#     tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)

#     # random_deletion_prob 제거
#     train_dataset, valid_dataset, index_to_label = get_datasets(
#         config.train_fn,
#         valid_ratio=config.valid_ratio
#     )

#     print("[DEBUG] Train dataset sample:", train_dataset[0])
#     print('|train| =', len(train_dataset), '|valid| =', len(valid_dataset))

#     total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
#     n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
#     n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

#     print('#total_iters =', n_total_iterations, '#warmup_iters =', n_warmup_steps)

#     model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
#     model = model_loader.from_pretrained(
#         config.pretrained_model_name,
#         num_labels=len(index_to_label)
#     )

#     training_args = TrainingArguments(
#         output_dir='./.checkpoints',
#         num_train_epochs=config.n_epochs,
#         per_device_train_batch_size=config.batch_size_per_device,
#         per_device_eval_batch_size=config.batch_size_per_device,
#         warmup_steps=n_warmup_steps,
#         weight_decay=0.01,
#         fp16=True,
#         eval_strategy='epoch',
#         save_strategy='epoch',
#         logging_steps=n_total_iterations // 100,
#         save_steps=n_total_iterations // config.n_epochs,
#         load_best_model_at_end=True,
#     )

#     def compute_metrics(pred):
#         labels = pred.label_ids
#         preds = pred.predictions.argmax(-1)
#         return {'accuracy': accuracy_score(labels, preds)}

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=TextClassificationCollator(tokenizer, config.max_length, with_text=False),
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()

#     torch.save({
#         'bert': trainer.model.state_dict(),
#         'config': config,
#         'classes': index_to_label,
#         'tokenizer': tokenizer,
#     }, config.model_fn)


# if __name__ == '__main__':
#     config = define_argparser()
#     main(config)


# 문자위치변경할때만
import argparse
import random
from sklearn.metrics import accuracy_score
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, AlbertForSequenceClassification
from transformers import Trainer, TrainingArguments
from dataset import TextClassificationCollator, TextClassificationDataset
from utils import read_text

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--pretrained_model_name', type=str, default='kykim/bert-kor-base')
    p.add_argument('--use_albert', action='store_true')
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=100)

    return p.parse_args()

def get_datasets(fn, valid_ratio=.2, augment=False):
    labels, texts = read_text(fn)

    unique_labels = list(set(labels))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    index_to_label = {i: label for i, label in enumerate(unique_labels)}

    labels = list(map(label_to_index.get, labels))

    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    train_dataset = TextClassificationDataset(
        texts[:idx], labels[:idx], augment=augment  # 학습 데이터에만 증강 활성화
    )
    valid_dataset = TextClassificationDataset(
        texts[idx:], labels[idx:], augment=False  # 검증 데이터는 원본 사용
    )

    return train_dataset, valid_dataset, index_to_label

def main(config):
    # Tokenizer 로드
    tokenizer = BertTokenizerFast.from_pretrained(config.pretrained_model_name)

    # 데이터셋 준비
    train_dataset, valid_dataset, index_to_label = get_datasets(
        config.train_fn,
        valid_ratio=config.valid_ratio,
        augment=False  # 증강은 데이터 로더에서 적용
    )

    print("[DEBUG] Train dataset sample:", train_dataset[0])
    print('|train| =', len(train_dataset), '|valid| =', len(valid_dataset))

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print('#total_iters =', n_total_iterations, '#warmup_iters =', n_warmup_steps)

    # 모델 준비
    model_loader = AlbertForSequenceClassification if config.use_albert else BertForSequenceClassification
    model = model_loader.from_pretrained(config.pretrained_model_name, num_labels=len(index_to_label))

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir='./.checkpoints',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
    )

    # 증강 확률 설정 (예: 30% 확률로 증강 적용)
    augment_prob = 0.2
    data_collator = TextClassificationCollator(
        tokenizer=tokenizer,
        max_length=config.max_length,
        augment_prob=augment_prob,  # 증강 확률 추가
        with_text=False
    )

    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=lambda pred: {
            'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1))
        },
    )

    # 학습 시작
    trainer.train()

    # 모델 저장
    torch.save({
        'bert': trainer.model.state_dict(),
        'config': config,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
