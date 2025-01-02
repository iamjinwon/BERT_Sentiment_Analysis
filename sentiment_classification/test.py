# import sys
# import argparse

# import torch
# import torch.nn.functional as F

# from transformers import BertTokenizerFast
# from transformers import BertForSequenceClassification, AlbertForSequenceClassification
# from sklearn.metrics import accuracy_score


# def define_argparser():
#     '''
#     Define argument parser to take inference using pre-trained model.
#     '''
#     p = argparse.ArgumentParser()

#     p.add_argument('--model_fn', required=True)
#     p.add_argument('--test_fn', required=True)  # 추가: 테스트 데이터 파일 경로
#     p.add_argument('--gpu_id', type=int, default=-1)
#     p.add_argument('--batch_size', type=int, default=256)
#     p.add_argument('--top_k', type=int, default=1)

#     config = p.parse_args()

#     return config


# def read_test_data(test_fn):
#     '''
#     Read test data and labels from the file.
#     '''
#     lines = []
#     labels = []

#     with open(test_fn, 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip() != '':
#                 label, text = line.strip().split('\t')
#                 lines.append(text)
#                 labels.append(label)

#     return lines, labels


# def main(config):
#     saved_data = torch.load(
#         config.model_fn,
#         map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
#     )

#     train_config = saved_data['config']
#     bert_best = saved_data['bert']
#     index_to_label = saved_data['classes']
#     label_to_index = {v: k for k, v in index_to_label.items()}  # Convert labels to indices

#     # Load test data
#     texts, true_labels = read_test_data(config.test_fn)
#     true_labels = [label_to_index[label] for label in true_labels]  # Convert labels to indices

#     with torch.no_grad():
#         # Declare model and load pre-trained weights.
#         tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)
#         model_loader = AlbertForSequenceClassification if train_config.use_albert else BertForSequenceClassification
#         model = model_loader.from_pretrained(
#             train_config.pretrained_model_name,
#             num_labels=len(index_to_label)
#         )
#         model.load_state_dict(bert_best)

#         if config.gpu_id >= 0:
#             model.cuda(config.gpu_id)
#         device = next(model.parameters()).device

#         # Don't forget to turn-on evaluation mode.
#         model.eval()

#         y_preds = []
#         for idx in range(0, len(texts), config.batch_size):
#             mini_batch = tokenizer(
#                 texts[idx:idx + config.batch_size],
#                 padding=True,
#                 truncation=True,
#                 return_tensors="pt",
#             )

#             x = mini_batch['input_ids'].to(device)
#             mask = mini_batch['attention_mask'].to(device)

#             # Take feed-forward
#             y_hat = model(x, attention_mask=mask).logits.argmax(dim=-1)
#             y_preds.extend(y_hat.cpu().tolist())

#         # Calculate accuracy
#         accuracy = accuracy_score(true_labels, y_preds)

#         print(f"Test Accuracy: {accuracy:.4f}")

#         # Optional: Print predictions
#         for i in range(10):
#             predicted_label = index_to_label[y_preds[i]]
#             print(f"Prediction: {predicted_label}\tText: {texts[i]}")


# if __name__ == '__main__':
#     config = define_argparser()
#     main(config)

import sys
import argparse
import os
import pandas as pd

import torch
import torch.nn.functional as F

from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AlbertForSequenceClassification
from sklearn.metrics import accuracy_score


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--test_fn', required=True)  # 테스트 데이터 파일 경로
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config


def read_test_data(test_fn):
    '''
    Read test data and labels from the file.
    '''
    lines = []
    labels = []

    with open(test_fn, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() != '':
                label, text = line.strip().split('\t')
                lines.append(text)
                labels.append(label)

    return lines, labels


def save_results_to_csv(results, save_dir='./results/'):
    '''
    Save results to a CSV file in the specified directory.
    Automatically avoids overwriting by incrementing the filename.
    '''
    os.makedirs(save_dir, exist_ok=True)
    file_index = 1
    save_path = os.path.join(save_dir, f'results{file_index}.csv')
    
    while os.path.exists(save_path):
        file_index += 1
        save_path = os.path.join(save_dir, f'results{file_index}.csv')

    df = pd.DataFrame(results, columns=['True Label', 'Predicted Label', 'Text'])
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f'Results saved to {save_path}')


def calculate_label_accuracy(true_labels, pred_labels, label_to_index, target_label):
    '''
    Calculate accuracy for a specific label.
    '''
    target_index = label_to_index[target_label]
    indices = [i for i, label in enumerate(true_labels) if label == target_index]

    if not indices:
        return 0.0  # Avoid division by zero

    true_subset = [true_labels[i] for i in indices]
    pred_subset = [pred_labels[i] for i in indices]

    return accuracy_score(true_subset, pred_subset)


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']
    label_to_index = {v: k for k, v in index_to_label.items()}  # Convert labels to indices

    # Load test data
    texts, true_labels = read_test_data(config.test_fn)
    true_labels = [label_to_index[label] for label in true_labels]  # Convert labels to indices

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = BertTokenizerFast.from_pretrained(train_config.pretrained_model_name)
        model_loader = AlbertForSequenceClassification if train_config.use_albert else BertForSequenceClassification
        model = model_loader.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        # Don't forget to turn-on evaluation mode.
        model.eval()

        y_preds = []
        results = []  # Store all results for CSV
        for idx in range(0, len(texts), config.batch_size):
            mini_batch = tokenizer(
                texts[idx:idx + config.batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            x = mini_batch['input_ids'].to(device)
            mask = mini_batch['attention_mask'].to(device)

            # Take feed-forward
            y_hat = model(x, attention_mask=mask).logits.argmax(dim=-1)
            y_preds.extend(y_hat.cpu().tolist())

        # Calculate overall accuracy
        overall_accuracy = accuracy_score(true_labels, y_preds)
        print(f"Overall Test Accuracy: {overall_accuracy:.4f}")

        # Calculate accuracy for 'positive' and 'negative' labels
        positive_accuracy = calculate_label_accuracy(true_labels, y_preds, label_to_index, 'positive')
        negative_accuracy = calculate_label_accuracy(true_labels, y_preds, label_to_index, 'negative')
        print(f"Accuracy for 'positive' label: {positive_accuracy:.4f}")
        print(f"Accuracy for 'negative' label: {negative_accuracy:.4f}")

        # Prepare results for CSV
        for text, true_label_idx, pred_label_idx in zip(texts, true_labels, y_preds):
            results.append([
                index_to_label[true_label_idx],
                index_to_label[pred_label_idx],
                text
            ])

        # Save results to CSV
        save_results_to_csv(results)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
