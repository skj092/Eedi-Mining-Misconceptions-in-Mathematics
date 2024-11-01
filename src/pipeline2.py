# %% [code]
import kagglehub
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from tqdm.autonotebook import trange
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
)
import copy
import warnings
warnings.filterwarnings('ignore')


def apk(actual, predicted, k=25):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=25):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of lists of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def inference(df, model, tokenizer, device):
    batch_size = 16
    max_length = 512
    sentences = list(df['query_text'].values)
    pids = list(df['order_index'].values)
    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True,
                             return_tensors="pt")
        features = batch_to_device(features, device)
        with torch.no_grad():
            outputs = model.model(**features)
            embeddings = last_token_pool(
                outputs.last_hidden_state, features['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            embeddings = embeddings.detach().cpu().numpy().tolist()
        all_embeddings.extend(embeddings)

    all_embeddings = [np.array(all_embeddings[idx]).reshape(
        1, -1) for idx in np.argsort(length_sorted_idx)]

    sentence_embeddings = np.concatenate(all_embeddings, axis=0)
    result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
    return result


# %% [code]

# Download latest version
path = kagglehub.dataset_download("sayoulala/v7-recall")
path_prefix = "/home/sonujha/rnd/Eedi-Mining-Misconceptions-in-Mathematics/data/"
model_path = "/kaggle/input/sfr-embedding-mistral/SFR-Embedding-2_R"
model_path = "Salesforce/SFR-Embedding-2_R"
lora_path = "/kaggle/input/v7-recall/epoch_19_model/adapter.bin"
lora_path = path + '/epoch_19_model/adapter.bin'
device = 'cpu'
VALID = False

# %% [code]
tokenizer = AutoTokenizer.from_pretrained(
    lora_path.replace("/adapter.bin", ""))
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModel.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.float32  # Use full precision for CPU
)

# LoRA config
config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Apply LoRA and load weights
model = get_peft_model(model, config)
d = torch.load(lora_path, map_location=model.device)
model.load_state_dict(d, strict=False)
model = model.eval()
print('model loaded')

# %% [code]
# # 输出模型的参数名和参数值
# for name, param in model.named_parameters():
#     if "base_model.model.layers.12.input_layernorm.weight"  in name:
#         print(f"参数名: {name}")
#         print(f"参数值: {param}")


# %% [code]
task_description = 'Given a math question and a misconcepte incorrect answer, please retrieve the most accurate reason for the misconception.'

# %% [code]
if VALID:
    tra = pd.read_parquet("/kaggle/input/v1-parquet/v1_val.parquet")
    print(tra.shape)
else:
    tra = pd.read_csv(f"{path_prefix}/test.csv")
    print(tra.shape)
misconception_mapping = pd.read_csv(f"{path_prefix}/misconception_mapping.csv")
if tra.shape[0] < 10:
    misconception_mapping = misconception_mapping.sample(
        n=5, random_state=2023)

# %% [code]
if VALID:
    train_data = []
    for _, row in tra.iterrows():
        for c in ['A', 'B', 'C', 'D']:
            if str(row[f"Misconception{c}Id"]) != "nan":
                # print(row[f"Misconception{c}Id"])
                real_answer_id = row['CorrectAnswer']
                real_text = row[f'Answer{real_answer_id}Text']
                query_text = f"###question###:{row['SubjectName']}-{row['ConstructName']}-{row['QuestionText']}\n###Correct Answer###:{real_text}\n###Misconcepte Incorrect answer###:{c}.{row[f'Answer{c}Text']}"
                row['query_text'] = get_detailed_instruct(
                    task_description, query_text)
                row['answer_id'] = row[f"Misconception{c}Id"]
                train_data.append(copy.deepcopy(row))
    train_df = pd.DataFrame(train_data)
    train_df['order_index'] = list(range(len(train_df)))
else:
    train_data = []
    for _, row in tra.iterrows():
        for c in ['A', 'B', 'C', 'D']:
            if c == row['CorrectAnswer']:
                continue
            if f'Answer{c}Text' not in row:
                continue
            real_answer_id = row['CorrectAnswer']
            real_text = row[f'Answer{real_answer_id}Text']
            query_text = f"###question###:{row['SubjectName']}-{row['ConstructName']}-{row['QuestionText']}\n###Correct Answer###:{real_text}\n###Misconcepte Incorrect answer###:{c}.{row[f'Answer{c}Text']}"
            row['query_text'] = get_detailed_instruct(
                task_description, query_text)
            row['answer_name'] = c
            train_data.append(copy.deepcopy(row))
    train_df = pd.DataFrame(train_data)
    train_df['order_index'] = list(range(len(train_df)))
print(train_df.shape)

# %% [code]
print('creating embedding')
train_embeddings = inference(train_df, model, tokenizer, device)
# {'0': x, '1': x2, '2': x3 ....., '10': x10} such that for each x_i, shape(x_i) = (4096, )
# dict with keys are 0-10 and values are array of shape (4096)

# %% [code]
misconception_mapping['query_text'] = misconception_mapping['MisconceptionName']
misconception_mapping['order_index'] = misconception_mapping['MisconceptionId']
print('creating embedding for misconcption')
doc_embeddings = inference(misconception_mapping, model, tokenizer, device)
'''
{2256: array([ 0.02357483, -0.00921631,  0.02268982, ...,  0.00654602,
         0.01154327, -0.008255  ]),
 2186: array([-0.03488159,  0.01288605,  0.03695679, ...,  0.02783203,
         0.00748825, -0.01389313]),
 2336: array([-0.00228882, -0.01669312,  0.00849152, ...,  0.00759125,
         0.00798035, -0.01704407]),
 455: array([-0.01324463,  0.01687622, -0.00965118, ...,  0.00986481,
        -0.03591919, -0.00258636]),
 2546: array([-0.008255  , -0.00894165, -0.01404572, ...,  0.01661682,
        -0.03018188, -0.00188637])}
 '''

# %% [code]
sentence_embeddings = np.concatenate(
    [e.reshape(1, -1) for e in list(doc_embeddings.values())])
index_text_embeddings_index = {index: paper_id for index, paper_id in
                               enumerate(list(doc_embeddings.keys()))}

# %% [code]
print('prediction')
predicts_test = []
for _, row in tqdm(train_df.iterrows()):
    query_id = row['order_index']
    query_em = train_embeddings[query_id].reshape(1, -1)

    cosine_similarity = np.dot(query_em, sentence_embeddings.T).flatten()

    sort_index = np.argsort(-cosine_similarity)[:25]
    pids = [index_text_embeddings_index[index] for index in sort_index]
    predicts_test.append(pids)

# %% [code]
if VALID:
    train_df['recall_ids'] = predicts_test
    print(mapk([[data] for data in train_df['answer_id'].values],
          train_df['recall_ids'].values))
else:
    train_df['MisconceptionId'] = [
        ' '.join(map(str, c)) for c in predicts_test]
    sub = []
    for _, row in train_df.iterrows():
        sub.append(
            {
                "QuestionId_Answer": f"{row['QuestionId']}_{row['answer_name']}",
                "MisconceptionId": row['MisconceptionId']
            }
        )
    submission_df = pd.DataFrame(sub)
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file created successfully!")
