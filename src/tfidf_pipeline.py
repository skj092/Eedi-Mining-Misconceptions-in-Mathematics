import pandas as pd
import numpy as np
from typing import Literal
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data Loading
data_dir = Path(
    "/home/sonujha/rnd/Eedi-Mining-Misconceptions-in-Mathematics/data/")

train = pd.read_csv(data_dir/'train.csv')
test = pd.read_csv(data_dir/'test.csv')
misconception_mapping = pd.read_csv(data_dir/'misconception_mapping.csv')
sample_submission = pd.read_csv(data_dir/'sample_submission.csv')


# Data Preprocessing
def make_all_question_text(df: pd.DataFrame) -> pd.DataFrame:
    df['AllQuestionText'] = df['ConstructName'] + " " + \
        df['SubjectName'] + " " + df['QuestionText']
    return df


def wide_to_long(df: pd.DataFrame, col: Literal['AnswerText', 'MisconceptionId']) -> pd.DataFrame:
    if col == "AnswerText":
        add_col = [f"Answer{alpha}Text" for alpha in ['A', 'B', 'C', 'D']]
        var_name = "AnswerType"

    elif col == "MisconceptionId":
        add_col = [f"Misconception{alpha}Id" for alpha in ['A', 'B', 'C', 'D']]
        var_name = "MisconceptionType"
    else:
        raise Exception

    return pd.melt(
        df[["QuestionId", "AllQuestionText", "CorrectAnswer"] + add_col],
        id_vars=["QuestionId", "AllQuestionText", "CorrectAnswer"],
        var_name=var_name,
        value_name=col
    )


train = make_all_question_text(train)
test = make_all_question_text(test)

train_long = wide_to_long(train, col="AnswerText")  # (id_vars, var_name, col)
test_long = wide_to_long(test, col="AnswerText")
train_long_miss = wide_to_long(train, col="MisconceptionId")


train_long["AnswerAlphabet"] = train_long["AnswerType"].str.extract(
    r'Answer([A-Z])Text$')
test_long["AnswerAlphabet"] = test_long["AnswerType"].str.extract(
    r'Answer([A-Z])Text$')
train_long_miss["MisconceptionAlphabet"] = train_long_miss["MisconceptionType"].str.extract(
    r'Misconception([A-Z])Id$')

train_long = pd.merge(
    train_long,
    train_long_miss[["QuestionId",
                     "MisconceptionId", "MisconceptionAlphabet"]],
    left_on=["QuestionId", "AnswerAlphabet"],
    right_on=["QuestionId", "MisconceptionAlphabet"]
)


def make_all_text(df: pd.DataFrame) -> pd.DataFrame:
    df['AllText'] = df['AllQuestionText'] + ' ' + df['AnswerText']
    return df


train_long = make_all_text(train_long)
test_long = make_all_text(test_long)

train_long = train_long.sort_values(
    ['QuestionId', 'AnswerType']).reset_index(drop=True)
test_long = test_long.sort_values(
    ['QuestionId', 'AnswerType']).reset_index(drop=True)


# Train TFIDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(
    pd.concat([train_long['AllText'], misconception_mapping['MisconceptionName']]))

train_long_vec = tfidf_matrix.toarray()[:len(train_long)]
misconception_mapping_vec = tfidf_matrix.toarray()[len(train_long):]

train_cos_sim_array = cosine_similarity(
    train_long_vec, misconception_mapping_vec)
train_sorted_indices = np.argsort(-train_cos_sim_array, axis=1)


def print_example(df: pd.DataFrame, sorted_indices: np.ndarray, idx: int) -> None:
    print(f"query idx {idx}")
    print(df['AllText'][idx])
    print("\n Cos sim no. 1")
    print(misconception_mapping["MisconceptionName"][sorted_indices[idx, 0]])
    print("\n Cos sim no. 2")
    print(misconception_mapping["MisconceptionName"][sorted_indices[idx, 1]])


#print_example(train_long, train_sorted_indices, 0)


# =========Evaluate================
train_long["PredictMisconceptionId"] = train_sorted_indices[:, :25].tolist()


def map_at_25(predictions, labels):
    map_sum = 0
    for x, y, in zip(predictions, labels):
        z = [i/i if y == j else 0 for i, j in zip(range(1, 26), x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)


print(map_at_25(
    train_long["PredictMisconceptionId"][train_long["MisconceptionId"].notnull()],
    train_long["MisconceptionId"][train_long["MisconceptionId"].notnull()],
))


def recall(prediction, labels):
    acc_num = np.sum([1 for x, y in zip(prediction, labels) if y in x])
    return acc_num / len(prediction)


print(recall(
    train_long["PredictMisconceptionId"][train_long["MisconceptionId"].notnull()],
    train_long["MisconceptionId"][train_long["MisconceptionId"].notnull()],
))

# =========Predict================
test_long_vec = vectorizer.transform(test_long["AllText"])
test_cos_sim_array = cosine_similarity(test_long_vec, misconception_mapping_vec)
test_sorted_indices = np.argsort(-test_cos_sim_array, axis=1)

# =========submission================
test_long["QuestionId_Answer"] = test_long['QuestionId'].astype('str') + "_" + test_long['AnswerAlphabet']
test_long["MisconceptionId"] = test_sorted_indices[:, :25].tolist()
test_long["MisconceptionId"] = test_long["MisconceptionId"].apply(lambda x: ' '.join(map(str, x)))
test_long = test_long[test_long["CorrectAnswer"] != test_long["AnswerAlphabet"]]
submission = test_long[["QuestionId_Answer", "MisconceptionId"]].reset_index(drop=True)
print(submission.head())
