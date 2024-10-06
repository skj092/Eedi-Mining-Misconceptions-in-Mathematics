import code
import pandas as pd
from pathlib import Path

data_dir = Path(
    "/home/sonujha/rnd/Eedi-Mining-Misconceptions-in-Mathematics/data/")


df = pd.read_csv(data_dir/'train.csv')  # (1869, 15)

'''
QuestionId: Unique Question idenfitier (int) - 1869
ConstructId: Unique construct idnetifier (int) - 757
ConstructName: Most granular level of knowledge related to question (str) - 757
SubjectId: Unique SubjectId idnetifier (int) - 163
SubjectName: More general context thant the construct (str) - 163
CorrectAnswer: A, B, C or D (char) - 4
QuestionText: Question text extracted from the question image using human-in-loop OCR (str) - 1869
AnswerAText: Answer Option A Text extracted form the question image using human-in-theloop OCR (str)
AnswerBText: Answer Option A Text extracted form the question image using human-in-theloop OCR (str)
AnswerCText: Answer Option A Text extracted form the question image using human-in-theloop OCR (str)
AnswerDText: Answer Option A Text extracted form the question image using human-in-theloop OCR (str)
MisconceptionAId: Unique misconception identifier (int). Ground Truth labels in `train.csv`; your task is to predict these labels for `test.csv`.
MisconceptionBId: Unique misconception identifier (int). Ground Truth labels in `train.csv`; your task is to predict these labels for `test.csv`.
MisconceptionCId: Unique misconception identifier (int). Ground Truth labels in `train.csv`; your task is to predict these labels for `test.csv`.
MisconceptionDId: Unique misconception identifier (int). Ground Truth labels in `train.csv`; your task is to predict these labels for `test.csv`.
'''

'''
hidded test has approximately 1000 questions.
'''
code.interact(local=locals())
