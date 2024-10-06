# Eedi-Mining-Misconceptions-in-Mathematics

# About:
- For every question there are one correct answer and 3 incorrect answer also known as distractors.
- Eash distractors is designed to correspond with a potential `misconception`.

# Goal:
- To predict the affinity between `misconception` and incorrect answers (distractors) in the MCQ.
- For each question and answer we have to predict the all the `misconception (reason)` which can be up to `25`.

# Data

## train.csv

- QuestionId: Unique Question idenfitier (int) - 1869
- ConstructId: Unique construct idnetifier (int) - 757
- ConstructName: Most granular level of knowledge related to question (str) - 757
- SubjectId: Unique SubjectId idnetifier (int) - 163
- SubjectName: More general context thant the construct (str) - 163
- CorrectAnswer: A, B, C or D (char) - 4
- QuestionText: Question text extracted from the question image using human-in-loop OCR (str) - 1869
- AnswerAText: Answer Option A Text extracted form the question image using human-in-theloop OCR (str)
- AnswerBText: Answer Option A Text extracted form the question image using human-in-theloop OCR (str)
- AnswerCText: Answer Option A Text extracted form the question image using human-in-theloop OCR (str)
- AnswerDText: Answer Option A Text extracted form the question image using human-in-theloop OCR (str)
- MisconceptionAId: Unique misconception identifier (int). Ground Truth labels in `train.csv`; your task is to predict these labels for `test.csv`.
- MisconceptionBId: Unique misconception identifier (int). Ground Truth labels in `train.csv`; your task is to predict these labels for `test.csv`.
- MisconceptionCId: Unique misconception identifier (int). Ground Truth labels in `train.csv`; your task is to predict these labels for `test.csv`.
- MisconceptionDId: Unique misconception identifier (int). Ground Truth labels in `train.csv`; your task is to predict these labels for `test.csv`.

## misconception_mapping.csv (2887, 2)
- MisconceptionId: 2887 int values
- MisconceptionName: Misconception name (str)

## test.csv (3, 11)
- All the column are same except no MisconceptionId columns therefore num columes are 15-5 = 11

## submission.csv
- For each `QuestionId-Answer` row in the test set, you must predict the corresponding `MisconceptionId`.
- You can predict up to 24 `MisconceptionId` values per row and these should be space-decimited.
```csv
QuestionId_Answer,MisconceptionId
1869_B,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1869_C,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
1869_D,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
```
