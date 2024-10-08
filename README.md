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
# Solution Ideas:

## [Pipeline 1](https://www.kaggle.com/code/sinchir0/baseline-tfidf-cos-sim)

1. import libraries
2. data loading: train, test, misconception
3. new column in train and test: all_question_text = construct_name + subjectname + question_text
4. wide_to_long: train, test, misconcpetion
5. new columns in train and test: AnswerAlphabset = extract answer index from answerType
6. new column in misconcpetion: MisconcpetionAlphabet = extract misconcption letter from misconcpetiontype
7. merge data: train/test + misconception
8. new column in trai and test: alltext = allquestiontext + answer text
9. short the dataframe
10. tfidv for the column : Alltext and Misconcpetion Name

-----------------------

Step4:
- train with shape (1869, 16) -> Wide to long -> Train with shape (7476, 5)

1. QuestionId: 1869 unique id
2. Allquestiontext: 1857 unique values
3. CorrectAnswer: 4 unique letters (A, B, C, D)
4. AnswerType: created by pd.mel (AnswerAText, AnswerBText, AnswerCText, AnswerDText)
5. AnswerText: create by pd.mel (4132 unique values)

- train with shape (1889, 16) -, Wide to long for missconcpetion -> train miss shape (7476, 5)

1. QuestionId: 1869 unique id
2. Allquestiontext: 1857 unique values
3. CorrectAnswer: 4 unique letters (A, B, C, D)
4. MisconceptionType: created by pd.mel ('MisconceptionAId', 'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId')
5. MisconceptionId: create by pd.mel (1606 unique values)

Step7:
'QuestionId',
'AllQuestionText',
'CorrectAnswer',
'AnswerType',
'AnswerText',
'AnswerAlphabet',
'MisconceptionId',
'MisconceptionAlphabet'
