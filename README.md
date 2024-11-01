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
- QuestionId:
- AllQuestionText:
- CorrectAnswer: A/B/C/D
- AnswerType: AnswerAText/AnswerBText/AnswerCText/AnswerDText
- AnswerText:
- AnswerAlphabet: A/B/C/D
- MisconceptionId:
- MisconceptionAlphabet: A/B/C/D

---------------------------

## [bge model] (#)
1. load train, test and misconception csv data
2. data preprocessing train.csv
    1. train.csv (1869, 16) -> train.csv (7476, 7) # (wide to long)
    2. Add new column "AllText" which is concat of 1. ConstructName, 2. SubjectName, 3. QuestionText 4. AnswerText # (7476, 8)
    3. Add new columnd "AnswerAlphabet" by extracting the alphabet from AnswerText # (7476, 9)
    4. New column 'QuestionId_Answer' which is concat of QuestionId and the AnswerAlphabet # (7476, 10)
3. data preprocessing misconcpetion.csv
    1. misconcpetion.csv (1869, 16) -> train.csv (7476, 7) # (wide to long)
    2. Add new columnd "AnswerAlphabet" by extracting the alphabet from MisconcpetionId # (7476, 9)
    3. New column 'QuestionId_Answer' which is concat of QuestionId and the AnswerAlphabet # (7476, 10)
    4. Sort by and select only two column "QuestionId_Answer" and "MisconceptionId"

4. concat misconcpetion.csv to train.csv (7476, 10)
5. load 'bge_large_en' embedding model using SentenceTransformer
6. create embedding of train_long['AllText'] column array (7476, 1024)
7. create embedding of misconcpetion_mapping['MisconceptionName'] column array (2887, 1024)
8. Find cosine difference between both of the embeddng matrices - (7476, 2887)

9. Get the sorted index of y in the cosine difference
10. Select only 25 y indexes and convert this data into pandas series and add this columns into train_long dataframe
11. filter out where misconceptionId is null from the train_long data
12. Explode on predictmisconceptionid to convert array of index into a single value
13. ['QuestionId', 'ConstructName', 'SubjectName', 'QuestionText', 'CorrectAnswer', 'AnswerType', 'AnswerText', 'AllText', 'AnswerAlphabet', 'QuestionId_Answer', 'MisconceptionId', 'PredictMisconceptionId', 'MisconceptionName', 'PredictMisconceptionName']

##  [SFR-Embedding-2_R](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519)
1. Set Directory paths
2. Load LoRA model
3. Load the test dataframe (3, 11)
4. Preprocess the test dataframe (9, 14)
    - Add column `query_text` which is the concat of some other columns
    - Add column `answer_name` which is either one of 'A-D'
    - Add column `order_index` which is index of each row.
5. Get embedding of test dataframe
    - A dictionary with keys are `order_index` and values are the embedding of correspond `query_text` of shape (4096,)
6. Load the `misconcption.csv` file (5, 2)
7. Add two column `query_text` and `order_index` and get the embeddings.
8. Get embedding of `misconception`
    - keys are `misconceptionId` and values are vector of shape (4096,)
9. Reshape the `misconcption` embedding of shape (5, 4096)
10. For each row in `test_df`:
    - Get the query id, and use it to get the corresponding embedding
    - Find `cosine similarity` with the `misconcption embedding`
    - Take only 25 and get the index of all these `misconcption`

