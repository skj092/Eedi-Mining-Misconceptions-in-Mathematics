import random
import os
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformer, losses
import code
from datasets import Dataset, load_dataset
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm import tqdm

mnli = load_dataset('glue', 'mnli', split='train').select(range(100))
train_dataset = mnli.remove_columns('idx')
os.environ["WANDB_DISABLED"] = "true"


'''
>>> train_dataset
Dataset({
    features: ['premise', 'hypothesis', 'label'],
    num_rows: 1000
})
'''

'''
>>> train_dataset[0]

{
        'premise': 'Conceptually cream skimming has two basic dimensions - product and geography.',
        'hypothesis': 'Product and geography are what make cream skimming work. ',
        'label': 1
}

'''
# Select only similar pair
mnli = mnli.filter(
    lambda x: True if x['label'] == 0 else False)

train_dataset = {"anchor": [], "positive": [], "negative": []}
soft_negative = mnli['hypothesis']
random.shuffle(soft_negative)
for row, soft_negative in tqdm(zip(mnli, soft_negative)):
    train_dataset['anchor'].append(row['premise'])
    train_dataset['positive'].append(row['hypothesis'])
    train_dataset['negative'].append(soft_negative)
train_dataset = Dataset.from_dict(train_dataset)
print(f"lenght of train dataset {len(train_dataset)}")


embedding_model = SentenceTransformer('bert-base-uncased')

# MultipleNegativesRankingLoss
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)


# Embedding similarity Evaluater
val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(sentences1=val_sts['sentence1'], sentences2=val_sts['sentence2'], scores=[
    score/5 for score in val_sts['label']], main_similarity='cosine',)

args = SentenceTransformerTrainingArguments(
    output_dir='cosineloss_embdding_model',
    num_train_epochs=1,
    per_gpu_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    report_to=None
)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator,
)
trainer.train()
print(evaluator(embedding_model))
'''
{
        'train_runtime': 313.1354,
        'train_samples_per_second': 3.194,
        'train_steps_per_second': 0.102,
        'train_loss': 1.1037036180496216,
        'epoch': 1.0
}
100%|█████████████████████████████████████████████████████████| 32/32 [05:11<00:00,  9.73s/it]
{
        'pearson_cosine': np.float64(0.5932199552077273),
        'spearman_cosine': np.float64(0.5938938653685093),
        'pearson_manhattan': np.float64(0.6085995032600656),
        'spearman_manhattan': np.float64(0.6140262061183868),
        'pearson_euclidean': np.float64(0.6055484673561824),
        'spearman_euclidean': np.float64(0.6114901100644043),
        'pearson_dot': np.float64(0.37494954679629566),
        'spearman_dot': np.float64(0.372246890012623),
        'pearson_max': np.float64(0.6085995032600656),
        'spearman_max': np.float64(0.6140262061183868)}
'''


code.interact(local=locals())
