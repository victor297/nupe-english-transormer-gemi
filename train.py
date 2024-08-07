from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
from datasets import Dataset

# Load your dataset
data = pd.read_csv('./nupe.csv')
data = data.rename(columns={'nupe': 'translation', 'eng': 'target'})
dataset = Dataset.from_pandas(data)

# Load the tokenizer and model
model_name = 'Helsinki-NLP/opus-mt-en-de'  # Use a similar model for fine-tuning
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Preprocess the dataset
def preprocess_function(examples):
    inputs = [str(input_text) for input_text in examples['translation']]
    targets = [str(target_text) for target_text in examples['target']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    logging_dir='./logs',
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./nupe-to-english-model')
tokenizer.save_pretrained('./nupe-to-english-model')
