import evaluate
import collections
from tqdm.auto import tqdm
import numpy as np
from train import trainer, valid_dataset, vn_qadts

metrics = evaluate.load('squad')

n_best = 20,
max_answer_length = 30

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature['example_id']].append(idx)
    
    predicted_answer = []
    for example in tqdm(examples):
        example_id = example['id']
        context = example['context']
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]['offset_mapping']
            
            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    answer = {'text': context[offsets[start_index][0] : offsets[end_index][1]],
                              'logit_score': start_logit[start_index] + end_logit[end_index]}

                    answers.append(answer)
        
        if len(answers) > 0:
            best_answer = max(answers, key = lambda x: x['logit_score'])
            predicted_answer.append({'id': example_id,
                                     'prediction_text': best_answer['text']})
        
        else:
            predicted_answer.append({'id': example_id, 'prediction_text': ""})
    
    theoretical_answers = [{'id': ex['id'], 'answer': ex['answer']} for ex in examples]
    return metrics.compute(predictions = predicted_answer, references = theoretical_answers), predicted_answer, theoretical_answers

predictions, _, _ = trainer.predict(valid_dataset)
start_logits, end_logits = predictions

metrics, predicted_answer, theoretical_answers = compute_metrics(start_logits, end_logits, valid_dataset, vn_qadts['validation'])
print('Metrics: ', metrics)

# trainer.push_to_hub()

# print('Question: ', vn_qadts['validation][0]['questions])
# print('Theoretical answer: ', theoretical_answer[0]['answer'])
# print('Predicted answer: ', predicted_answer[0]['prediction_text'])

# print('Question: ', vn_qadts['validation][1]['questions])
# print('Theoretical answer: ', theoretical_answer[1]['answer'])
# print('Predicted answer: ', predicted_answer[1]['prediction_text'])

# print('Question: ', vn_qadts['validation][2]['questions])
# print('Theoretical answer: ', theoretical_answer[2]['answer'])
# print('Predicted answer: ', predicted_answer[2]['prediction_text'])