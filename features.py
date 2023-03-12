class Features:
    def __init__(self, tokenizer, max_length, stride):
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = tokenizer

    def train_processing(self, dataset):
        questions = [q.strip() for q in dataset['question']]
        inputs = self.tokenizer(questions, dataset['context'],
                                max_length = self.max_length,
                                truncation = 'only_second',
                                stride = self.stride,
                                return_overflowing_tokens = True,
                                reutrn_offsets_mapping = True,
                                padding = 'max_length')
        # Start char and end char of each token      
        offset_mapping = inputs.pop('offset_mapping')
        sample_map = inputs.pop('overflow_to_sample_mapping')
        answers = dataset['answer']
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])
            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            # Index of the start token of the context 
            context_start = idx

            while sequence_ids[idx] == 1:
                idx += 1
            # Index of the end token of the context    
            context_end = idx - 1

            # Create label
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs['start_positions'] = start_positions
        inputs['end_positions'] = end_positions

        return inputs

    def valid_processing(self, dataset):
        questions = [q.strip() for q in dataset['question']]
        inputs = self.tokenizer(questions, dataset['context'],
                                max_length = self.max_length,
                                truncation = 'only_second',
                                stride = self.stride,
                                return_overflowing_tokens = True,
                                return_offsets_mapping = True,
                                padding = 'max_length')
        
        sample_map = inputs.pop('overflow_to_sample_mapping')
        example_ids = []
        for i in range(len(inputs['input_ids'])):
            sample_idx = sample_map[i]
            example_ids.append(dataset['id'][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs['offset_mapping'][i]
            inputs['offset_mapping'][i] = [j if sequence_ids[k] == 1 else None for k, j in enumerate(offset)]

        inputs['example_id'] = example_ids 
        return inputs 