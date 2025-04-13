from datasets import load_dataset
import xml.etree.ElementTree as ET
import csv

class PolarityMapping:
    INDEX_TO_POLARITY = { 0: None, 1: 'positive', 2: 'negative', 3: 'neutral' }
    # INDEX_TO_ONEHOT = { i: [1 if i == j else 0 for j in INDEX_TO_POLARITY] for i in INDEX_TO_POLARITY } 
    # POLARITY_TO_INDEX = { polarity: index for index, polarity in INDEX_TO_POLARITY.items() }
    INDEX_TO_ONEHOT = { 0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1] }
    POLARITY_TO_INDEX = { None: 0, 'positive': 1, 'negative': 2, 'neutral': 3 }

class Eval2014Loader:
    @staticmethod
    def load(train_csv_path, val_csv_path, test_csv_path):
        dataset_paths = {'train': train_csv_path, 'val': val_csv_path, 'test': test_csv_path}
        raw_datasets = load_dataset('csv', data_files={ k: v for k, v in dataset_paths.items() if v })
        return raw_datasets
    
    @staticmethod
    def xmlToCSV(xmlPath, csvPath):
        tree = ET.parse(xmlPath)
        root = tree.getroot()

        # Collect all unique categories and aspect terms from the XML
        unique_categories = set()
        sentences_data = []

        for sentence in root.findall('sentence'):
            review = sentence.find("text").text

            categories = sentence.find('aspectCategories')
            # aspect_terms = sentence.find('aspectTerms')

            row_data = {'Review': review}  # Initialize a row with the review text
            
            if categories is not None:
                for category in categories.findall('aspectCategory'):
                    cat_name = category.attrib['category']
                    polarity = category.attrib['polarity']
                    unique_categories.add(cat_name)
                    # Set the corresponding polarity value (1 for positive, 2 for negative, 3 for neutral)
                    if polarity == 'positive':
                        row_data[cat_name] = 1
                    elif polarity == 'negative':
                        row_data[cat_name] = 2
                    elif polarity == 'neutral':
                        row_data[cat_name] = 3

            # if aspect_terms is not None:
            #     for aspect in aspect_terms.findall('aspectTerm'):
            #         aspect_term = aspect.attrib['term']
            #         polarity = aspect.attrib['polarity']
            #         unique_categories.add(aspect_term)
            #         # Set the corresponding polarity value for aspect terms
            #         if polarity == 'positive':
            #             row_data[aspect_term] = 1
            #         elif polarity == 'negative':
            #             row_data[aspect_term] = 2
            #         elif polarity == 'neutral':
            #             row_data[aspect_term] = 3

            sentences_data.append(row_data)

        # Convert categories to a sorted list for consistent ordering
        sorted_categories = sorted(unique_categories)

        # Write to CSV
        with open(csvPath, mode='w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            header = ['Review'] + sorted_categories
            writer.writerow(header)

            # Write the rows
            for row in sentences_data:
                csv_row = [row.get('Review', '')] + [row.get(category, 0) for category in sorted_categories]
                writer.writerow(csv_row)
        
        print(f"CSV file generated:" + csvPath)

    @staticmethod
    def preprocess_and_tokenize(text_data, preprocessor, tokenizer,  batch_size, max_length):
        print('[INFO] Tokenizing text data...')

        def transform_each_batch(batch):
            preprocessed_batch = preprocessor.process_batch(batch)
            return tokenizer(preprocessed_batch, max_length=max_length, padding='max_length', truncation=True)
        
        if isinstance(text_data, str):
            return transform_each_batch([text_data])

        # For datasets (like pandas DataFrame or Hugging Face datasets)
        return text_data.map(
            lambda batch: transform_each_batch(batch['Review']),
            batched=True, batch_size=batch_size
        ).remove_columns('Review')
    
    @staticmethod
    def labels_to_flatten_onehot(datasets):
        print('[INFO] Transforming "Aspect#Categoy,Polarity" labels to flattened one-hot encoding...')
        model_input_names = ['input_ids', 'token_type_ids', 'attention_mask']
        label_columns = [col for col in datasets['train'].column_names if col not in ['Review', *model_input_names]]
        def transform_each_review(review): # Convert each Aspect#Categoy,Polarity to one-hot encoding and merge them into 1D list
            review['FlattenOneHotLabels'] = sum([
                PolarityMapping.INDEX_TO_ONEHOT[review[aspect_category]] # Get one-hot encoding
                for aspect_category in label_columns
            ], []) # Need to be flattened to match the model's output shape
            return review 
        return datasets.map(transform_each_review, num_proc=8).select_columns(['FlattenOneHotLabels', *model_input_names])