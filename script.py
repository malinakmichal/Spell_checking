import pandas as pd
from transformers import AutoModelForSeq2SeqLM, BertTokenizer, T5Tokenizer, BertForMaskedLM, logging
import torch
from rouge_score import rouge_scorer
import numpy as np
from spellchecker import SpellChecker


dataset_misspelled = [
    ("I hav a dreem to chase.", "I have a dream to chase."),
    ("She is a grea friend.", "She is a great friend."),
    ("It was an amazng experience.", "It was an amazing experience."),
    ("The weather is quite lovly today.", "The weather is quite lovely today."),
    ("He is definitly coming to the party.", "He is definitely coming to the party."),
    ("We should go to the beech this weekend.", "We should go to the beach this weekend."),
    ("This is a beautful painting.", "This is a beautiful painting."),
    ("Can you beleive it's already October.", "Can you believe it's already October."),
    ("I need to fix the leaky fauce.", "I need to fix the leaky faucet."),
    ("The quick brown fox jumps over the lazzy dog.", "The quick brown fox jumps over the lazy dog."),
    ("She enjoys readng books in her free time.", "She enjoys reading books in her free time."),
    ("My favorite fruits are bananas and appless.", "My favorite fruits are bananas and apples."),
    ("I like to drink coffe every morning.", "I like to drink coffee every morning."),
    ("He is a talented writter.", "He is a talented writer."),
    ("The flowers are blooming in the gardn.", "The flowers are blooming in the garden."),
    ("I always forget my umbrella in the rainny season.", "I always forget my umbrella in the rainy season."),
    ("They went to the restraunt for dinner.", "They went to the restaurant for dinner."),
    ("Her birthday is on the 25th of Decmber.", "Her birthday is on the 25th of December."),
    ("This cake is delicius.", "This cake is delicious."),
    ("I will send you the infromation.", "I will send you the information."),
    ("He has a great sense of humr.", "He has a great sense of humor.")
]


def prepare_data():
    """
    Loads data from an Excel file and prepares it for model input.

    Returns:
    pd.DataFrame: A DataFrame containing data created by me.
    pd.DataFrame: A DataFrame containing data from excel file.
    """

    dataset_mine = pd.DataFrame(dataset_misspelled, columns=["Errneous Sentence", "Correct Sentence"])

    dataset = pd.read_excel("GTD.xlsx")
    dataset_downloaded = dataset[dataset['Error Type'] == "Spelling Error"]
    dataset_downloaded = dataset_downloaded[["Correct Sentence", "Errneous Sentence", "Error Type"]]
    dataset_downloaded = dataset_downloaded.dropna()

    return dataset_mine, dataset_downloaded


def pySpellChecker(dataset):
    """
    Evaluates the PySpellChecker model using the input test dataset and prints out the model accuracy.

    Parameters:
    dataset (pd.DataFrame): The input test dataset for evaluating model accuracy.
    
    Returns:
    None
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    spell = SpellChecker()
    sum = 0
    rouge1 = []
    data_len = dataset.shape[0]
    for row in dataset.iterrows():
        text_split = row[1]['Errneous Sentence'].split()
        result = []
        for word in text_split:
            result += [word if not spell.unknown([word]) else (spell.correction(word) if spell.correction(word) else word)]

        res_len = len(result)
        if res_len > 1:
            result[0] = result[0].capitalize()
        result = ' '.join(result)
        if res_len > 1 and result[-1] != ".":
            result += "."

        if result == row[1]['Correct Sentence']:
            sum += 1

        score = scorer.score(row[1]['Correct Sentence'], result)
        rouge1.append(score['rouge1'])

    f1 = np.mean([s.fmeasure for s in rouge1])
    print("Rouge F1 score: ", str(round(f1, 2)*100) + " %")
    print("Sentence accuracy: " + str(round(sum*100/data_len, 2)) + " %")



def LLM_spell_checker(dataset):
    """
    Evaluates fine-tunned model for grammar-correction using the input test dataset and prints out the model accuracy.

    Parameters:
    dataset (pd.DataFrame): The input test dataset for evaluating model accuracy.
    
    Returns:
    None
    """
    checkpoint = "vennify/t5-base-grammar-correction"

    tokenizer_T5 = T5Tokenizer.from_pretrained(checkpoint, legacy=True)
    model_TP = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    sum = 0
    data_len = dataset.shape[0]
    rouge1 = []
    for row in dataset.iterrows():
        input_ids = tokenizer_T5(row[1]['Errneous Sentence'], return_tensors="pt").input_ids
        size = len(input_ids[0])
        output = model_TP.generate(input_ids=input_ids, max_new_tokens=size*1.2)
        result = tokenizer_T5.decode(output[0], skip_special_tokens=True)
        
        if result == row[1]['Correct Sentence']:
            sum += 1

        score = scorer.score(row[1]['Correct Sentence'], result)
        rouge1.append(score['rouge1'])

    f1 = np.mean([s.fmeasure for s in rouge1])
    print("Rouge F1 score: ", str(round(f1, 2)*100) + " %")
    print("Sentence accuracy: " + str(round(sum*100/data_len, 2)) + " %")


def Combination_spell_checker(dataset):
    """
    Evaluates PySpellChecker combined with fine-tunned model for grammar-correction using the input test dataset and prints out the model accuracy.

    Parameters:
    dataset (pd.DataFrame): The input test dataset for evaluating model accuracy.
    
    Returns:
    None
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    tokenizer_BERT = BertTokenizer.from_pretrained('bert-base-uncased')
    model_BERT = BertForMaskedLM.from_pretrained('bert-base-uncased')
    sum = 0
    data_len = dataset.shape[0]
    rouge1 = []
    for row in dataset.iterrows():
        sentence = row[1]['Errneous Sentence']
        if len(sentence) > 1 and sentence[-1] in [".", "?", "!"]:
            sentence = sentence[:-1]
        text_split = sentence.split()
        
        result = []
        spell = SpellChecker()
        for index, word in enumerate(text_split):
            if not spell.unknown([word]):
                result += [word]
            else:
                if spell.correction(word):
                    candidates = list(spell.candidates(word))
                    text_split_mask = list(text_split)
                    text_split_mask[index] = "[MASK]"
                    sentence_mask = " ".join(text_split_mask)
                    input_ids = tokenizer_BERT(sentence_mask, return_tensors="pt").input_ids
                    mask_index = (input_ids == tokenizer_BERT.mask_token_id).nonzero(as_tuple=True)[1].item()
                
                    candidate_ids = [tokenizer_BERT(word, add_special_tokens=False).input_ids[0] for word in candidates]

                    with torch.no_grad():
                        outputs = model_BERT(input_ids)
                        logits = outputs.logits[0, mask_index]

                    candidate_scores = {word: logits[token_id].item() for word, token_id in zip(candidates, candidate_ids)}

                    best_word = max(candidate_scores, key=candidate_scores.get) 
                    result += [best_word]
                else:
                    result += [word]

        res_len = len(result)
        if res_len > 1:
            result[0] = result[0].capitalize()
        result = ' '.join(result)
        if res_len > 1 and result[-1] != ".":
            result += "."

        if result == row[1]['Correct Sentence']:
            sum += 1

        score = scorer.score(row[1]['Correct Sentence'], result)
        rouge1.append(score['rouge1'])

    f1 = np.mean([s.fmeasure for s in rouge1])
    print("Rouge F1 score: ", str(round(f1, 2)*100) + " %")
    print("Sentence accuracy: " + str(round(sum*100/data_len, 2)) + " %")


if __name__ == "__main__":
    dataset_mine, dataset_downloaded = prepare_data() 

    logging.set_verbosity_error()

    print("PySpellChecker on my data:")
    pySpellChecker(dataset_mine)
    print()
    print("PySpellChecker on downloaded data:")
    pySpellChecker(dataset_downloaded)

    print("\n")

    print("LLM on my data:")
    LLM_spell_checker(dataset_mine)
    print()
    print("LLM on downloaded data:")
    LLM_spell_checker(dataset_downloaded)

    print("\n")

    print("Combination of both on my data:")
    Combination_spell_checker(dataset_mine)
    print()
    print("Combination on downloaded data:")
    Combination_spell_checker(dataset_downloaded)
