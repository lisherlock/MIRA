# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:39:52 2024

@author: limy
"""

import jieba
from rouge import Rouge
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import re
import csv


def calculate_rouge(reference, generated):
    import jieba
    from rouge import Rouge

    ref_words = ' '.join(jieba.lcut(reference))
    gen_words = ' '.join(jieba.lcut(generated))
    rouge = Rouge()
    scores = rouge.get_scores(gen_words, ref_words)

    total_r = 0
    total_p = 0
    total_f = 0
    count = 0

    for score in scores:
        for rouge_type in score.keys():
            total_r += score[rouge_type]['r']
            total_p += score[rouge_type]['p']
            total_f += score[rouge_type]['f']
            count += 1

    avg_r = total_r / count if count > 0 else 0
    avg_p = total_p / count if count > 0 else 0
    avg_f = total_f / count if count > 0 else 0

    avg_scores = {'Average_R': avg_r, 'Average_P': avg_p, 'Average_F': avg_f}
    return avg_scores



from bert_score import score

def calculate_bertscore(reference, generated):
    P, R, F1 = score(
        [generated],
        [reference],
        model_type= '',  # bert-base-chinese path
        num_layers=12, 
        idf=False,
        rescale_with_baseline=False
    )
    return {'Precision': P.mean().item(), 'Recall': R.mean().item(), 'F1': F1.mean().item()}




from sentence_transformers import SentenceTransformer, util

def calculate_sentence_similarity(reference, generated):
    model = SentenceTransformer(r'')  # SentenceTransformer(distiluse-base-multilingual-cased-v1')
    embeddings = model.encode([reference, generated])
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()






def extract_medical_entities(text):
    entities = medical_ner_model.extract_entities(text)
    return set(entities)

def compare_entities(reference, generated):
    ref_entities = extract_medical_entities(reference)
    gen_entities = extract_medical_entities(generated)
    common_entities = ref_entities.intersection(gen_entities)
    precision = len(common_entities) / len(gen_entities) if gen_entities else 0
    recall = len(common_entities) / len(ref_entities) if ref_entities else 0
    return {'Precision': precision, 'Recall': recall}



def evaluate_medical_report(reference, generated):
    rouge_scores = calculate_rouge(reference, generated)
    bert_scores = calculate_bertscore(reference, generated)
    sentence_similarity = calculate_sentence_similarity(reference, generated)
    evaluation = {
        'ROUGE': rouge_scores,
        'BERTScore': bert_scores,
        'SentenceSimilarity': sentence_similarity,
    }
    return evaluation


def Split_report(text):
    lines = text.split('\n')
    processed_lines = [re.sub(r'^\s*\d+\s*[.,、\-)\]]*\s*', '', line) for line in lines]

    single_line_text = ' '.join(processed_lines)

    if single_line_text == '' or single_line_text == ' ':
        return text
    else:
        return single_line_text


def clean_text(text):
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,.;，。；、]+', '', text)
    return text.strip()


# 主函数
if __name__ == '__main__':

    input_folder = r''
    output_folder = r''

    csv_name = 'total_report_chexpert_final'
    folder_lists = ['fold1', 'fold2', 'fold3']


    for folder_list in folder_lists:

        input_csv_path = os.path.join(input_folder, folder_list, 'DR', csv_name + '.csv')

        csv_file = pd.read_csv(input_csv_path, encoding='utf-8-sig')
        IDs = csv_file['ID'].values.tolist()
        impressions = csv_file['impression'].values.tolist()
        gts = csv_file['gt'].values.tolist()


        column_name = ['ID', 'ROUGE-R', 'ROUGE-P', 'ROUGE-F', 'BERTScore-P', 'BERTScore-R', 'BERTScore-F', 'Sentence Similarity']
        init_csv = pd.DataFrame(columns=column_name)
        init_csv.to_csv(os.path.join(output_folder, csv_name + '_result_' + folder_list + '.csv'), index=False, mode='w')
        

        for i in tqdm(range(len(IDs))):
            cleaned_gt = clean_text(gts[i]) 
            cleaned_impression = clean_text(impressions[i])
            result = evaluate_medical_report(cleaned_gt, cleaned_impression)
   
            pd.DataFrame(np.column_stack((
                                            str(IDs[i]),
                                            round(result['ROUGE']['Average_R'], 4), 
                                            round(result['ROUGE']['Average_P'], 4), 
                                            round(result['ROUGE']['Average_F'], 4), 
                                            round(result['BERTScore']['Precision'], 4), 
                                            round(result['BERTScore']['Recall'], 4), 
                                            round(result['BERTScore']['F1'], 4),
                                            round(result['SentenceSimilarity'], 4)
                                        ))
                                        ).to_csv(os.path.join(output_folder,  csv_name + '_result_' + folder_list + '.csv'), index=False, mode='a', header=False)

            



