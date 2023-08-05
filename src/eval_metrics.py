import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score,classification_report


def multiclass_acc3(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    senti=[0,0,0]
    true=[0,0,0]
    for i,n in enumerate(senti):
        senti[i]=np.sum(truths==i)
    for i,pred in enumerate(preds):
        if np.round(pred)==np.round(truths[i]):
            true[int(pred)]+=1
    print(senti)
    print(true)
    acc_all=np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
    acc_neg=true[0]/senti[0] 
    acc_neu=true[1]/senti[1]
    acc_pos=true[2]/senti[2]
    return acc_all,acc_neg,acc_neu,acc_pos


def eval_3_senti_RU(results, truths, exclude_zero=False):
    test_preds = results.max(1,keepdim=False)[1].cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    
    non_neu = np.array([i for i, e in enumerate(test_truth) if e != 2 or (not exclude_zero)])

    test_preds_a3 = np.clip(test_preds, a_min=1., a_max=3.)
    test_truth_a3 = np.clip(test_truth, a_min=1., a_max=3.)

    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a3,a_neg,a_neu,a_pos = multiclass_acc3(test_preds, test_truth)
    f_score = f1_score(np.round(test_preds), np.round(test_truth), average='micro')
    binary_truth = (test_truth[non_neu] > 0)
    binary_preds = (test_preds[non_neu] > 0)
    print("acc_3: ", mult_a3)
    print("F1 score: ", f_score)
    print("-" * 50)
    return mult_a3,f_score

def eval_3_senti_MVSA(results, truths, exclude_zero=False):
    test_preds = results.max(1,keepdim=False)[1].cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    
    non_neu = np.array([i for i, e in enumerate(test_truth) if e != 2 or (not exclude_zero)])

    test_preds_a3 = np.clip(test_preds, a_min=1., a_max=3.)
    test_truth_a3 = np.clip(test_truth, a_min=1., a_max=3.)

    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a3,a_neg,a_neu,a_pos = multiclass_acc3(test_preds, test_truth)
    f_score = f1_score(np.round(test_preds), np.round(test_truth), average='macro')
    binary_truth = (test_truth[non_neu] > 0)
    binary_preds = (test_preds[non_neu] > 0)
    print("mult_acc_3: ", mult_a3)
    print("F1 score: ", f_score)
    print("-" * 50)
    return mult_a3,f_score












