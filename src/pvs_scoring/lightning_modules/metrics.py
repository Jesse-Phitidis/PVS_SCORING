from monai.metrics import DiceMetric
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import torch
import numpy as np

class SegmentorValMetric(DiceMetric):
    def aggregate(self, *args, **kwargs):
        return {"DSC": super().aggregate(*args, **kwargs)[0].item()}
    
    
class ClassifierValMetric:

    def __init__(self): 
        self.BG_derivative_preds, self.CSO_derivative_preds = [], []
        self.BG_derivative_gts, self.CSO_derivative_gts = [], []
        
    def __call__(self, pred, gt):
        # pred is [(B,C), (B,C)] and gt is [(B), (B)]
        self.BG_derivative_preds.append(pred[0]), self.CSO_derivative_preds.append(pred[1])
        self.BG_derivative_gts.append(gt[0]), self.CSO_derivative_gts.append(gt[1])
        
    def reset(self):
        self.BG_derivative_preds, self.CSO_derivative_preds = [], []
        self.BG_derivative_gts, self.CSO_derivative_gts = [], []
        
    def aggregate(self):
        BG_derivative_preds, CSO_derivative_preds = np.array(torch.cat(self.BG_derivative_preds, dim=0)), np.array(torch.cat(self.CSO_derivative_preds, dim=0))
        BG_derivative_gts, CSO_derivative_gts = np.array(torch.cat(self.BG_derivative_gts, dim=0)), np.array(torch.cat(self.CSO_derivative_gts, dim=0))
        
        # Average Precision Score
        BG_AP = average_precision_score(label_binarize(BG_derivative_gts, classes=[0,1,2]), BG_derivative_preds, average=None)
        CSO_AP = average_precision_score(label_binarize(CSO_derivative_gts, classes=[0,1,2]), CSO_derivative_preds, average=None)
        
        # F1 Score
        BG_F1 = f1_score(BG_derivative_gts, np.argmax(BG_derivative_preds, axis=1), average=None, labels=[0,1,2], zero_division=1.0)
        CSO_F1 = f1_score(CSO_derivative_gts, np.argmax(CSO_derivative_preds, axis=1), average=None, labels=[0,1,2], zero_division=1.0)
        
        # Recall Score
        BG_recall = recall_score(BG_derivative_gts, np.argmax(BG_derivative_preds, axis=1), average=None, labels=[0,1,2], zero_division=1.0)
        CSO_recall = recall_score(CSO_derivative_gts, np.argmax(CSO_derivative_preds, axis=1), average=None, labels=[0,1,2], zero_division=1.0)
        
        # Precision Score
        BG_precision = precision_score(BG_derivative_gts, np.argmax(BG_derivative_preds, axis=1), average=None, labels=[0,1,2], zero_division=1.0)
        CSO_precision = precision_score(CSO_derivative_gts, np.argmax(CSO_derivative_preds, axis=1), average=None, labels=[0,1,2], zero_division=1.0)
        
        output = {}
        
        for i in range(3):
            output[f"BG_derivative_{i + 1}/AP"] = BG_AP[i]
            output[f"BG_derivative_{i + 1}/F1"] = BG_F1[i]
            output[f"BG_derivative_{i + 1}/recall"] = BG_recall[i]
            output[f"BG_derivative_{i + 1}/precision"] = BG_precision[i]
            output[f"CSO_derivative_{i + 1}/AP"] = CSO_AP[i]
            output[f"CSO_derivative_{i + 1}/F1"] = CSO_F1[i]
            output[f"CSO_derivative_{i + 1}/recall"] = CSO_recall[i]
            output[f"CSO_derivative_{i + 1}/precision"] = CSO_precision[i]
            
        output["BG_derivative/AP"] = np.mean(BG_AP)
        output["BG_derivative/F1"] = np.mean(BG_F1)
        output["BG_derivative/recall"] = np.mean(BG_recall)
        output["BG_derivative/precision"] = np.mean(BG_precision)
        output["CSO_derivative/AP"] = np.mean(CSO_AP)
        output["CSO_derivative/F1"] = np.mean(CSO_F1)
        output["CSO_derivative/recall"] = np.mean(CSO_recall)
        output["CSO_derivative/precision"] = np.mean(CSO_precision)
        
        output[f"AP"] = (output["BG_derivative/AP"] + output["CSO_derivative/AP"]) / 2
        output[f"F1"] = (output["BG_derivative/F1"] + output["CSO_derivative/F1"]) / 2
        
        return output