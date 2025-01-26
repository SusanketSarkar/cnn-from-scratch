import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        # Convert probabilities to class predictions
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        return accuracy_score(y_true_classes, y_pred_classes)

    @staticmethod
    def precision_recall_f1(y_true, y_pred):
        # Convert probabilities to class predictions
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        
        # Calculate metrics (macro averaging)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_classes, 
            y_pred_classes, 
            average='macro'
        )
        return precision, recall, f1

    @staticmethod
    def all_metrics(y_true, y_pred):
        accuracy = Metrics.accuracy(y_true, y_pred)
        precision, recall, f1 = Metrics.precision_recall_f1(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    @staticmethod
    def confusion_matrix(y_true, y_pred, normalize=True):
        from sklearn.metrics import confusion_matrix
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm