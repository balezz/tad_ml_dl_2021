import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    accuracy = 0
    # TODO: Implement computing accuracy
    # raise Exception("Not implemented!")
    accuracy = np.sum(prediction == ground_truth) / len(ground_truth)
    # end
    return accuracy
