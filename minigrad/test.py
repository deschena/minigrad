from nn import *
from optim import *

from minigrad import nn
from minigrad import optim
import math
import torch 
from matplotlib import pyplot as plt

# =====================================================================================================================================================================
# Setups pyplot layout and deactivate PyTorch autograd
from matplotlib.pyplot import figure
figure(figsize=(8, 8), dpi=80)
torch.set_grad_enabled(False)
# =====================================================================================================================================================================

def generate_circle_samples(nb_samples, seed=42):
    """Generate data as asked in the project description.

    Args:
        nb_samples (int): How many samples are required
        seed (int, optional): Seed of random generator, for reproducibility. Defaults to 42.

    Returns:
        (samples, labels): (2D points in space, labels associated)
    """
    torch.random.manual_seed(seed)
    samples = torch.rand(size=(nb_samples, 2))
    center = torch.tensor([[0.5, 0.5]])

    dist = ((samples - center) ** 2).sum(dim=1) ** 0.5
    radius = 1 / ((2 * math.pi) ** 0.5)

    labels = torch.empty(nb_samples)
    labels[dist >= radius] = 0.0
    labels[dist < radius] = 1.0

    return samples, labels
# =====================================================================================================================================================================

def f1_score (true_positive, positive, gt_positive):
    """Compute the F1-score

    Args:
        true_positive (Int): Number of true positives
        positive (Int): Number of positively classified samples
        gt_positive (Int): Number of effectively positive samples

    Returns:
        float: F1-score
    """
    if positive <= 0 or gt_positive <= 0 or true_positive <= 0:
        return 0
    precision = true_positive / positive
    recall = true_positive / gt_positive

    return 2 / (1 / precision + 1 / recall)
# =====================================================================================================================================================================

def compute_statistics(loss_fct, model, x, y):
    """Compute statistics of the model on labelled data. Expect binary classification: output will be clipped/rounded to {0, 1}

    Args:
        loss_fct (Callable): Loss function used during training
        model (nn.Module): ML model under assessment
        x (Iterable): Samples to use to evaluate the model
        y (Iterable): Associated labels, will be compared with prediction of the model

    Returns:
        (float, float, float): (mean loss, accuracy, F1-score)
    """
    
    loss = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for sample, label in zip(x, y):
        pred = model(sample)
        loss += loss_fct(pred, label)
        pred_class = 1 if pred > 0.5 else 0

        true_positive += 1 if label == 1 and pred_class == 1 else 0
        false_positive += 1 if label == 0 and pred_class == 1 else 0
        true_negative += 1 if label == 0 and pred_class == 0 else 0
        false_negative += 1 if label == 1 and pred_class == 0 else 0

    mean_loss = loss / len(y)
    accuracy = (true_positive + true_negative) / len(y)
    f1 = f1_score(true_positive, true_positive + false_positive, true_positive + false_negative)
    return mean_loss, accuracy, f1
# =====================================================================================================================================================================

def train_model(model, loss_fct, optimizer, x_train, y_train, x_test, y_test, epochs = 300):
    """Training loop of the model

    Args:
        model (nn.Module): Model to train
        loss_fct (Callable): Performance measure. Of the form (pred, target) -> loss
        optimizer (Optim): Optimizer observing model, used to update weights of the model
        x_train (Iterable): Training samples
        y_train (Iterable): Training labels
        x_test (Iterable): Test samples
        y_test (Iterable): Test labels
        epochs (int, optional): Number of epochs to train the model. Defaults to 300.

    Returns:
        (6*Iterable): (Training loss, Train accuracy, Train F1-score, Test loss, Test accuracy, Test F1-score)
    """
    train_losses = []
    train_accuracy = []
    train_f1_score = []

    test_losses = []
    test_accuracy = []
    test_f1_score = []

    for _ in range(epochs):
        for sample, label in zip(x_train, y_train):
            # Forward pass
            pred = model(sample)
            loss_fct(pred, label)
            # Backward pass
            loss_grad = loss_fct.backward(pred, label)
            optimizer.zero_grad()
            model.backward(loss_grad, sample)
            optimizer.step()
        # Save statistics for plotting
        train_mean_loss, train_accuracy_epoch, train_f1 = compute_statistics(loss_fct, model, x_train, y_train)
        train_losses.append(train_mean_loss)
        train_accuracy.append(train_accuracy_epoch)
        train_f1_score.append(train_f1)

        test_mean_loss, test_accuracy_epoch, test_f1 = compute_statistics(loss_fct, model, x_test, y_test)
        test_losses.append(test_mean_loss)
        test_accuracy.append(test_accuracy_epoch)
        test_f1_score.append(test_f1)

    return train_losses, train_accuracy, train_f1_score, test_losses, test_accuracy, test_f1_score
# =====================================================================================================================================================================

def plot(x, y_train, y_test, metric, model_struct, filename, xlog=False, ylog=False):
    """Plot metric related to training of the model

    Args:
        x (Iterable): x axis data
        y_train (Iterable): Metric on train set
        y_test (Iterable): Metric on test set
        metric (str): Name of the metric used. Displayed in title
        model_struct (str): Description of the model structure. Displayed in title
        filename (str): Name of file where to save the figure
        xlog (bool, optional): Use logscale on x axis. Defaults to False.
        ylog (bool, optional): Use logscale on y axis. Defaults to False.
    """
    fig, ax = plt.subplots()
    ax.set_title(f"{metric} ({model_struct})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_yscale("log" if ylog else "linear")
    ax.set_xscale("log" if xlog else "linear")
    ax.plot(x, y_train, label=f"Train {metric}")
    ax.plot(x, y_test, label=f"Test {metric}")
    ax.grid()
    ax.legend()
    fig.savefig(filename)
# =====================================================================================================================================================================

def circle_experiment(nb_samples=1000, nb_epoch=300):
    """Simple first experiment with circular boundary

    Args:
        nb_samples (int, optional): Number of samples for train/test. Defaults to 1000.
        nb_epoch (int, optional): Number of training epochs. Defaults to 300.
    """
    x_train, y_train = generate_circle_samples(nb_samples, seed=42)
    x_test, y_test = generate_circle_samples(nb_samples, seed=66)

    model = nn.Sequential(
    nn.Linear(2, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 25),
    nn.ReLU(),
    nn.Linear(25, 1)
    )

    optimizer = optim.SGD(model, lr=0.001)
    loss_fct = nn.MSELoss()

    (train_losses, 
    train_accuracy, 
    train_f1_score, 
    test_losses, 
    test_accuracy, 
    test_f1_score) = train_model(model, loss_fct, optimizer, x_train, y_train, x_test, y_test, epochs=nb_epoch)

    x = range(nb_epoch)
    model_struct = "3 hidden layers, 25 units, ReLU"
    plot(x, train_losses, test_losses, "Loss", model_struct, "loss_circle.png")
    plot(x, train_accuracy, test_accuracy, "Accuracy", model_struct, "accuracy_circle.png")
    plot(x, train_f1_score, test_f1_score, "F1-score", model_struct, "f1_circle.png")
# =====================================================================================================================================================================

def main():
    circle_experiment(nb_epoch=300)
# =====================================================================================================================================================================

if __name__ == "__main__":
    main()