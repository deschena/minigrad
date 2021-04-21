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
    if positive == 0 or gt_positive == 0 or true_positive == 0:
        return 0
    precision = true_positive / positive
    recall = true_positive / gt_positive

    return 2 / (1/precision + 1/recall)
# =====================================================================================================================================================================

def compute_statistics(loss_fct, model, x, y):
    
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