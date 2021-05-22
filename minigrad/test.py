import nn
import optim
import math
import torch 
from matplotlib import pyplot as plt

# =====================================================================================================================================================================
# Setups pyplot layout and deactivate PyTorch autograd
# NOTE: RUNNING THE FILE SHOULD TAKE LESS THAN 10 MINUTES. IT TRAINS SEVERAL MODELS AND DRAWS PLOTS
from matplotlib.pyplot import figure
figure(dpi=300)
torch.set_grad_enabled(False)
torch.manual_seed(999)
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
    return mean_loss.item(), accuracy, f1
# =====================================================================================================================================================================

def train_model(model, loss_fct, optimizer, x_train, y_train, x_test, y_test, epochs = 100):
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
            # loss_fct(pred, label)
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

def plot_adam_vs_sgd(x, test_adam, test_sgd, test_sgdm, metric, momentum):
    """Utility function for comparing convergence of Adam and SGD

    Args:
        x (Iterable): x axis values for plots
        train_adam (Iterable): Metric at each epoch on training data with adam optimizer
        test_adam (Iterable): Metric at each epoch on validation data with adam optimizer
        train_sgd (Iterable): Metric at each epoch on training data with SGD optimizer
        test_sgd (Iterable): Metric at each epoch on validation data with SGD optimizer
        metric (str): Metric under assessment
    """
    figure(figsize=(8, 6), dpi=300)
    plt.plot(x, test_adam, label="Adam")
    plt.plot(x, test_sgd, label="SGD")
    plt.plot(x, test_sgdm, label=f"SGD w/ momentum = {momentum}")

    plt.xlabel("Epoch")
    plt.grid()
    plt.legend(fontsize=20)
    plt.savefig(f"{metric}_adam_sgd.png")
    plt.tight_layout()
    plt.clf()
    # =====================================================================================================================================================================

def plot_circle_samples(samples, labels):

    central_points = samples[labels == 1]
    outter_points = samples[labels == 0]
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(central_points[:, 0], central_points[:, 1], c="limegreen")
    ax.scatter(outter_points[:, 0], outter_points[:, 1], c="m")
    circ = plt.Circle(  (0.5, 0.5), 1 / (2 * math.pi) ** 0.5, 
                        fill=False, 
                        color="red", 
                        linewidth=6)
    ax.add_artist(circ)
    fig.savefig("samples.png", dpi=300)
    plt.clf()
# =====================================================================================================================================================================

def simple_three_layer_model(use_sigmoid=False):
    model = nn.Sequential(
            nn.Linear(2, 25),
            nn.ReLU(),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
            )
    if use_sigmoid:
        model.add_module(nn.Sigmoid())
    return model
# =====================================================================================================================================================================
def compare_optims(nb_samples=1000, nb_epochs=70, sgd_momentum=0.9, use_sigmoid=False, ):
    """Simple first experiment with circular boundary

    Args:
        nb_samples (int, optional): Number of samples for train/test. Defaults to 1000.
        nb_epochs (int, optional): Number of training epochs. Defaults to 300.
    """
    x_train, y_train = generate_circle_samples(nb_samples, seed=42)
    x_test, y_test = generate_circle_samples(nb_samples, seed=66)

    lrs = [0.1, 0.01, 0.001]
    # Train first model with SGD without momentum
    best_lr_sgd = -1
    best_f1_sgd = -1
    f1_seq_sgd = None

    for lr in lrs:
        model = simple_three_layer_model(use_sigmoid)
        optimizer = optim.SGD(model, momentum=0)
        loss_fct = nn.MSELoss()
        print(f"Start training with SGD lr={lr}")
        _, _, _, _, _, test_f1_score_SGD = train_model(model, loss_fct, optimizer, x_train, y_train, x_test, y_test, epochs=nb_epochs)
        curr_f1 = max(test_f1_score_SGD)
        if curr_f1 > best_f1_sgd:
            best_f1_sgd = curr_f1
            f1_seq_sgd = test_f1_score_SGD
            best_lr_sgd = lr

    # Train second model with SGD with momentum
    best_lr_sgdm = -1
    best_f1_sgdm = -1
    f1_seq_sgdm = None

    for lr in lrs:
        model = simple_three_layer_model(use_sigmoid)
        optimizer = optim.SGD(model, momentum=sgd_momentum)
        loss_fct = nn.MSELoss()
        print(f"Start training with SGDM lr={lr}")
        _, _, _, _, _, test_f1_score_SGDM = train_model(model, loss_fct, optimizer, x_train, y_train, x_test, y_test, epochs=nb_epochs)
        curr_f1 = max(test_f1_score_SGDM)
        if curr_f1 > best_f1_sgdm:
            best_f1_sgdm = curr_f1
            f1_seq_sgdm = test_f1_score_SGDM
            best_lr_sgdm = lr

    # Train third model with Adam
    best_lr_adam = -1
    best_f1_adam = -1
    f1_seq_adam = None

    for lr in lrs:
        model = simple_three_layer_model(use_sigmoid)
        optimizer = optim.Adam(model, lr=lr)
        loss_fct = nn.MSELoss()
        print(f"Start training with Adam lr={lr}")
        _, _, _, _, _, test_f1_score_adam = train_model(model, loss_fct, optimizer, x_train, y_train, x_test, y_test, epochs=nb_epochs)
        
        curr_f1 = max(test_f1_score_adam)
        if curr_f1 > best_f1_adam:
            best_f1_adam = curr_f1
            f1_seq_adam = test_f1_score_adam
            best_lr_adam = lr

    
    # Plot best f1_score for each optimizer
    x = range(nb_epochs) 
    plot_adam_vs_sgd(x,
                    f1_seq_adam, 
                    f1_seq_sgd,
                    f1_seq_sgdm,
                    "F1-score",
                    momentum=sgd_momentum)

    # Compute the best result for each method
    print(f"Adam result: F1: {best_f1_adam :.3f} with lr={best_lr_adam}")
    print(f"SGD result: F1: {best_f1_sgd :.3f} with lr={best_lr_sgd}")
    print(f"SGDM result: F1: {best_f1_sgdm :.3f} with lr={best_lr_sgdm}")

# =====================================================================================================================================================================

def main():
    samples, labels = generate_circle_samples(1000)
    plot_circle_samples(samples, labels)
    compare_optims()
# =====================================================================================================================================================================

if __name__ == "__main__":
    main()