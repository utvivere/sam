import argparse
import torch
import wandb

from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats

import sys; sys.path.append("..")
from sam import SAM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=False, action="store_true", help"Include argument --adaptive if you want to set it to True")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--wandb_name", default="test0", type=str)
    parser.add_argument("--norm", default=2, type=int)
    #parser.add_argument("--datasets")
    args = parser.parse_args()
        
    wandb.login(key='14aca18a3cf267e1aea9c50e64f59e33d3bae401')
    wandb.init(
        # set the wandb project where this run will be logged
        project="optml",
        name = args.wandb_name,

        # track hyperparameted run metadata
        config={
            "dataset": "CIFAR-10",
            'rho': args.rho,
            "adaptive": args.adaptive,
            "norm": args.norm,
            "wandb_name": args.wandb_name,
            "epochs": 10,
        }
    )

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        epoch_train_loss = 0
        epoch_train_correct = 0
        num_train_batches = 0

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                epoch_train_loss += loss.sum().item()
                epoch_train_correct += correct.sum().item()
                num_train_batches += len(targets)
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        # Compute average loss and accuracy for the epoch
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_correct = epoch_train_correct / num_train_batches

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        epoch_eval_loss = 0
        epoch_eval_correct = 0
        num_eval_batches = 0

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                correct = torch.argmax(predictions, 1) == targets
                epoch_eval_loss += loss.sum().item()
                epoch_eval_correct += correct.sum().item()
                num_eval_batches += len(targets)
                log(model, loss.cpu(), correct.cpu())

        # Compute average loss and accuracy for the epoch
        avg_eval_loss = epoch_eval_loss / num_eval_batches
        avg_eval_correct = epoch_eval_correct / num_eval_batches
        wandb.log({"eval_loss": avg_eval_loss, "eval_accuracy": avg_eval_correct, "train_loss": avg_train_loss, "train_accuracy": avg_train_correct})

    log.flush()
