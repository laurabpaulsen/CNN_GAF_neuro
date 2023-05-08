import numpy as np
from pathlib import Path
import argparse
import torch
import multiprocessing as mp

# sklearn tools
from sklearn.metrics import classification_report

# local imports 
from cnn_funcs import load_gafs, prep_dataloaders, prep_model, train_model, plot_history, predict

def parse_args():
    parser = argparse.ArgumentParser(description='Train a CNN on GAFs')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sub', type=str, default='sub-01')

    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(__file__)

    # load in data
    if args.sub == 'all':
        gaf_path = path.parents[1] / "data" / "gaf"
        all = True
    else:
        gaf_path = path.parents[1] / "data" / "gaf" / args.sub
        all = False

    gafs, labels = load_gafs(gaf_path, n_jobs=mp.cpu_count(), all_subjects=all)

    # get dataloaders
    train_loader, val_loader, test_loader, y_test = prep_dataloaders(gafs, labels, batch_size=args.batch_size)

    # prep model
    model, optimizer, criterion = prep_model(lr = args.lr)

    # train model
    history = train_model(model, optimizer, criterion, train_loader, val_loader, epochs=args.epochs)

    # subject output path
    sub_mdl_path = path.parents[1] / "mdl" / args.sub 
    
    # check that outpath exists
    if not sub_mdl_path.exists():
        sub_mdl_path.mkdir()

    # save model
    torch.save(model.state_dict(), sub_mdl_path / "gaf_model.pt")

    # plot losses and accuracies
    plot_history(history, save_path= sub_mdl_path / "history.png")

    # test model
    predictions = predict(model, test_loader)

    # report metrics
    clf_report= classification_report(y_test, np.round(predictions), target_names=["Animate", "Inanimate"])
    accuracy = (y_test == np.round(predictions)).sum() / len(y_test)

    # save metrics
    with open(sub_mdl_path / "classification_report.txt", "w") as f:
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Subject: {args.sub}\n")

        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(clf_report)

if __name__ == "__main__":
    main()