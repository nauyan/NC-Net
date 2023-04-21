from src.losses import combined_loss
from src.metrics import dice_score
from src import trainUtil
from src.model import NC_Net
from src.dataloader import Dataset
from src.augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
)
import config

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import multiprocessing


def train(train_dir, test_dir, dataset_name):

    model = NC_Net(
        encoder=config.encoder,
        encoder_weights=config.encoder_weights,
        device=config.device,
    )

    preprocessing_fn = None

    train_dataset = Dataset(
        train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        mode="train",
    )

    valid_dataset = Dataset(
        test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        mode="test",
    )

    batch_size = config.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=32,
    )

    metrics = [
        dice_score,
    ]

    optimizer = torch.optim.RAdam([
        dict(params=model.parameters(),
             lr=config.learning_rate,
             betas=(0.9, 0.999)),
    ])

    loss_fn = combined_loss
    train_epoch = trainUtil.TrainEpoch(
        model,
        loss=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        device=config.device,
        verbose=True,
    )

    valid_epoch = trainUtil.ValidEpoch(
        model,
        loss=loss_fn,
        metrics=metrics,
        device=config.device,
        verbose=True,
    )

    writer_path = "./{}/NC-Net_{}".format(config.tensorboard_logs,
                                          dataset_name)
    writer = SummaryWriter(writer_path)
    min_loss = 9999
    max_score = 0
    last_save = 0
    for i in range(1, config.epochs):

        print("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        writer.add_scalar("Loss/train", train_logs[loss_fn.__name__], i)
        writer.add_scalar("Loss/valid", valid_logs[loss_fn.__name__], i)
        writer.add_scalar("Dice/train", train_logs[metrics[0].__name__], i)
        writer.add_scalar("Dice/valid", valid_logs[metrics[0].__name__], i)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], i)

        if min_loss > valid_logs[loss_fn.__name__]:
            min_loss = valid_logs[loss_fn.__name__]
            torch.save(
                model.state_dict(),
                "./{}/NC-Net_{}.pth".format(config.checkpoints_dir,
                                            dataset_name),
            )
            last_save = i
            print("Model saved Loss!")

        if max_score < valid_logs[metrics[0].__name__]:
            max_score = valid_logs[metrics[0].__name__]
            torch.save(
                model.state_dict(),
                "./{}/NC-Net_{}_metric.pth".format(config.checkpoints_dir,
                                                   dataset_name),
            )
            last_save = i
            print("Model saved Metric!")

        if i - last_save >= 80:
            last_save = i
            optimizer.param_groups[0][
                "lr"] = optimizer.param_groups[0]["lr"] * 0.5
            print("Decrease decoder learning rate to ",
                  optimizer.param_groups[0]["lr"])

    writer.flush()


if __name__ == "__main__":
    dataset_name = "all"
    train_dir = "data/{}/train/".format(dataset_name)
    test_dir = "data/{}/test/".format(dataset_name)
    train(train_dir, test_dir, dataset_name)

    # dataset_name = "consep"
    # train_dir = "data/{}/train/".format(dataset_name)
    # test_dir = "data/{}/test/".format(dataset_name)
    # train(train_dir, test_dir, dataset_name)

    # dataset_name = "pan1"
    # train_dir = "data/{}/train/".format(dataset_name)
    # test_dir = "data/{}/test/".format(dataset_name)
    # train(train_dir, test_dir, dataset_name)

    # dataset_name = "pan2"
    # train_dir = "data/{}/train/".format(dataset_name)
    # test_dir = "data/{}/test/".format(dataset_name)
    # train(train_dir, test_dir, dataset_name)

    # dataset_name = "pan3"
    # train_dir = "data/{}/train/".format(dataset_name)
    # test_dir = "data/{}/test/".format(dataset_name)
    # train(train_dir, test_dir, dataset_name)

    # dataset_name = "liz1"
    # train_dir = "data/{}/train/".format(dataset_name)
    # test_dir = "data/{}/test/".format(dataset_name)
    # train(train_dir, test_dir, dataset_name)

    # dataset_name = "liz2"
    # train_dir = "data/{}/train/".format(dataset_name)
    # test_dir = "data/{}/test/".format(dataset_name)
    # train(train_dir, test_dir, dataset_name)

    # dataset_name = "liz3"
    # train_dir = "data/{}/train/".format(dataset_name)
    # test_dir = "data/{}/test/".format(dataset_name)
    # train(train_dir, test_dir, dataset_name)
