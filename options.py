import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Frequency Attention Networks",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_dir", type=str, default="../data/train_SIDD.hdf5",
                        help="train image directory")
    parser.add_argument("--val_dir", type=str, default="../data/val_SIDD.hdf5",
                        help="train image directory")
    parser.add_argument("--weight", type=str, default=None,
                        help="checkpoint file for fine-tuning")
    parser.add_argument("--output_path", type=str, default="./checkpoints",
                        help="checkpoint output directory")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="log directory")

    parser.add_argument("--depth_S", type=int, default=5,
                        help="the depth of Est-net")
    parser.add_argument("--depth_U", type=int, default=4,
                        help="the layers of De-net")
    parser.add_argument("--feature_dims", type=int, default=64,
                        help="initial feature dims of De-net")
    parser.add_argument("--wave", type=str, default='db1',
                        help="type of wavelet basis function")
    parser.add_argument("--level", type=int, default=1,
                        help="multi-resolution layers of wavelet transform")

    parser.add_argument("--lr_initial", type=int, default=0.0002,
                        help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=60,
                        help="number of whole epochs")
    parser.add_argument("--cycle_epochs", type=int, default=30,
                        help="number of epochs for warm restart")
    parser.add_argument("--loss_mode", type=str, default='l1_map',
                        help="loss function mode including l1_map, l2_map, l1, l2")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="batch size")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="parallel workers, suggest setting as 0 when loading a hdf5 file")
    parser.add_argument("--print_freq", type=int, default=20,
                        help="the frequency to print loss of current batch")
    parser.add_argument("--save_model_freq", type=int, default=5,
                        help="save model frequency")
    args = parser.parse_args()

    return args
