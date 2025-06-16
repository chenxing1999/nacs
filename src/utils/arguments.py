import argparse


def parse_bool(v):
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args(argv=None):

    parser = argparse.ArgumentParser()

    # General options
    parser.add_argument(
        "--arch",
        default="resnet20",
        choices=[
            "resnet20",
            "resnet18",
            "resnet50",
            "deepfm",
            "dcnv2",
            "dcn_mix",
            "fmlp",
        ],
        help="model architecture",
    )
    parser.add_argument("--subset_path")
    parser.add_argument("--loss")
    parser.add_argument("--n_splits", type=int)
    parser.add_argument("--data_size", type=float)

    parser.add_argument("--data_dir", default="~/data")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "tinyimagenet",
            "criteo",
            "avazu",
            "criteo_cl",
            "avazu_cl",
        ],
        help="dataset: " + " (default: cifar10)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-equal_num", dest="equal_num", action="store_false")
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--resume_from_epoch", default=0, type=int, help="resume from a specific epoch"
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="mini-batch size (default: 128)"
    )
    parser.add_argument(
        "--infer_batch_size",
        default=None,
        type=int,
        help="mini-batch size (default: batch_size)",
    )
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="momentum")
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        help="weight decay (default: 5e-4)",
    )
    parser.add_argument(
        "--save-dir",
        default="./outputs",
        type=str,
        help="The directory used to save output",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=200,
        help="Saves checkpoints at every specified number of epochs",
    )
    parser.add_argument("--gpu", type=int, nargs="+", default=[0])

    parser.add_argument(
        "--selection_method",
        default="crest",
        choices=["none", "random", "crest", "crest_ctr"],
        help="subset selection method",
    )
    parser.add_argument("--smtk", type=int, help="smtk", default=0)
    parser.add_argument(
        "--train_frac", "-s", type=float, default=0.1, help="training fraction"
    )
    parser.add_argument("--lr_milestones", type=int, nargs="+", default=[100, 150])
    parser.add_argument(
        "--gamma", type=float, default=0.1, help="learning rate decay parameter"
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--runs", type=int, help="num runs", default=1)
    parser.add_argument(
        "--warm_start_epochs",
        default=20,
        type=int,
        help="epochs to warm start learning rate",
    )
    parser.add_argument(
        "--subset_start_epoch",
        default=0,
        type=int,
        help="epoch to start subset selection",
    )

    # data augmentation options
    parser.add_argument(
        "--cache_dataset",
        default=True,
        type=parse_bool,
        const=True,
        nargs="?",
        help="cache the dataset in memory",
    )
    parser.add_argument(
        "--clean_cache_selection",
        default=False,
        type=parse_bool,
        const=True,
        nargs="?",
        help="clean the cache when selecting a new subset",
    )
    parser.add_argument(
        "--clean_cache_iteration",
        default=True,
        type=parse_bool,
        const=True,
        nargs="?",
        help="clean the cache after iterating over the dataset",
    )

    # Crest options
    parser.add_argument(
        "--approx_moment",
        default=True,
        type=parse_bool,
        const=True,
        nargs="?",
        help="use momentum in approximation",
    )
    parser.add_argument(
        "--approx_with_coreset",
        default=True,
        type=parse_bool,
        const=True,
        nargs="?",
        help="use all (selected) coreset data for loss function approximation",
    )
    parser.add_argument(
        "--check_interval",
        default=1,
        type=int,
        help="frequency to check the loss difference",
    )
    parser.add_argument(
        "--num_minibatch_coreset",
        default=5,
        type=int,
        help="number of minibatches to select together",
    )
    parser.add_argument(
        "--batch_num_mul",
        default=5,
        type=float,
        help="multiply the number of minibatches to select together",
    )
    parser.add_argument(
        "--interval_mul",
        default=1.0,
        type=float,
        help="multiply the interval to check the loss difference",
    )
    parser.add_argument(
        "--check_thresh_factor",
        default=0.1,
        type=float,
        help="use loss times this factor as the loss threshold",
    )
    parser.add_argument(
        "--shuffle",
        default=True,
        type=parse_bool,
        const=True,
        nargs="?",
        help="use shuffled minibatch coreset",
    )

    # random subset options
    parser.add_argument(
        "--random_subset_size",
        default=0.01,
        type=float,
        help="partition the training data to select subsets",
    )
    parser.add_argument(
        "--partition_start",
        default=0,
        type=int,
        help="which epoch to start selecting by minibatches",
    )

    # dropping examples below a loss threshold
    parser.add_argument(
        "--drop_learned",
        default=False,
        type=parse_bool,
        const=True,
        nargs="?",
        help="drop learned examples",
    )
    parser.add_argument(
        "--watch_interval",
        default=5,
        type=int,
        help="decide whether an example is learned based on how many epochs",
    )
    parser.add_argument(
        "--drop_interval",
        default=20,
        type=int,
        help="decide whether an example is learned based on how many epochs",
    )
    parser.add_argument("--drop_thresh", default=0.1, type=float, help="loss threshold")
    parser.add_argument("--min_train_size", default=40000, type=int)

    # others
    parser.add_argument(
        "--use_wandb", default=False, type=parse_bool, const=True, nargs="?"
    )

    args = parser.parse_args(argv)

    if args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "tinyimagenet":
        args.num_classes = 200
    elif args.dataset in ["criteo", "avazu", "criteo_cl", "avazu_cl"]:
        args.num_classes = 2
        args.infer_batch_size = 4096
    else:
        raise NotImplementedError

    if args.infer_batch_size is None:
        args.infer_batch_size = args.batch_size

    args.save_dir = get_exp_name(args)

    return args


def get_args_v2(argv=None):
    parser = argparse.ArgumentParser()

    # main arguments
    parser.add_argument(
        "arch",
        choices=["dcnv2", "deepfm", "dcn_mix"],
        help="model architechture",
    )

    parser.add_argument(
        "dataset",
        choices=["criteo"],
        help="Dataset names",
    )

    # options
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--no-equal_num",
        dest="equal_num",
        action="store_false",
        help="Same number of positive and negative samples",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed")


def get_exp_name(args):
    grd = ""
    grd += args.selection_method if args.selection_method != "none" else ""
    grd += f"-batchnummul{args.batch_num_mul}-interalmul{args.interval_mul}"
    grd += f"_thresh-factor{args.check_thresh_factor}"
    folder = f"/{args.dataset}"
    args.save_dir += f"{folder}_{args.arch}_lr{args.lr}"
    if args.warm_start_epochs > 0:
        args.save_dir += f"_warm-{args.warm_start_epochs}"
    subset_size = args.train_frac
    args.save_dir += f"_train{subset_size:.2f}"
    if args.random_subset_size < 1.0:
        args.save_dir += (
            f"_random{args.random_subset_size:.2f}-start{args.partition_start}"
        )
    grd += (
        f"_dropevery{args.drop_interval}-loss{args.drop_thresh}"
        f"-watch{args.watch_interval}"
        if args.drop_learned
        else ""
    )
    args.save_dir += f"_batchsize{args.batch_size}_{grd}"
    if args.selection_method == "crest":
        args.save_dir += "_coreset" if args.approx_with_coreset else "_subset"
        args.save_dir += "_momentum" if args.approx_moment else ""

    args.save_dir += f"_seed_{args.seed}"

    return args.save_dir
