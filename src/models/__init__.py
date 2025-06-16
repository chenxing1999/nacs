from .dcn import DCN_Mix, DCNv2
from .deepfm import DeepFM
from .final_mlp import FinalMLP
from .resnet18 import ResNet18
from .resnet20 import ResNet20


def get_model(dataset, arch):
    assert arch in ["dcnv2", "dcn_mix", "deepfm", "fmlp"]
    field_dims = dataset.dataset.field_dims

    if arch == "deepfm":
        model = DeepFM(field_dims, 10, [100, 100, 100], 0.5, True)
    elif arch == "dcnv2":
        model = DCNv2(field_dims, 10, [100, 100, 100])
    elif arch == "dcn_mix":
        model = DCN_Mix(field_dims, 10, [100, 100, 100])
    elif arch == "fmlp":
        model = FinalMLP(
            field_dims,
            10,
            [100, 100, 100],
            [100, 100, 100],
            num_heads=50,
        )
    else:
        raise ValueError()
    return model
