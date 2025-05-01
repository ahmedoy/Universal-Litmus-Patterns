from utils.vgg_mod_model import VGG_MOD
from utils.resnet_model import ResNet
from utils.densenet_model import DenseNetCIFAR
from utils.vit_model import load_vit, ViTSmall14

class TrainingConf:
    available_attacks = ("SIG", "Badnet")
    mask_path = None
    attack_name = 'Badnet'
    model = load_vit
    architecture_name = ViTSmall14.architecture_name