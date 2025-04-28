from utils.vgg_mod_model import VGG_MOD
from utils.resnet_model import ResNet
from utils.densenet_model import DenseNetCIFAR

class TrainingConf:
    available_attacks = ("SIG", "Badnet")
    mask_path = None
    attack_name = 'Badnet'
    model = DenseNetCIFAR