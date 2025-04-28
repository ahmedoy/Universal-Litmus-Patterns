from utils.vgg_mod_model import VGG_MOD
from utils.resnet_model import ResNet

class TrainingConf:
    mask_path = None # none for attacks where the mask isn't pregerenerated
    attack_name = 'SIG'
    model = VGG_MOD