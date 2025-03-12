import argparse



class ArgsParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
    
    def set_arguments(self):
        parser = self.parser

        parser.add_argument('--network', default='vgg16_bn', type=str)
        parser.add_argument('--depth', default=19, type=int)
        parser.add_argument('--dataset', default='cifar10', type=str)

        # densenet
        parser.add_argument('--growthRate', default=12, type=int)
        parser.add_argument('--compressionRate', default=2, type=int)

        # wrn, densenet
        parser.add_argument('--widen_factor', default=1, type=int)
        parser.add_argument('--dropRate', default=0.0, type=float)


        parser.add_argument('--device', default='cuda', type=str)
        parser.add_argument('--resume', '-r', action='store_true')
        parser.add_argument('--load_path', default='', type=str)
        parser.add_argument('--log_dir', default='runs/pretrain', type=str)


        parser.add_argument('--optimizer', default='kfac', type=str)
        parser.add_argument('--batch_size', default=64, type=float)
        parser.add_argument('--epoch', default=1, type=int)
        parser.add_argument('--milestone', default=None, type=str)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--momentum', default=0.9, type=float)
        parser.add_argument('--stat_decay', default=0.95, type=float)
        parser.add_argument('--damping', default=1e-3, type=float)
        parser.add_argument('--kl_clip', default=1e-2, type=float)
        parser.add_argument('--weight_decay', default=3e-3, type=float)
        parser.add_argument('--TCov', default=10, type=int)
        parser.add_argument('--TScal', default=10, type=int)
        parser.add_argument('--TInv', default=100, type=int)

        parser.add_argument('--precision', default="float32", type=str)
        parser.add_argument('--approx', default="diagonal", type=str)
        parser.add_argument('--loss', default="bce", type=str)
        parser.add_argument("--save", default=False, type=bool, help="Save kronecker factors to prior")
        parser.add_argument("--evaluate", default=True, type=bool, help="Compute final PAC-Bayes bound")
        

        parser.add_argument('--prefix', default=None, type=str)
        
    def parse_args(self):
        return self.parser.parse_args()