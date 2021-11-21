from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        PROJECT_PATH = "/home/jh/Final_Project/JH_web_demo_v2/inference/"

        self.parser.add_argument('--warp_checkpoint', type=str, default=PROJECT_PATH+'checkpoints/PFAFN/warp_model_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--gen_checkpoint', type=str, default=PROJECT_PATH+'checkpoints/PFAFN/gen_model_final.pth', help='load the pretrained model from the specified location')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

        self.isTrain = False
