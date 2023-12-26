import argparse
import datetime
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader

import Util
from FusionNet import FusionNet
from LowerLightEnhance.LowerLightEnhanceNetwork import LowerLightEnhanceNetwork
from LowerLightEnhance.multi_read_data import MemoryFriendlyLoader
from TaskFusion_dataset import Fusion_dataset
from loss import Fusionloss
from trainEnlightFusion import saveTrainOrTestEnlightenImageY


def testTAIFT(type='test'):
    fusion_model_path = './model/Fusion/fusion_model.pth'
    fused_dir = args.fusionResultPath
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)
    tAIFTModel = FusionNet(output=1)
    tAIFTModel.eval()
    tAIFTModel.cuda()
    tAIFTModel.load_state_dict(torch.load(fusion_model_path))
    print('TAIFT load_stat_dict done!')
    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    with torch.no_grad():
        for it, (images_vis, images_ir, name) in enumerate(test_loader):
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)

            if args.gpu >= 0:
                images_vis = images_vis.cuda(args.gpu)
                images_ir = images_ir.cuda(args.gpu)

            images_vis_ycrcb = Util.RGB2YCrCb(images_vis)
            logits, binary_out = tAIFTModel(images_vis_ycrcb,
                                            images_ir)
            saveTAFIMOutput(fused_dir, images_vis_ycrcb, logits, name)
    pass


def saveTAFIMOutput(fused_dir, images_vis_ycrcb, logits, name):
    fusion_ycrcb = torch.cat(
        (logits, images_vis_ycrcb[:, 1:2, :,
                 :], images_vis_ycrcb[:, 2:, :, :]),
        dim=1,
    )
    fusion_image = Util.YCrCb2RGB(fusion_ycrcb)
    ones = torch.ones_like(fusion_image)
    zeros = torch.zeros_like(fusion_image)
    fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
    fusion_image = torch.where(
        fusion_image < zeros, zeros, fusion_image)
    fused_image = fusion_image.cpu().numpy()
    fused_image = fused_image.transpose((0, 2, 3, 1))
    fused_image = (fused_image - np.min(fused_image)) / (
            np.max(fused_image) - np.min(fused_image)
    )
    fused_image = np.uint8(255.0 * fused_image)
    for k in range(len(name)):
        image = fused_image[k, :, :, :]
        image = image.squeeze()
        image = Image.fromarray(image)
        save_path = os.path.join(fused_dir, name[k])
        image.save(save_path)
        print('TAIFT {0} Sucessfully!'.format(save_path))


def test_LLIET():
    lLIETModel = LowerLightEnhanceNetwork(stage=args.stage)
    lLIETModel.eval()
    lLIETModel.cuda()
    lLIETModel.load_state_dict(torch.load(args.inlightmodel))
    print('load LLIET dict done!')

    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')
    enlight_test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=False, num_workers=0)

    lLIETModel.eval()
    with torch.no_grad():
        for _, (input, image_name) in enumerate(enlight_test_queue):
            with torch.no_grad():
                input = Variable(input).cuda()
            inputTrain_ycrcb = Util.RGB2YCrCb(input)
            inputTrainY = inputTrain_ycrcb[:, :1]
            illu_list, ref_list, input_list, atten = lLIETModel(inputTrainY)
            saveTrainOrTestEnlightenImageY(0, image_name, inputTrain_ycrcb, ref_list)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='EnlightFusion')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3, help='epochs')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--stage', type=int, default=3, help='epochs')
    parser.add_argument('--data_path', type=str, default='./MSRS/Visible/test/MSRS',
                        help='location of the data corpus')

    parser.add_argument('--inlightmodel', type=str, default='./lianHeXuLian/EnlightOut/Model/enhanceVisMode.pt',
                        help='location of the data corpus')
    parser.add_argument('--fusionResultPath', type=str, default='./lianHeXuLian/EnlightOut/MSRSFusion',
                        help='location of the data corpus')
    args = parser.parse_args()
    test_LLIET()
    testTAIFT()
