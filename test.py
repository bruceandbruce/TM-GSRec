# import os
# import numpy as np
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
# from options.test_options import TestOptions
# from data import CreateDataLoader
# from models import create_model
# from util1.visualizer import Visualizer
# from util1 import html

# from utils import utils_image as util

# import torch
# import cv2

# import lpips

# #python test.py --dataroot IXI_short.csv --name G1D20_continue1 --gpu_ids 0 --model resvit_one --which_model_netG GaussianSR_isbi --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 1908 --serial_batches --fineSize 256 --loadSize 256 --results_dir EXP_results/ --checkpoints_dir EXP_checkpoints --which_epoch 3 --pre_trained_resnet 0 --pre_trained_transformer 0 --mask G1D20


# # 后续考虑加入的指标
# # from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
# # from pytorch_msssim import ssim, ms_ssim

# '''
# python3 test_wyz.py --dataroot /home/wyz/Desktop/IXI/IXI_process_scripts/data_ixi_t2_pd.csv --name T2_PD_IXI_resvit_v2 --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 12000 --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest --pre_trained_path ./checkpoints/T2_PD_IXI_resvit_v2/latest_net_G.pth --pre_trained_resnet 0 --pre_trained_transformer 0
# python3 test_wyz.py --dataroot /home/wyz/Desktop/IXI/IXI_process_scripts/data_ixi_t2_pd.csv --name T2_PD_IXI_swinres_v0 --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 12000 --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest --pre_trained_path ./checkpoints/T2_PD_IXI_swinres_v0/latest_net_G.pth --pre_trained_resnet 0 --pre_trained_transformer 0
# '''

# if __name__ == '__main__':
#     opt = TestOptions().parse()
#     opt.nThreads = 1   # test code only supports nThreads = 1
#     opt.batchSize = 1  # test code only supports batchSize = 1
#     opt.serial_batches = True  # no shuffle
#     opt.no_flip = True  # no flip

#     opt.phase = 'test'
# #    opt.phase = 'valid'
#     data_loader = CreateDataLoader(opt)
#     dataset = data_loader.load_data()
#     model = create_model(opt)
    
# #    visualizer = Visualizer(opt)
#     #opt.phase = 'test'

# ##################################################################################################
# ##################################################################################################

#     # 验证数据个数是否相等
#     dataset_size_test = len(dataset)
#     # print('\033[1;31m---test_wyz.py---\033[0m')
#     # print("dataset_size_test = ", dataset_size_test)
#     # dataset_size_test = 11008

# ##################################################################################################
# ##################################################################################################

#     mae_avg = np.zeros([opt.how_many])
#     psnr_avg = np.zeros([opt.how_many])
#     ssim_avg = np.zeros([opt.how_many])
#     lpips_avg = np.zeros([opt.how_many])

#     another_ssim_avg = np.zeros([opt.how_many])
#     lips_rec_avg = np.zeros([opt.how_many])

#     zf_mae_avg = np.zeros([opt.how_many])
#     zf_psnr_avg = np.zeros([opt.how_many])
#     zf_ssim_avg = np.zeros([opt.how_many])
#     zf_lpips_avg = np.zeros([opt.how_many])

#     another_zf_ssim_avg = np.zeros([opt.how_many])
#     lips_zf_avg = np.zeros([opt.how_many])
# #    mae_avg_normal = np.zeros([opt.how_many])
# #    psnr_avg_normal = np.zeros([opt.how_many])
# #    ssim_avg_normal = np.zeros([opt.how_many])

#     # create website
# #    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# #    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     loss_fn_alex = lpips.LPIPS(net='alex').to(device)

#     # test
#     for i, data in enumerate(dataset):
#         if i >= opt.how_many:
#             break

#         # A是输入的零填充图像，B是Ground Truth图像，C是辅助模态图像

#         model.set_input(data)
#         model.test()

#         # print('\033[1;31m---test_wyz.py---\033[0m')
#         # print("model.fake_B.shape = ", model.fake_B.shape)
#         # model.fake_B.shape =  torch.Size([1, 1, 256, 256])
#         # wyz

#         #########################################################################################

#         lpips_fake_im_ = model.fake_B
#         lpips_real_im_ = model.real_B
#         lpips_ZF_im = model.real_A

#         # 计算lpips值
#         # evaluate lpips
#         lpips_ = util.calculate_lpips_single(loss_fn_alex, lpips_real_im_, lpips_fake_im_)
#         lpips_ = lpips_.data.squeeze().float().cpu().numpy()
#         lips_rec_avg[i] = lpips_
#         # evaluate lpips zf
#         zf_lpips_ = util.calculate_lpips_single(loss_fn_alex, lpips_real_im_, lpips_ZF_im)
#         zf_lpips_ = zf_lpips_.data.squeeze().float().cpu().numpy()
#         lips_zf_avg[i] = zf_lpips_

#         #########################################################################################

#         # 重建的图像
#         fake_im_ = model.fake_B.squeeze(0).cpu().data.numpy()
#         # 真实的ground truth图像
#         real_im_ = model.real_B.squeeze(0).cpu().data.numpy()

#         #########################################################################################
#         # 零填充图像
#         ZF_im = model.real_A.squeeze(0).cpu().data.numpy()


#         # print('\033[1;31m---test_wyz.py---\033[0m')
#         # print("fake_im_.shape = ", fake_im_.shape)
#         # print("real_im_.shape = ", real_im_.shape)
#         # print("ZF_im.shape = ", ZF_im.shape)
#         # fake_im_.shape = (1, 256, 256)
#         # real_im_.shape = (1, 256, 256)
#         # ZF_im.shape = (1, 256, 256)

#         ssim_fake_im_ = fake_im_.squeeze(0)
#         ssim_real_im_ = real_im_.squeeze(0)
#         ssim_ZF_im = ZF_im.squeeze(0)

#         # print('\033[1;31m---test_wyz_v3.py---\033[0m')
#         # print("ssim_fake_im_.shape = ", ssim_fake_im_.shape)
#         # print("ssim_real_im_.shape = ", ssim_real_im_.shape)
#         # print("ssim_ZF_im.shape = ", ssim_ZF_im.shape)

#         # ssim_fake_im_.shape = (256, 256)
#         # ssim_real_im_.shape = (256, 256)
#         # ssim_ZF_im.shape = (256, 256)

#         #another_ssim_avg[i] = util.calculate_ssim_single(ssim_real_im_, ssim_fake_im_)
#         #another_zf_ssim_avg[i] = util.calculate_ssim_single(ssim_real_im_, ssim_ZF_im)

#         #########################################################################################

#         # 重建图像与真实图像的差值
#         mae_avg[i] = abs(fake_im_-real_im_).mean()
#         # 重建图像的PSNR
#         psnr_avg[i] = psnr(real_im_, fake_im_, data_range=1)
#         # 重建图像SSIM
#         ssim_avg[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_im_, (1, 2, 0)), data_range=1,channel_axis=2)
#         # print('\033[1;31m---test_wyz.py---\033[0m')
#         # print("psnr_avg = ", psnr_avg[i])

#         #########################################################################################

#         # ZF图像与真实图像的差值
#         zf_mae_avg[i] = abs(ZF_im-real_im_).mean()
#         # ZF图像的PSNR
#         zf_psnr_avg[i] = psnr(real_im_, ZF_im, data_range=1)
#         # ZF图像SSIM
#         zf_ssim_avg[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(ZF_im, (1, 2, 0)), data_range=1, channel_axis=2)

#         #########################################################################################

#         # 将fake_im_归一化

# #        fake_image_min = np.min(fake_im_)
# #        fake_image_max = np.max(fake_im_)
# #        fake_image_normal = (fake_im_ - fake_image_min) / (fake_image_max - fake_image_min)

# #        mae_avg_normal[i] = abs(fake_image_normal - real_im_).mean()
# #        psnr_avg_normal[i] = psnr(real_im_, fake_image_normal, data_range=1)
# #        ssim_avg_normal[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_image_normal, (1, 2, 0)), data_range=1, multichannel=True)


#         #########################################################################################
#         #########################################################################################

#         # 多模态图像与输入零填充图像

#         # 辅助模态图像
#         #real_assistant = model.real_C.squeeze(0).cpu().data.numpy()

#         # 输入零填重图像
#         input_im_ = model.real_A.squeeze(0).cpu().data.numpy()



#         # print('\033[1;31m---test_wyz_v2.py---\033[0m')
#         # print("fake_im_.shape = ", fake_im_.shape)
#         # print("real_im_.shape = ", real_im_.shape)
#         # print("real_assitant.shape = ", real_assistant.shape)
#         # fake_im_.shape = (1, 256, 256)
#         # real_im_.shape = (1, 256, 256)
#         # real_assitant.shape = (1, 256, 256)

#         #########################################################################################
#         #########################################################################################

#         # 获取图像名字以及slice id
#         img_path = model.get_image_paths()
#         slice_idx = model.get_slice_idx()  # wyz
#         volumn_name = img_path[0].split('/')[-1].split('.')[0]  # wyz
#         print('%04d: process image... %s-%d' % (i, volumn_name, slice_idx))  # wyz

#         #########################################################################################
#         #########################################################################################

#         # 存储图像
#         # print('\033[1;31m---test_wyz_v2.py---\033[0m')
#         # print("opt.results_dir = ", opt.results_dir)
#         # opt.results_dir = EXP_results/

#         # 零填充图像
#         isExists = os.path.exists(os.path.join(opt.results_dir, 'ZF'))
#         if not isExists:
#             os.makedirs(os.path.join(opt.results_dir, 'ZF'))
#         # GT图像
#         isExists = os.path.exists(os.path.join(opt.results_dir, 'GT'))
#         if not isExists:
#             os.makedirs(os.path.join(opt.results_dir, 'GT'))
#         # 重建图像
#         isExists = os.path.exists(os.path.join(opt.results_dir, 'Recon'))
#         if not isExists:
#             os.makedirs(os.path.join(opt.results_dir, 'Recon'))
#         # 零填充差值图像
#         isExists = os.path.exists(os.path.join(opt.results_dir, 'Different_zero'))
#         if not isExists:
#             os.makedirs(os.path.join(opt.results_dir, 'Different_zero'))
#         # 重建差值图像
#         isExists = os.path.exists(os.path.join(opt.results_dir, 'Different_rec'))
#         if not isExists:
#              os.makedirs(os.path.join(opt.results_dir, 'Different_rec'))
#         # 辅助图像
#         isExists = os.path.exists(os.path.join(opt.results_dir, 'Assist'))
#         if not isExists:
#             os.makedirs(os.path.join(opt.results_dir, 'Assist'))
#         #########################################################################################
#         #########################################################################################

#         # 数据压缩，为了存储数据

#         # 重建的图像
#         fake_im_for_save = fake_im_.squeeze(0)

#         # 真实的ground truth图像
#         real_im_for_save = real_im_.squeeze(0)

#         # 辅助模态图像
#         #real_assistant_for_save = real_assistant.squeeze(0)

#         # 输入的零填充图像
#         input_im_for_save = input_im_.squeeze(0)


#         #########################################################################################
#         #########################################################################################

#         # 重建的图像
#         fake_image_for_df = model.fake_B
#         # 真实的ground truth图像
#         real_image_for_df = model.real_B
#         # 输入零填重图像
#         input_image_for_df = model.real_A

#         diff_gen_x10 = torch.mul(torch.abs(torch.sub(real_image_for_df, fake_image_for_df)), 5)
#         diff_lq_x10 = torch.mul(torch.abs(torch.sub(real_image_for_df, input_image_for_df)), 5)

#         # print('\033[1;31m---test_wyz_v2.py---\033[0m')
#         # print("fake_im_for_save.shape = ", fake_im_for_save.shape)
#         # print("real_im_for_save.shape = ", real_im_for_save.shape)
#         # print("real_assistant_for_save.shape = ", real_assistant_for_save.shape)
#         # fake_im_for_save.shape = (256, 256)
#         # real_im_for_save.shape = (256, 256)
#         # real_assistant_for_save.shape = (256, 256)

#         input_im_for_save = (np.clip(input_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
#         fake_im_for_save = (np.clip(fake_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
#         real_im_for_save = (np.clip(real_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
#         #real_assistant_for_save = (np.clip(real_assistant_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8

#         #diff_gen_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_gen)), 10)
#         #diff_lq_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_lq)), 10)

#         diff_gen_x10 = diff_gen_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
#         diff_lq_x10 = diff_lq_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()

#         diff_gen_x10 = (diff_gen_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
#         diff_lq_x10 = (diff_lq_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8

#         diff_gen_x10_color = cv2.applyColorMap(diff_gen_x10, cv2.COLORMAP_JET)
#         diff_lq_x10_color = cv2.applyColorMap(diff_lq_x10, cv2.COLORMAP_JET)

#         #########################################################################################
#         #########################################################################################

#         # 图像名字
#         input_image_dir = os.path.join(opt.results_dir, 'ZF' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
#         fake_image_dir = os.path.join(opt.results_dir, 'Recon' + "/"+ str(volumn_name) + str(slice_idx.numpy()) + ".png")
#         real_image_dir = os.path.join(opt.results_dir, 'GT' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
#         #assistant_image_dir = os.path.join(opt.results_dir, 'Assist' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
#         diff_gen_image_dir = os.path.join(opt.results_dir, 'Different_rec' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
#         diff_lq_image_dir = os.path.join(opt.results_dir, 'Different_zero' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")

#         # 保存图像
#         cv2.imwrite(input_image_dir, input_im_for_save)
#         cv2.imwrite(fake_image_dir, fake_im_for_save)
#         cv2.imwrite(real_image_dir, real_im_for_save)
#         #cv2.imwrite(assistant_image_dir, real_assistant_for_save)
#         cv2.imwrite(diff_gen_image_dir, diff_gen_x10_color)
#         cv2.imwrite(diff_lq_image_dir, diff_lq_x10_color)

#         #########################################################################################
#         #########################################################################################

#         # 图像原始维度
#         # print('\033[1;31m---test_wyz_v2.py---\033[0m')
#         # print("fake_im_.shape = ", fake_im_.shape)
#         # print("real_im_.shape = ", real_im_.shape)
#         # print("real_assitant.shape = ", real_assistant.shape)
#         # fake_im_.shape = (1, 256, 256)
#         # real_im_.shape = (1, 256, 256)
#         # real_assitant.shape = (1, 256, 256)

#         # print('\033[1;31m---test_wyz_v2.py---\033[0m')
#         # print("fake_im_.max = ", np.max(fake_im_))
#         # print("fake_im_.min = ", np.min(fake_im_))
#         # print("real_im_.max = ", np.max(real_im_))
#         # print("real_im_.min = ", np.min(real_im_))
#         # print("real_assistant.max = ", np.max(real_assistant))
#         # print("real_assistant.min = ", np.min(real_assistant))

#         # fake_im_.max = 0.85009825
#         # fake_im_.min = -0.123904385
#         # real_im_.max = 1.0
#         # real_im_.min = 0.0
#         # real_assistant.max = 1.0
#         # real_assistant.min = 0.0

#         # 将图像压缩到2D

#         # fake_im_

#     mae = np.mean(mae_avg)
#     mae_std = np.std(mae_avg)
#     mean_psnr = np.mean(psnr_avg)
#     std_psnr = np.std(psnr_avg)
#     mean_ssim = np.mean(ssim_avg)
#     std_ssim = np.std(ssim_avg)
#     mean_another_ssim_avg = np.mean(another_ssim_avg)
#     another_std_ssim = np.std(another_ssim_avg)
#     mean_lips_rec_avg = np.mean(lips_rec_avg)
#     rec_std_lips = np.std(lips_rec_avg)
#     print('testing -  \n mae: %.4f, \n mae_std: %.4f, \n psnr_mean: %.4f, \n psnr_std:%.4f, \n ssim_mean: %.4f, \n ssim_std:%.4f, \n another_ssim:%.4f, \n another_ssim_std:%.4f, \n mean_lips_rec_avg:%.4f, \n rec_std_lips:%.4f' % (mae, mae_std, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_another_ssim_avg, another_std_ssim, mean_lips_rec_avg, rec_std_lips))


#     zf_mae = np.mean(zf_mae_avg)
#     zf_mae_std = np.std(zf_mae_avg)
#     zf_mean_psnr = np.mean(zf_psnr_avg)
#     zf_std_psnr = np.std(zf_psnr_avg)
#     zf_mean_ssim = np.mean(zf_ssim_avg)
#     zf_std_ssim = np.std(zf_ssim_avg)
#     mean_another_zf_ssim_avg = np.mean(another_zf_ssim_avg)
#     another_std_ssim = np.std(another_zf_ssim_avg)
#     mean_lips_zf_avg = np.mean(lips_zf_avg)
#     zf_std_lips = np.std(lips_zf_avg)
#     print('testing -  \n zf_mae: %.4f, \n zf_mae_std: %.4f, \n zf_psnr_mean: %.4f, \n zf_psnr_std:%.4f, \n zf_ssim_mean: %.4f, \n zf_ssim_std:%.4f, \n another_zf_ssim_mean:%.4f, \n another_zf_ssim_std:%.4f, \n mean_lips_zf_avg:%.4f, \n zf_std_lips:%.4f' % (zf_mae, zf_mae_std, zf_mean_psnr, zf_std_psnr, zf_mean_ssim, zf_std_ssim, mean_another_zf_ssim_avg, another_std_ssim, mean_lips_zf_avg, zf_std_lips))

import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util1.visualizer import Visualizer
from util1 import html

from utils import utils_image as util

import torch
import cv2

import lpips

#python test.py --dataroot IXI_short.csv --name G1D20_continue1 --gpu_ids 0 --model resvit_one --which_model_netG GaussianSR_isbi --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 1908 --serial_batches --fineSize 256 --loadSize 256 --results_dir EXP_results/ --checkpoints_dir EXP_checkpoints --which_epoch 3 --pre_trained_resnet 0 --pre_trained_transformer 0 --mask G1D20


# 后续考虑加入的指标
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
# from pytorch_msssim import ssim, ms_ssim

'''
python3 test_wyz.py --dataroot /home/wyz/Desktop/IXI/IXI_process_scripts/data_ixi_t2_pd.csv --name T2_PD_IXI_resvit_v2 --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 12000 --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest --pre_trained_path ./checkpoints/T2_PD_IXI_resvit_v2/latest_net_G.pth --pre_trained_resnet 0 --pre_trained_transformer 0
python3 test_wyz.py --dataroot /home/wyz/Desktop/IXI/IXI_process_scripts/data_ixi_t2_pd.csv --name T2_PD_IXI_swinres_v0 --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 12000 --serial_batches --fineSize 256 --loadSize 256 --results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest --pre_trained_path ./checkpoints/T2_PD_IXI_swinres_v0/latest_net_G.pth --pre_trained_resnet 0 --pre_trained_transformer 0
'''

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    opt.phase = 'test'
#    opt.phase = 'valid'
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    
#    visualizer = Visualizer(opt)
    #opt.phase = 'test'

##################################################################################################
##################################################################################################

    # 验证数据个数是否相等
    dataset_size_test = len(dataset)
    # print('\033[1;31m---test_wyz.py---\033[0m')
    # print("dataset_size_test = ", dataset_size_test)
    # dataset_size_test = 11008

##################################################################################################
##################################################################################################

    mae_avg = np.zeros([opt.how_many])
    psnr_avg = np.zeros([opt.how_many])
    ssim_avg = np.zeros([opt.how_many])
    lpips_avg = np.zeros([opt.how_many])

    another_ssim_avg = np.zeros([opt.how_many])
    lips_rec_avg = np.zeros([opt.how_many])

    zf_mae_avg = np.zeros([opt.how_many])
    zf_psnr_avg = np.zeros([opt.how_many])
    zf_ssim_avg = np.zeros([opt.how_many])
    zf_lpips_avg = np.zeros([opt.how_many])

    another_zf_ssim_avg = np.zeros([opt.how_many])
    lips_zf_avg = np.zeros([opt.how_many])
#    mae_avg_normal = np.zeros([opt.how_many])
#    psnr_avg_normal = np.zeros([opt.how_many])
#    ssim_avg_normal = np.zeros([opt.how_many])

    # create website
#    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
#    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        # A是输入的零填充图像，B是Ground Truth图像，C是辅助模态图像

        model.set_input(data)
        model.test()

        # print('\033[1;31m---test_wyz.py---\033[0m')
        # print("model.fake_B.shape = ", model.fake_B.shape)
        # model.fake_B.shape =  torch.Size([1, 1, 256, 256])
        # wyz

        #########################################################################################

        lpips_fake_im_ = model.fake_B
        lpips_real_im_ = model.real_B
        lpips_ZF_im = model.real_A

        # 计算lpips值
        # evaluate lpips
        lpips_ = util.calculate_lpips_single(loss_fn_alex, lpips_real_im_, lpips_fake_im_)
        lpips_ = lpips_.data.squeeze().float().cpu().numpy()
        lips_rec_avg[i] = lpips_
        # evaluate lpips zf
        zf_lpips_ = util.calculate_lpips_single(loss_fn_alex, lpips_real_im_, lpips_ZF_im)
        zf_lpips_ = zf_lpips_.data.squeeze().float().cpu().numpy()
        lips_zf_avg[i] = zf_lpips_

        #########################################################################################

        # 重建的图像
        fake_im_ = model.fake_B.squeeze(0).cpu().data.numpy()
        # 真实的ground truth图像
        real_im_ = model.real_B.squeeze(0).cpu().data.numpy()

        #########################################################################################
        # 零填充图像
        ZF_im = model.real_A.squeeze(0).cpu().data.numpy()


        # print('\033[1;31m---test_wyz.py---\033[0m')
        # print("fake_im_.shape = ", fake_im_.shape)
        # print("real_im_.shape = ", real_im_.shape)
        # print("ZF_im.shape = ", ZF_im.shape)
        # fake_im_.shape = (1, 256, 256)
        # real_im_.shape = (1, 256, 256)
        # ZF_im.shape = (1, 256, 256)

        ssim_fake_im_ = fake_im_.squeeze(0)
        ssim_real_im_ = real_im_.squeeze(0)
        ssim_ZF_im = ZF_im.squeeze(0)

        # print('\033[1;31m---test_wyz_v3.py---\033[0m')
        # print("ssim_fake_im_.shape = ", ssim_fake_im_.shape)
        # print("ssim_real_im_.shape = ", ssim_real_im_.shape)
        # print("ssim_ZF_im.shape = ", ssim_ZF_im.shape)

        # ssim_fake_im_.shape = (256, 256)
        # ssim_real_im_.shape = (256, 256)
        # ssim_ZF_im.shape = (256, 256)

        #another_ssim_avg[i] = util.calculate_ssim_single(ssim_real_im_, ssim_fake_im_)
        #another_zf_ssim_avg[i] = util.calculate_ssim_single(ssim_real_im_, ssim_ZF_im)

        #########################################################################################

        # 重建图像与真实图像的差值
        mae_avg[i] = abs(fake_im_-real_im_).mean()
        # 重建图像的PSNR
        psnr_avg[i] = psnr(real_im_, fake_im_, data_range=1)
        # 重建图像SSIM
        ssim_avg[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_im_, (1, 2, 0)), data_range=1,channel_axis=2)
        # print('\033[1;31m---test_wyz.py---\033[0m')
        # print("psnr_avg = ", psnr_avg[i])

        #########################################################################################

        # ZF图像与真实图像的差值
        zf_mae_avg[i] = abs(ZF_im-real_im_).mean()
        # ZF图像的PSNR
        zf_psnr_avg[i] = psnr(real_im_, ZF_im, data_range=1)
        # ZF图像SSIM
        zf_ssim_avg[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(ZF_im, (1, 2, 0)), data_range=1, channel_axis=2)

        #########################################################################################

        # 将fake_im_归一化

#        fake_image_min = np.min(fake_im_)
#        fake_image_max = np.max(fake_im_)
#        fake_image_normal = (fake_im_ - fake_image_min) / (fake_image_max - fake_image_min)

#        mae_avg_normal[i] = abs(fake_image_normal - real_im_).mean()
#        psnr_avg_normal[i] = psnr(real_im_, fake_image_normal, data_range=1)
#        ssim_avg_normal[i] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_image_normal, (1, 2, 0)), data_range=1, multichannel=True)


        #########################################################################################
        #########################################################################################

        # 多模态图像与输入零填充图像

        # 辅助模态图像
        #real_assistant = model.real_C.squeeze(0).cpu().data.numpy()

        # 输入零填重图像
        input_im_ = model.real_A.squeeze(0).cpu().data.numpy()



        # print('\033[1;31m---test_wyz_v2.py---\033[0m')
        # print("fake_im_.shape = ", fake_im_.shape)
        # print("real_im_.shape = ", real_im_.shape)
        # print("real_assitant.shape = ", real_assistant.shape)
        # fake_im_.shape = (1, 256, 256)
        # real_im_.shape = (1, 256, 256)
        # real_assitant.shape = (1, 256, 256)

        #########################################################################################
        #########################################################################################

        # 获取图像名字以及slice id
        img_path = model.get_image_paths()
        slice_idx = model.get_slice_idx()  # wyz
        volumn_name = img_path[0].split('/')[-1].split('.')[0]  # wyz
        print('%04d: process image... %s-%d' % (i, volumn_name, slice_idx.numpy()))  # wyz

        #########################################################################################
        #########################################################################################

        # 存储图像
        # print('\033[1;31m---test_wyz_v2.py---\033[0m')
        # print("opt.results_dir = ", opt.results_dir)
        # opt.results_dir = EXP_results/

        # 零填充图像
        isExists = os.path.exists(os.path.join(opt.results_dir, 'ZF'))
        if not isExists:
            os.makedirs(os.path.join(opt.results_dir, 'ZF'))
        # GT图像
        isExists = os.path.exists(os.path.join(opt.results_dir, 'GT'))
        if not isExists:
            os.makedirs(os.path.join(opt.results_dir, 'GT'))
        # 重建图像
        isExists = os.path.exists(os.path.join(opt.results_dir, 'Recon'))
        if not isExists:
            os.makedirs(os.path.join(opt.results_dir, 'Recon'))
        # 零填充差值图像
        isExists = os.path.exists(os.path.join(opt.results_dir, 'Different_zero'))
        if not isExists:
            os.makedirs(os.path.join(opt.results_dir, 'Different_zero'))
        # 重建差值图像
        isExists = os.path.exists(os.path.join(opt.results_dir, 'Different_rec'))
        if not isExists:
             os.makedirs(os.path.join(opt.results_dir, 'Different_rec'))
        # 辅助图像
        isExists = os.path.exists(os.path.join(opt.results_dir, 'Assist'))
        if not isExists:
            os.makedirs(os.path.join(opt.results_dir, 'Assist'))
        #########################################################################################
        #########################################################################################

        # 数据压缩，为了存储数据

        # 重建的图像
        fake_im_for_save = fake_im_.squeeze(0)

        # 真实的ground truth图像
        real_im_for_save = real_im_.squeeze(0)

        # 辅助模态图像
        #real_assistant_for_save = real_assistant.squeeze(0)

        # 输入的零填充图像
        input_im_for_save = input_im_.squeeze(0)


        #########################################################################################
        #########################################################################################

        # 重建的图像
        fake_image_for_df = model.fake_B
        # 真实的ground truth图像
        real_image_for_df = model.real_B
        # 输入零填重图像
        input_image_for_df = model.real_A

        diff_gen_x10 = torch.mul(torch.abs(torch.sub(real_image_for_df, fake_image_for_df)), 5)
        diff_lq_x10 = torch.mul(torch.abs(torch.sub(real_image_for_df, input_image_for_df)), 5)

        # print('\033[1;31m---test_wyz_v2.py---\033[0m')
        # print("fake_im_for_save.shape = ", fake_im_for_save.shape)
        # print("real_im_for_save.shape = ", real_im_for_save.shape)
        # print("real_assistant_for_save.shape = ", real_assistant_for_save.shape)
        # fake_im_for_save.shape = (256, 256)
        # real_im_for_save.shape = (256, 256)
        # real_assistant_for_save.shape = (256, 256)

        input_im_for_save = (np.clip(input_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        fake_im_for_save = (np.clip(fake_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        real_im_for_save = (np.clip(real_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8
        #real_assistant_for_save = (np.clip(real_assistant_for_save, 0, 1) * 255.0).round().astype(np.uint8)  # float32 to uint8

        #diff_gen_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_gen)), 10)
        #diff_lq_x10 = torch.mul(torch.abs(torch.sub(img_gt, img_lq)), 10)

        diff_gen_x10 = diff_gen_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_lq_x10 = diff_lq_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()

        diff_gen_x10 = (diff_gen_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8
        diff_lq_x10 = (diff_lq_x10 * 255.0).round().astype(np.uint8)  # float32 to uint8

        diff_gen_x10_color = cv2.applyColorMap(diff_gen_x10, cv2.COLORMAP_JET)
        diff_lq_x10_color = cv2.applyColorMap(diff_lq_x10, cv2.COLORMAP_JET)

        #########################################################################################
        #########################################################################################

        # 图像名字
        input_image_dir = os.path.join(opt.results_dir, 'ZF' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
        fake_image_dir = os.path.join(opt.results_dir, 'Recon' + "/"+ str(volumn_name) + str(slice_idx.numpy()) + ".png")
        real_image_dir = os.path.join(opt.results_dir, 'GT' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
        #assistant_image_dir = os.path.join(opt.results_dir, 'Assist' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
        diff_gen_image_dir = os.path.join(opt.results_dir, 'Different_rec' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")
        diff_lq_image_dir = os.path.join(opt.results_dir, 'Different_zero' + "/" + str(volumn_name) + str(slice_idx.numpy()) + ".png")

        # 保存图像
        cv2.imwrite(input_image_dir, input_im_for_save)
        cv2.imwrite(fake_image_dir, fake_im_for_save)
        cv2.imwrite(real_image_dir, real_im_for_save)
        #cv2.imwrite(assistant_image_dir, real_assistant_for_save)
        cv2.imwrite(diff_gen_image_dir, diff_gen_x10_color)
        cv2.imwrite(diff_lq_image_dir, diff_lq_x10_color)

        #########################################################################################
        #########################################################################################

        # 图像原始维度
        # print('\033[1;31m---test_wyz_v2.py---\033[0m')
        # print("fake_im_.shape = ", fake_im_.shape)
        # print("real_im_.shape = ", real_im_.shape)
        # print("real_assitant.shape = ", real_assistant.shape)
        # fake_im_.shape = (1, 256, 256)
        # real_im_.shape = (1, 256, 256)
        # real_assitant.shape = (1, 256, 256)

        # print('\033[1;31m---test_wyz_v2.py---\033[0m')
        # print("fake_im_.max = ", np.max(fake_im_))
        # print("fake_im_.min = ", np.min(fake_im_))
        # print("real_im_.max = ", np.max(real_im_))
        # print("real_im_.min = ", np.min(real_im_))
        # print("real_assistant.max = ", np.max(real_assistant))
        # print("real_assistant.min = ", np.min(real_assistant))

        # fake_im_.max = 0.85009825
        # fake_im_.min = -0.123904385
        # real_im_.max = 1.0
        # real_im_.min = 0.0
        # real_assistant.max = 1.0
        # real_assistant.min = 0.0

        # 将图像压缩到2D

        # fake_im_

    mae = np.mean(mae_avg)
    mae_std = np.std(mae_avg)
    mean_psnr = np.mean(psnr_avg)
    std_psnr = np.std(psnr_avg)
    mean_ssim = np.mean(ssim_avg)
    std_ssim = np.std(ssim_avg)
    mean_another_ssim_avg = np.mean(another_ssim_avg)
    another_std_ssim = np.std(another_ssim_avg)
    mean_lips_rec_avg = np.mean(lips_rec_avg)
    rec_std_lips = np.std(lips_rec_avg)
    print('testing -  \n mae: %.4f, \n mae_std: %.4f, \n psnr_mean: %.4f, \n psnr_std:%.4f, \n ssim_mean: %.4f, \n ssim_std:%.4f, \n another_ssim:%.4f, \n another_ssim_std:%.4f, \n mean_lips_rec_avg:%.4f, \n rec_std_lips:%.4f' % (mae, mae_std, mean_psnr, std_psnr, mean_ssim, std_ssim, mean_another_ssim_avg, another_std_ssim, mean_lips_rec_avg, rec_std_lips))


    zf_mae = np.mean(zf_mae_avg)
    zf_mae_std = np.std(zf_mae_avg)
    zf_mean_psnr = np.mean(zf_psnr_avg)
    zf_std_psnr = np.std(zf_psnr_avg)
    zf_mean_ssim = np.mean(zf_ssim_avg)
    zf_std_ssim = np.std(zf_ssim_avg)
    mean_another_zf_ssim_avg = np.mean(another_zf_ssim_avg)
    another_std_ssim = np.std(another_zf_ssim_avg)
    mean_lips_zf_avg = np.mean(lips_zf_avg)
    zf_std_lips = np.std(lips_zf_avg)
    print('testing -  \n zf_mae: %.4f, \n zf_mae_std: %.4f, \n zf_psnr_mean: %.4f, \n zf_psnr_std:%.4f, \n zf_ssim_mean: %.4f, \n zf_ssim_std:%.4f, \n another_zf_ssim_mean:%.4f, \n another_zf_ssim_std:%.4f, \n mean_lips_zf_avg:%.4f, \n zf_std_lips:%.4f' % (zf_mae, zf_mae_std, zf_mean_psnr, zf_std_psnr, zf_mean_ssim, zf_std_ssim, mean_another_zf_ssim_avg, another_std_ssim, mean_lips_zf_avg, zf_std_lips))


# ====================== 以下是为GUI添加的导出函数（完全不影响原有命令行功能） ======================
def run_test_gui(
    dataroot,
    checkpoints_dir,
    results_dir,
    name="G1D20_continue1",
    which_epoch="3",
    mask="G1D20",
    gpu_ids="0",
    how_many=1908,
    progress_callback=None,
    log_callback=None
):
    """
    供GUI调用的多模态磁共振图像重建测试函数
    所有参数与原命令行参数一一对应
    """
    from options.test_options import TestOptions
    from data import CreateDataLoader
    from models import create_model
    import lpips
    import numpy as np
    import os
    import cv2
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from utils import utils_image as util
    import torch

    # 1. 创建并配置测试选项（完全复制原有的opt设置）
    opt = TestOptions().parse(args=[])  # 关键：传入空列表，不解析命令行参数
    
    # 基础固定参数（与原命令行一致，不需要用户修改）
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.phase = 'test'
    opt.model = 'resvit_one'
    opt.which_model_netG = 'GaussianSR_isbi'  # 如果你用的是resvit，这里改成'resvit'
    opt.dataset_mode = 'aligned2'
    opt.norm = 'batch'
    opt.output_nc = 1
    opt.input_nc = 1
    opt.fineSize = 256
    opt.loadSize = 256
    opt.pre_trained_resnet = 0
    opt.pre_trained_transformer = 0
    
    # 从GUI传入的动态参数
    opt.dataroot = dataroot
    opt.checkpoints_dir = checkpoints_dir
    opt.results_dir = results_dir
    opt.name = name
    opt.which_epoch = which_epoch
    opt.mask = mask
    opt.gpu_ids = gpu_ids
    opt.how_many = how_many
    
    # 2. 初始化日志
    if log_callback:
        log_callback("="*60)
        log_callback("多模态磁共振图像重建任务开始")
        log_callback(f"数据集路径: {dataroot}")
        log_callback(f"模型目录: {checkpoints_dir}/{name}")
        log_callback(f"模型版本: epoch_{which_epoch}")
        log_callback(f"欠采样掩码: {mask}")
        log_callback(f"结果保存路径: {results_dir}")
        log_callback(f"GPU设备: {gpu_ids}")
        log_callback(f"测试样本数: {how_many}")
        log_callback("="*60)
    
    # 3. 创建数据加载器
    if log_callback:
        log_callback("正在加载测试数据集...")
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(dataset)
    if log_callback:
        log_callback(f"数据集加载完成，共{dataset_size}个样本")
    
    # 4. 创建并加载模型
    if log_callback:
        log_callback("正在加载重建模型...")
    model = create_model(opt)
    if log_callback:
        log_callback("模型加载成功")
    
    # 5. 初始化指标数组
    mae_avg = np.zeros(opt.how_many)
    psnr_avg = np.zeros(opt.how_many)
    ssim_avg = np.zeros(opt.how_many)
    lips_rec_avg = np.zeros(opt.how_many)
    
    zf_mae_avg = np.zeros(opt.how_many)
    zf_psnr_avg = np.zeros(opt.how_many)
    zf_ssim_avg = np.zeros(opt.how_many)
    lips_zf_avg = np.zeros(opt.how_many)
    
    # 6. 初始化LPIPS指标
    device = torch.device(f'cuda:{gpu_ids}' if torch.cuda.is_available() and gpu_ids != '-1' else 'cpu')
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    # 7. 创建结果保存目录
    os.makedirs(os.path.join(results_dir, 'ZF'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'GT'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'Recon'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'Different_zero'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'Different_rec'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'Assist'), exist_ok=True)
    
    # 8. 执行重建测试
    result_files = []
    if log_callback:
        log_callback("开始执行图像重建...")
    
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        
        # 模型前向传播
        model.set_input(data)
        model.test()
        
        # 获取图像数据
        fake_im_ = model.fake_B.squeeze(0).cpu().data.numpy()
        real_im_ = model.real_B.squeeze(0).cpu().data.numpy()
        ZF_im = model.real_A.squeeze(0).cpu().data.numpy()
        
        # 计算LPIPS指标
        lpips_fake_im_ = model.fake_B
        lpips_real_im_ = model.real_B
        lpips_ZF_im = model.real_A
        
        lpips_ = util.calculate_lpips_single(loss_fn_alex, lpips_real_im_, lpips_fake_im_)
        lips_rec_avg[i] = lpips_.data.squeeze().float().cpu().numpy()
        
        zf_lpips_ = util.calculate_lpips_single(loss_fn_alex, lpips_real_im_, lpips_ZF_im)
        lips_zf_avg[i] = zf_lpips_.data.squeeze().float().cpu().numpy()
        
        # 计算MAE、PSNR、SSIM指标
        mae_avg[i] = abs(fake_im_ - real_im_).mean()
        psnr_avg[i] = psnr(real_im_, fake_im_, data_range=1)
        ssim_avg[i] = ssim(
            np.transpose(real_im_, (1, 2, 0)), 
            np.transpose(fake_im_, (1, 2, 0)), 
            data_range=1, 
            channel_axis=2
        )
        
        zf_mae_avg[i] = abs(ZF_im - real_im_).mean()
        zf_psnr_avg[i] = psnr(real_im_, ZF_im, data_range=1)
        zf_ssim_avg[i] = ssim(
            np.transpose(real_im_, (1, 2, 0)), 
            np.transpose(ZF_im, (1, 2, 0)), 
            data_range=1, 
            channel_axis=2
        )
        
        # 获取图像信息
        img_path = model.get_image_paths()
        slice_idx = model.get_slice_idx()
        volumn_name = img_path[0].split('/')[-1].split('.')[0]
        file_name = f"{volumn_name}{slice_idx.numpy()}.png"
        
        if log_callback and i % 100 == 0:  # 每100张输出一次日志
            log_callback(f"正在处理第{i+1}/{opt.how_many}张图像: {file_name}")
        
        # 保存图像（完全复制原有的保存逻辑）
        fake_im_for_save = fake_im_.squeeze(0)
        real_im_for_save = real_im_.squeeze(0)
        input_im_for_save = ZF_im.squeeze(0)
        
        fake_image_for_df = model.fake_B
        real_image_for_df = model.real_B
        input_image_for_df = model.real_A
        
        diff_gen_x10 = torch.mul(torch.abs(torch.sub(real_image_for_df, fake_image_for_df)), 5)
        diff_lq_x10 = torch.mul(torch.abs(torch.sub(real_image_for_df, input_image_for_df)), 5)
        
        input_im_for_save = (np.clip(input_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)
        fake_im_for_save = (np.clip(fake_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)
        real_im_for_save = (np.clip(real_im_for_save, 0, 1) * 255.0).round().astype(np.uint8)
        
        diff_gen_x10 = diff_gen_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        diff_lq_x10 = diff_lq_x10.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        
        diff_gen_x10 = (diff_gen_x10 * 255.0).round().astype(np.uint8)
        diff_lq_x10 = (diff_lq_x10 * 255.0).round().astype(np.uint8)
        
        diff_gen_x10_color = cv2.applyColorMap(diff_gen_x10, cv2.COLORMAP_JET)
        diff_lq_x10_color = cv2.applyColorMap(diff_lq_x10, cv2.COLORMAP_JET)
        
        # 保存所有图像
        input_image_dir = os.path.join(results_dir, 'ZF', file_name)
        fake_image_dir = os.path.join(results_dir, 'Recon', file_name)
        real_image_dir = os.path.join(results_dir, 'GT', file_name)
        diff_gen_image_dir = os.path.join(results_dir, 'Different_rec', file_name)
        diff_lq_image_dir = os.path.join(results_dir, 'Different_zero', file_name)
        
        cv2.imwrite(input_image_dir, input_im_for_save)
        cv2.imwrite(fake_image_dir, fake_im_for_save)
        cv2.imwrite(real_image_dir, real_im_for_save)
        cv2.imwrite(diff_gen_image_dir, diff_gen_x10_color)
        cv2.imwrite(diff_lq_image_dir, diff_lq_x10_color)
        
        # 记录结果文件（只记录重建图像用于GUI展示）
        result_files.append((file_name, fake_image_dir))
        
        # 更新进度条
        if progress_callback:
            progress = int((i + 1) / opt.how_many * 100)
            progress_callback(progress)
    
    # 9. 计算并输出平均指标
    mae = np.mean(mae_avg)
    mae_std = np.std(mae_avg)
    mean_psnr = np.mean(psnr_avg)
    std_psnr = np.std(psnr_avg)
    mean_ssim = np.mean(ssim_avg)
    std_ssim = np.std(ssim_avg)
    mean_lips = np.mean(lips_rec_avg)
    std_lips = np.std(lips_rec_avg)
    
    zf_mae = np.mean(zf_mae_avg)
    zf_mae_std = np.std(zf_mae_avg)
    zf_mean_psnr = np.mean(zf_psnr_avg)
    zf_std_psnr = np.std(zf_psnr_avg)
    zf_mean_ssim = np.mean(zf_ssim_avg)
    zf_std_ssim = np.std(zf_ssim_avg)
    zf_mean_lips = np.mean(lips_zf_avg)
    zf_std_lips = np.std(lips_zf_avg)
    
    if log_callback:
        log_callback("="*60)
        log_callback("重建任务完成！")
        log_callback("="*60)
        log_callback("【重建结果指标】")
        log_callback(f"MAE: {mae:.4f} ± {mae_std:.4f}")
        log_callback(f"PSNR: {mean_psnr:.4f} ± {std_psnr:.4f} dB")
        log_callback(f"SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")
        log_callback(f"LPIPS: {mean_lips:.4f} ± {std_lips:.4f}")
        log_callback("-"*60)
        log_callback("【零填充(ZF)基准指标】")
        log_callback(f"MAE: {zf_mae:.4f} ± {zf_mae_std:.4f}")
        log_callback(f"PSNR: {zf_mean_psnr:.4f} ± {zf_std_psnr:.4f} dB")
        log_callback(f"SSIM: {zf_mean_ssim:.4f} ± {zf_std_ssim:.4f}")
        log_callback(f"LPIPS: {zf_mean_lips:.4f} ± {zf_std_lips:.4f}")
        log_callback("="*60)
    
    # 10. 返回结果
    return {
        "result_files": result_files,
        "metrics": {
            "recon": {
                "mae": mae,
                "mae_std": mae_std,
                "psnr": mean_psnr,
                "psnr_std": std_psnr,
                "ssim": mean_ssim,
                "ssim_std": std_ssim,
                "lpips": mean_lips,
                "lpips_std": std_lips
            },
            "zf": {
                "mae": zf_mae,
                "mae_std": zf_mae_std,
                "psnr": zf_mean_psnr,
                "psnr_std": zf_std_psnr,
                "ssim": zf_mean_ssim,
                "ssim_std": zf_std_ssim,
                "lpips": zf_mean_lips,
                "lpips_std": zf_std_lips
            }
        }
    }