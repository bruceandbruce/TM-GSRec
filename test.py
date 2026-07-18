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

#python test.py --dataroot IXI_short.csv --name G1D20 --gpu_ids 0 --model resvit_one --which_model_netG GaussianSR_isbi --dataset_mode aligned2 --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 1908 --serial_batches --fineSize 256 --loadSize 256 --results_dir EXP_results/ --checkpoints_dir EXP_checkpoints --which_epoch 5 --pre_trained_resnet 0 --pre_trained_transformer 0 --mask G1D20


# 后续考虑加入的指标
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
# from pytorch_msssim import ssim, ms_ssim



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
    # ---------- 在这里插入上面的代码 ----------
    netG = model.netG
    netG.eval()
    total_params = sum(p.numel() for p in netG.parameters())


    print(f"生成器总参数量: {total_params:,}")
    from thop import profile, clever_format
    dummy_input0 = torch.randn(1, 2, 256, 256).contiguous()
    dummy_input1 = torch.randn(1, 2, 256, 256).contiguous()
    dummy_input2 = torch.randn(1, 2, 256, 256).contiguous()
    dummy_input3 = torch.randn(1, 65536,2)
    dummy_input4 = torch.tensor([1]) 
    print(dummy_input4.shape)
    dummy_input5 = torch.randn(1, 65536,2)
    if opt.gpu_ids:
        dummy_input0 = dummy_input0.to(opt.gpu_ids[0])
        dummy_input1 = dummy_input1.to(opt.gpu_ids[0])
        dummy_input2 = dummy_input2.to(opt.gpu_ids[0])
        dummy_input3 = dummy_input3.to(opt.gpu_ids[0])
        dummy_input4 = dummy_input4.to(opt.gpu_ids[0])
        dummy_input5 = dummy_input5.to(opt.gpu_ids[0])
    flops, params = profile(netG, inputs=(dummy_input0,dummy_input1,dummy_input2,dummy_input3,dummy_input4,dummy_input5), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}, 参数量(thop): {params}")
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
        print('%04d: process image... %s-%d' % (i, volumn_name, slice_idx))  # wyz

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


#    mae_normal = np.mean(mae_avg_normal)
#    mean_psnr_normal = np.mean(psnr_avg_normal)
#    std_psnr_normal = np.std(psnr_avg_normal)
#    mean_ssim_normal = np.mean(ssim_avg_normal)
#    std_ssim_normal = np.std(ssim_avg_normal)
#    print('testing - mae_normal: %.5f, psnr_mean_normal: %.3f, psnr_std_normal:%.3f, ssim_mean_normal: %.3f, ssim_std_normal:%.3f' % (mae_normal, mean_psnr_normal, std_psnr_normal, mean_ssim_normal, std_ssim_normal))

    # 正则化之后反而更小，如果需要正则化，需要把正则化模块加入到训练网络中
    # testing - mae: 0.01633, psnr_mean: 30.513, psnr_std:1.419, ssim_mean: 0.919, ssim_std:0.015
    # testing - mae_normal: 0.03011, psnr_mean_normal: 29.346, psnr_std_normal:2.861, ssim_mean_normal: 0.653, ssim_std_normal:0.118



"""
        if opt.dataset_mode=='aligned_mat':
            visuals=model.get_current_visuals()
            #visuals['real_A']=visuals['real_A'][:,:,0:3]
            #visuals['real_B']=visuals['real_B'][:,:,0:3]
            #visuals['fake_B']=visuals['fake_B'][:,:,0:3]    
            img_path = model.get_image_paths()
            img_path[0]=img_path[0]+str(i)
        elif  opt.dataset_mode=='unaligned_mat':   
            visuals=model.get_current_visuals()
            slice_select=[opt.input_nc/2,opt.input_nc/2,opt.input_nc/2]
            visuals['real_A']=visuals['real_A'][:,:,slice_select]
            visuals['real_B']=visuals['real_B'][:,:,slice_select]
            visuals['fake_A']=visuals['fake_A'][:,:,slice_select]
            visuals['fake_B']=visuals['fake_B'][:,:,slice_select]
            visuals['rec_A']=visuals['rec_A'][:,:,slice_select]
            visuals['rec_B']=visuals['rec_B'][:,:,slice_select]
            #temp_visuals['idt_A']=temp_visuals['idt_A'][:,:,slice_select]
            #temp_visuals['idt_B']=temp_visuals['idt_B'][:,:,slice_select]                    
            img_path = model.get_image_paths()
            img_path[0]=img_path[0]+str(i)            
        else:
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            slice_idx = model.get_slice_idx()  # wyz

        volumn_name = img_path[0].split('/')[-1].split('.')[0]  # wyz
        print('%04d: process image... %s-%d' % (i, volumn_name, slice_idx))  # wyz
        
        # 只需要可视化一部分 slice 即可
        if i < 1000:
            visualizer.save_images(webpage, visuals, img_path, slice_idx, aspect_ratio=opt.aspect_ratio)  # wyz

#   wyz
    mae = np.mean(mae_avg)
    mean_psnr = np.mean(psnr_avg)
    std_psnr = np.std(psnr_avg)
    mean_ssim = np.mean(ssim_avg)
    std_ssim = np.std(ssim_avg)

    print('testing - mae: %.5f, psnr_mean: %.3f, psnr_std:%.3f, ssim_mean: %.3f, ssim_std:%.3f' % (mae, mean_psnr, std_psnr, mean_ssim, std_ssim))
#
#
    webpage.save()
"""
