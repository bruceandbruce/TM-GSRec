import os
import time
import numpy as np, h5py 

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from options.train_options_wyz import TrainOptions
from data import CreateDataLoader
from models import create_model
from util1.visualizer import Visualizer

# python train.py --lambda_adv 0 --dataroot IXI_short.csv --name G1D20 --gpu_ids 1 --model resvit_one --which_model_netG GaussianSR_isbi --which_direction AtoB --lambda_A 100 --dataset_mode aligned2 --norm batch --pool_size 0 --output_nc 1 --input_nc 1 --loadSize 256 --fineSize 256 --niter 1 --niter_decay 50 --save_epoch_freq 1 --checkpoints_dir EXP_checkpoints/ --display_id 0 --pre_trained_transformer 0 --pre_trained_resnet 0 --lr 2e-4 --batchSize 1 --mask G1D20 --continue_train

def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

if __name__ == '__main__':
    opt = TrainOptions().parse()

    # training data - 训练集
    opt.phase = 'train' 
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    #dataset_size = len(data_loader)
    #print(f'================================ Validation images = {dataset_size} ================================')

    # logger
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()

    # validation data - 验证集
    opt.phase = 'valid'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    #dataset_size_val = len(data_loader_val)
    #print(f'================================ Validation images = {dataset_size_val} ================================')

    if opt.model == 'cycle_gan':
        L1_avg = np.zeros([2, opt.niter + opt.niter_decay, len(dataset_val)])      
        psnr_avg = np.zeros([2, opt.niter + opt.niter_decay, len(dataset_val)])    
        ssim_avg = np.zeros([2, opt.niter + opt.niter_decay, len(dataset_val)])      
    else:
        L1_avg = np.zeros([opt.niter + opt.niter_decay, len(dataset_val)])      
        psnr_avg = np.zeros([opt.niter + opt.niter_decay, len(dataset_val)])
        ssim_avg = np.zeros([opt.niter + opt.niter_decay, len(dataset_val)])
   
    # 实例化模型    
    model = create_model(opt)

    # 可视化器
    visualizer = Visualizer(opt)

    # 开始迭代
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        ## training step
        opt.phase = 'train'
        # opt.phase = 'valid'
        for i, data in enumerate(dataset):
            # 计时/计数
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            ##### 输入数据 / 前向反向
            model.set_input(data)

            start = time.perf_counter()
            model.optimize_parameters_time()
            end = time.perf_counter()
            execution_time = (end - start)
            # print("每个切片的重建时间：", execution_time)


            model.optimize_parameters()

            # 训练时指标计算测试
            fake_im = model.fake_B.cpu().data.numpy()  # (16, 1, 256, 256)
            real_im = model.real_B.cpu().data.numpy()
            psnr_list = []
            ssim_list = []
            for j in range(real_im.shape[0]):
                real_im_ = real_im[j, ...]
                fake_im_ = fake_im[j, ...]

                psnr_list.append(psnr(real_im_, fake_im_, data_range=1))
                ssim_list.append(ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_im_, (1, 2, 0)), data_range=1, channel_axis=2))
                #ssim_list.append(ssim(real_im_, fake_im_, data_range=1, multichannel=False))   # 不行

            print('[Epoch%3d] training - psnr_mean: %.3f, psnr_std: %.3f, ssim_mean: %.3f, ssim_std: %.3f' % (epoch, np.mean(psnr_list), np.std(psnr_list), np.mean(ssim_list), np.std(ssim_list)) )
   
            #print(f"fake_im.max(): {fake_im.max()}, fake_im.min(): {fake_im.min()} ============== ")
            #print(f"real_im.max(): {real_im.max()}, real_im.min(): {real_im.min()} ============== ")  # real_im.max(): 1.0, real_im.min(): 0.0

            # 当前可视化结果与数据
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                if opt.dataset_mode=='aligned_mat':
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)
                elif  opt.dataset_mode=='unaligned_mat':   
                    temp_visuals=model.get_current_visuals()
                    temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]
                    temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
                    temp_visuals['fake_A']=temp_visuals['fake_A'][:,:,0:3]
                    temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
                    temp_visuals['rec_A']=temp_visuals['rec_A'][:,:,0:3]
                    temp_visuals['rec_B']=temp_visuals['rec_B'][:,:,0:3]
                    if opt.lambda_identity>0:
                      temp_visuals['idt_A']=temp_visuals['idt_A'][:,:,0:3]
                      temp_visuals['idt_B']=temp_visuals['idt_B'][:,:,0:3]                    
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                else:
                    temp_visuals = model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                    
            # 打印
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            # 保存
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()

        ## validaiton step
        # 验证和保存频率取决于 opt.save_epoch_freq
        if epoch % opt.save_epoch_freq == 0:
            logger = open(os.path.join(save_dir, 'log.txt'), 'a')
            print(opt.dataset_mode)

            cnt = 0  # 0 ~ len(valid set)-1

            opt.phase = 'valid'
            for i, data_val in enumerate(dataset_val):   # 按 batch 数  
                ##### 输入数据 / 前向 
                model.set_input(data_val)  
                model.test()
	    
                fake_im = model.fake_B.cpu().data.numpy()
                real_im = model.real_B.cpu().data.numpy()

                #print(fake_im.shape, psnr_avg.shape)  # (16, 1, 256, 256) (32, 256)

                for j in range(real_im.shape[0]):
                    real_im_ = real_im[j, ...]
                    fake_im_ = fake_im[j, ...]
                    # 计算并记录单个样本的指标
                    L1_avg[epoch-1, cnt] = abs(fake_im_-real_im_).mean()
                    psnr_avg[epoch-1, cnt] = psnr(real_im_, fake_im_, data_range=1)
                    ssim_avg[epoch-1, cnt] = ssim(np.transpose(real_im_, (1, 2, 0)), np.transpose(fake_im_, (1, 2, 0)), data_range=1, channel_axis=2)
                    cnt += 1    

            # 计算整个验证集的指标
            l1_avg_loss = np.mean(L1_avg[epoch-1])
            mean_psnr = np.mean(psnr_avg[epoch-1])
            std_psnr = np.std(psnr_avg[epoch-1])
            mean_ssim = np.mean(ssim_avg[epoch-1])
            std_ssim = np.std(ssim_avg[epoch-1])

            print_log(logger, '[Epoch %3d] l1_avg_loss: %.5f, psnr_mean: %.3f, psnr_std:%.3f, ssim_mean: %.3f, ssim_std:%.3f' % (epoch, l1_avg_loss, mean_psnr, std_psnr, mean_ssim, std_ssim))
            print_log(logger, '')
            logger.close()

            print('########## saving the model at the end of epoch %d, iters %d #############' % (epoch, total_steps))
	    
            model.save('latest')	   
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate()
