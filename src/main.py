import os, torch
import torchvision.models as models
from torch.optim import SGD
from fc_correction import CosineLinear, correct_weights
from os.path import exists
from datasets import CLImageFolder
import torchvision.models as models
from opts import parse_args
from utils import get_logger
from torch import nn
import numpy as np
from utils import LinearLR, seed_everything, save_model
from training import train, test, save_representations
import copy

def per_timestep_loop(opt, logger, dataset):
    assert(opt.model in ['resnet50_I1B', 'resnet50'])
    if opt.model == 'resnet50':
        model = models.resnet50(weights="IMAGENET1K_V2")
    elif opt.model == 'resnet50_I1B':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
    
    linear_size = list(model.children())[-1].in_features

    if opt.fc == 'cosine_linear':
        model.fc = CosineLinear(in_features=linear_size, out_features=opt.num_curr_classes)
    else:
        model.fc = nn.Linear(linear_size, opt.num_curr_classes)

    prevmodel = None

    if opt.pretrain_modelpath is not None:
        logger.info('==> Loading from '+opt.pretrain_modelpath+' for continual training..')
        model.load_state_dict(torch.load(opt.pretrain_modelpath)['state_dict'])
        if opt.distill is not None and opt.timestep > 1:
            prevmodel = copy.deepcopy(model)
            prevmodel = prevmodel.cuda()
        else:
            prevmodel = None
            
    if opt.expand_size > 0: # Add new classes to the final classifier 
        # Expand the linear size and set the new weights and biases to small random and zero respectively.
        new_weights = (torch.ones(opt.expand_size, linear_size))
        new_weights = 0.001 * nn.init.xavier_normal_(new_weights)
        model.fc.weight = nn.Parameter(torch.cat((model.fc.weight, new_weights), dim=0))
        if model.fc.bias is not None:
            new_biases = torch.zeros(opt.expand_size)
            model.fc.bias =  nn.Parameter(torch.cat((model.fc.bias, new_biases), dim=0))
    
    # model = torch.nn.DataParallel(model.cuda()) # Use Dataparallel for now if running on multiple GPUs -- need for distillation experiments if not wanting to store

    optimizer = SGD(model.parameters(), lr=opt.maxlr, momentum=opt.momentum, weight_decay=opt.weight_decay)  
    scheduler = LinearLR(optimizer, T=opt.total_steps)

    if opt.fc == 'linear_only': # option to only tune the fully-connected layers
        for param in model.parameters():
            param.requires_grad = False

        model.fc.weight.requires_grad = True
        if model.fc.bias is not None:
            model.fc.bias.requires_grad = True

    model, optimizer, scheduler = train(opt=opt, loader=dataset.trainloader, model=model,optimizer=optimizer, scheduler=scheduler, logger=logger, prevmodel=prevmodel)
    
    # Postprocessing
    if opt.dset_mode == 'class_incremental' and opt.calibrator!=None:
        model = correct_weights(model=model, valloader=dataset.valloader, calibration_method=opt.calibrator, logger=logger, expand_size=opt.expand_size)
        
    if opt.sampling_mode in ['herding', 'kmeans']:
        save_representations(opt=opt, loader=dataset.trainloader_eval, model=model, mode='features')
    elif opt.sampling_mode in ['unc_lc', 'max_loss']:
        save_representations(loader=dataset.trainloader_eval, model=model, mode='predictions')
        

    # Testing Part 
    predsarr, labelsarr = test(loader=dataset.testloader, model=model, logger=logger)
    np.save(opt.log_dir+'/'+opt.exp_name+'/labels_'+str(opt.timestep)+'_cltestset.npy', labelsarr.numpy())
    np.save(opt.log_dir+'/'+opt.exp_name+'/preds_'+str(opt.timestep)+'_cltestset.npy', predsarr.numpy())
    del predsarr, labelsarr
     
    if opt.dataset == 'Imagenet2K':
        predsarr, labelsarr = test(loader=dataset.pretestloader, model=model, logger=logger)
        np.save(opt.log_dir+'/'+opt.exp_name+'/labels_'+str(opt.timestep)+'_pretestset.npy', labelsarr.numpy())
        np.save(opt.log_dir+'/'+opt.exp_name+'/preds_'+str(opt.timestep)+'_pretestset.npy', predsarr.numpy())
        del predsarr, labelsarr
        
    save_model(opt, model)
    print('Finished timestep: ', opt.timestep)


if __name__ == '__main__':
    # Parse arguments and init loggers
    torch.multiprocessing.set_sharing_strategy('file_system') # For error: Too many files open
    opt = parse_args()
    opt.exp_name = f'{opt.dataset}_{opt.model}_{opt.dset_mode}_{opt.sampling_mode}_{opt.optimizer}_{opt.maxlr}_{opt.total_steps}_{opt.increment_size}'
    opt.timestep = 0
    console_logger = get_logger(folder=opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.info('==> Params for this experiment:'+str(opt))
    seed_everything(opt.seed)
    opt.expand_size = 0
    
    # Pretraining phase starts here
    console_logger.debug('==> Loading pretraining dataset..')
    dataset = CLImageFolder(opt=opt)
    opt.num_curr_classes = len(dataset.curr_classes)
    
    # Continual phase starts here
    start=0
            
    for timestep in range(opt.num_timesteps):
        opt.timestep = timestep+1 # Note: opt.timestep starts from 1 and not 0.
        os.makedirs(opt.log_dir+'/'+opt.exp_name+'/'+str(opt.timestep)+'/', exist_ok=True)

        if opt.model_type == 'gdumb' or opt.timestep==1:
            opt.pretrain_modelpath = None
        elif opt.model_type == 'normal':
            opt.pretrain_modelpath = opt.log_dir+'/'+opt.exp_name+'/'+str(opt.timestep-1)+'/last.ckpt'
        assert(opt.model_type in ['normal', 'gdumb'])
        
        dataset.get_next_timestep_dataloader()
        opt.expand_size = dataset.expand_size

        if opt.dataset == 'CGLM':
            opt.num_curr_classes = len(dataset.curr_classes)

        if (not exists(opt.log_dir+'/'+opt.exp_name+'/'+str(opt.timestep+1)+'/last.ckpt')):
            console_logger.info('==> Starting training of timestep '+str(opt.timestep)+'..')
            per_timestep_loop(opt=opt, logger=console_logger, dataset=dataset)
            console_logger.info('==> Completed training for timestep '+str(opt.timestep)+'..')
        opt.num_curr_classes = len(dataset.curr_classes)

    console_logger.info('Experiment completed! Total timesteps: '+str(opt.timestep))
