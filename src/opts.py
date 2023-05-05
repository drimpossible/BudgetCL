import argparse

def parse_args():
   parser = argparse.ArgumentParser(description='main.py')
   # Changing options -- Apart from these arguments, we do not mess with other arguments

   ## Paths
   parser.add_argument('--log_dir', type=str, default='../logs/', help='Full path to the directory where all logs are stored')
   parser.add_argument('--order_file_dir', type=str, default='../../order_files/', help='Full path to the ordering files')

   ## Dataset
   parser.add_argument('--dataset', type=str, default='Imagenet', choices=['Imagenet2K', 'CGLM'], help='Dataset used for CL')
   parser.add_argument('--timestep', type=int, default=0, help='Timestep to start learning from (for resuming)')
   parser.add_argument('--train_batch_size', type=int, default=1500, help='Batch size to be used in training')
   parser.add_argument('--test_batch_size', type=int, default=1500, help='Batch size to be used in testing')
   parser.add_argument('--crop_size', type=int, default=224, help='Size of the image input')
   parser.add_argument('--dset_mode', type=str, default='class_incremental', choices=['class_incremental', 'data_incremental', 'time_incremental'], help='Dataset Ordering to choose from')
   parser.add_argument('--num_classes_per_timestep', type=int, default=0, help='Number of classes per timestep in case of class incremental setting')
   parser.add_argument('--num_timesteps', type=int, default=20, help='Number of timesteps to split data over')
   parser.add_argument('--increment_size', type=int, default=0, help='Number of samples per timestep in case of time/data incremental setting')

#    ## Model
   parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD'], help='Optimizer type chosen for the network')
   parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50_i1b', 'resnet50'], help='Model architecture')
   parser.add_argument('--model_type', type=str, default='normal', choices=['normal', 'gdumb'], help='Model architecture')
   parser.add_argument('--sampling_mode', type=str, default='uniform', choices=['uniform', 'class_balanced', 'lastk', 'herding', 'kmeans', 'unc_lc', 'max_loss', 'recency_biased'], help='Sampling Strategies Tested')
   parser.add_argument('--fc',type=str, default='linear', choices=['linear','cosine_linear','linear_only', 'ace'], help='Last layer training strategies')
   parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer Momentum')
   parser.add_argument('--distill', type=str, default=None, choices=['cosine','mse','bce', 'ce'], help='Disllation Mode')
   parser.add_argument('--calibrator',type=str, default=None, choices=['Temperature', 'WA', 'BiC'], help='Types of restart: ER and GDumb or do full offline training')
   parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights or not')

#    ## Experiment Deets
   parser.add_argument('--exp_name', type=str, default='test', help='Experiment name. Saving is done as log_dir/exp_name')
   parser.add_argument('--maxlr', type=float, default=0.001, help='Starting Learning rate')
   parser.add_argument('--total_steps', type=int, default=80, help='Maximum number of training steps')
   
#    # Default options
   parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
   parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
   parser.add_argument('--clip', type=float, default=2.0, help='Gradient Clipped if val >= clip, gives stable training')
   parser.add_argument('--num_workers', type=int, default=16, help='Starting learning rate')
   opt = parser.parse_args()
   return opt
