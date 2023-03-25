import argparse

def parse_args():
   parser = argparse.ArgumentParser(description='main.py')
   # Changing options -- Apart from these arguments, we do not mess with other arguments

   ## Paths
   parser.add_argument('--log_dir', type=str, default='../logs/', help='Full path to the directory where all logs are stored')
   parser.add_argument('--order_file_dir', type=str, default='../../order_files/', help='Full path to the order file')

   ## Dataset
   parser.add_argument('--dataset', type=str, default='Imagenet', choices=['Imagenet', 'GLDv2'], help='Dataset')
   parser.add_argument('--timestep', type=int, default=0, help='Training timestep')
   parser.add_argument('--train_batch_size', type=int, default=1500, help='Batch size to be used in training')
   parser.add_argument('--test_batch_size', type=int, default=1500, help='Batch size to be used in testing')
   parser.add_argument('--crop_size', type=int, default=224, help='Size of the image input')
   parser.add_argument('--dset_mode', type=str, default='class_incremental', choices=['class_incremental', 'random'], help='Number of parallel worker threads')
   parser.add_argument('--num_classes_per_timestep', type=int, default=0, help='Number of parallel worker threads')
   parser.add_argument('--num_timesteps', type=int, default=20, help='Number of parallel worker threads')
   parser.add_argument('--increment_size', type=int, default=0, help='Number of parallel worker threads')

#    ## Model
   parser.add_argument('--optimizer', type=str, default='LARS', choices=['LARS','SGD'], help='Optimizer type chosen for the network')
   parser.add_argument('--model', type=str, default='resnet50', choices=['resnet18', 'resnet50_I1B', 'resnet50', 'resnet50_2', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf'], help='Model architecture')
   parser.add_argument('--save_name', type=str, default='test', help='Model architecture')
   parser.add_argument('--sampling_mode', type=str, default='uniform', choices=['uniform', 'classwise_stratified', 'lastk','timewise_stratified', 'timeclasswise_stratified', 'sqrt_accbalanced', 'reservoir','accbalanced','log_accbalanced','exp_accbalanced','herding', 'kmeans', 'classwise_herding','classwise_kmeans','unc_lc','max_loss'], help='Number of parallel worker threads')
   parser.add_argument('--model_type',type=str, default='Normal', choices=['GDumb','Normal'], help='Types of restart: ER and GDumb or do full offline training')
   parser.add_argument('--fc',type=str, default='Linear', choices=['Linear','cosine_linear','linear_only','ACE'], help='Types of restart: ER and GDumb or do full offline training')
   parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer Momentum')

   parser.add_argument('--distill', type=str, default=None, help='Disllation Mode')

#    ## Experiment Deets
   parser.add_argument('--exp_name', type=str, default='test', help='Full path to the order file')
   parser.add_argument('--maxlr', type=float, default=0.001, help='Starting Learning rate')
   parser.add_argument('--total_steps', type=int, default=80, help='Maximum number of epochs')
   parser.add_argument('--repeat_sampling', type=int, default=1, help='Maximum number of epochs')
   parser.add_argument('--repeated_steps', type=int, default=0)
   parser.add_argument('--val_freq', type=int, default=100, help='Printing utils')
   parser.add_argument('--timecount_alpha', type=float, default=1.0, help='Training timestep')
   parser.add_argument('--baselog_val', type=float, default=2, help='Training timestep')
   
#    # Default options
   parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
   parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
   parser.add_argument('--clip', type=float, default=2.0, help='Gradient Clipped if val >= clip')
   parser.add_argument('--num_workers', type=int, default=16, help='Starting Learning rate')
   parser.add_argument('--pretrained', type=bool, default=True, help='USe pretrained weights or not')

   parser.add_argument('--calibrator',type=str, default=None, choices=['Temperature', 'WA', 'BiC'], help='Types of restart: ER and GDumb or do full offline training')

   parser.add_argument('--print_freq', type=int, default=100, help='Seed for reproducibility')
   opt = parser.parse_args()
   return opt
