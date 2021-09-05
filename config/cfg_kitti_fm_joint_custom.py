DEPTH_LAYERS = 50
POSE_LAYERS = 18
FRAME_IDS = [0, -1, 1, 's']
IMGS_PER_GPU = 2
HEIGHT = 192#320
WIDTH = 640#1024

data = dict(
    name = 'kitti',
    split = 'exp_custom', # train files are eigen test split
    height = HEIGHT,
    width = WIDTH,
    frame_ids = FRAME_IDS,
    in_path = '/content/drive/MyDrive/Dự án/KITTI Dataset/Raw Data',
    gt_depth_path = '/content/logs/gt.npy',
    png = True,
    stereo_scale = True if 's' in FRAME_IDS else False,
)

model = dict(
    name = 'mono_fm_joint',
    depth_num_layers = DEPTH_LAYERS,
    pose_num_layers = POSE_LAYERS,
    frame_ids = FRAME_IDS,
    imgs_per_gpu = IMGS_PER_GPU,
    height = HEIGHT,
    width = WIDTH,
    scales = [0, 1, 2, 3],
    min_depth = 0.1,
    max_depth = 100.0,
    depth_pretrained_path = None,
    pose_pretrained_path =  None,
    extractor_pretrained_path = None,
    automask = False if 's' in FRAME_IDS else True,
    disp_norm = False if 's' in FRAME_IDS else True,
    dis=1e-3,
    cvt=1e-3,
    perception_weight = 1e-3,
    smoothness_weight = 1e-3,
)

resume_from = None
finetune = '/content/logs/kitti_fm_joint'
total_epochs = 2
imgs_per_gpu = IMGS_PER_GPU
learning_rate = 0
workers_per_gpu = 4
validate = True

optimizer = dict(type='Adam', lr=learning_rate, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[20,30],
    gamma=0.5,
)

checkpoint_config = dict(interval=1)
log_config = dict(interval=50,
                  hooks=[dict(type='TextLoggerHook'),])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
workflow = [('train', 1)]