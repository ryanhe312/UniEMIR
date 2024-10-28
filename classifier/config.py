class Config:
    lr = 0.0005
    epoch = 200
    train_batch_size = 3
    test_batch_size = 32

class NoiseConfig(Config):
    task = "noise"
    train_data_root=r'/home/user2/dataset/microscope/EMDiffuse/EMDiffuse_dataset/brain_train/denoise/train_gt'
    test_data_root=r'/home/user2/dataset/microscope/EMDiffuse/EMDiffuse_dataset/brain_test/denoise/train_gt'

class ZoomConfig(Config):
    lr = 0.0001
    task = "blur"
    train_data_root=r'/home/user2/dataset/microscope/EMDiffuse/EMDiffuse_dataset/brain_train/zoom/train_gt'
    test_data_root=r'/home/user2/dataset/microscope/EMDiffuse/EMDiffuse_dataset/brain_test/zoom/train_gt'