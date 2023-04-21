version = 1.0

# dataset_name = "consep"
# dataset_raw = "data/archives/CoNSeP/"
# train_dir = "data/consep/train/"
# test_dir = "data/consep/test/"

# Combined Dataset CoNSep, PanNuke, Lizard
dataset_name = "all"
dataset_raw = None
train_dir = "data/all/train/"
test_dir = "data/all/test/"

checkpoints_dir = "checkpoints"
tensorboard_logs = "run_logs"

encoder = "tu-tf_efficientnet_b5_ns"
encoder_weights = None

# CoNSep Weights
# inference_weights = "NC-Net_consep_metric.pth"

# All Weights
inference_weights = "NC-Net_all_metric.pth"

device = "cuda"
batch_size = 16
epochs = 1000
learning_rate = 0.01
watershed_threshold = 0.2
