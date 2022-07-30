version = 1.0

dataset_name = "consep"
dataset_raw = "data/archives/CoNSeP/"
train_dir = "data/consep/train/"
test_dir = "data/consep/test/"


checkpoints_dir = "checkpoints"
tensorboard_logs = "run_logs"

encoder = "tu-tf_efficientnet_b5_ns"
encoder_weights = None

inference_weights = "NC-Net_consep_metric.pth"

device = "cuda"
batch_size = 16
epochs = 1000
learning_rate = 0.01
watershed_threshold = 0.2