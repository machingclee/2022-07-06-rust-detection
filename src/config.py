input_width = 2 ** 10
input_height = 2 ** 10
image_shape = (input_height, input_width)

anchor_ratios = [0.5, 1, 2]
anchor_scales = [16 * 16, 16 * 8, 16 * 32, 16 * 64]

grid_nums = [(input_height // 16, input_width // 16)]

target_pos_iou_thres = 0.7
target_neg_iou_thres = 0.3

rpn_n_sample = 128
rpn_pos_ratio = 0.5

n_pos = rpn_pos_ratio * rpn_n_sample


roi_n_sample = 128
roi_pos_ratio = 0.25
roi_pos_iou_thresh = 0.5
roi_neg_iou_thresh = 0.3


nms_train_iou_thresh = 0.7
nms_eval_iou_thresh = 0.7
n_train_pre_nms = 12000
n_train_post_nms = 2000
n_eval_pre_nms = 6000
n_eval_post_nms = roi_n_sample
min_size = 16

train_annotation_files = ["annotations/train.txt"]
test_annotation_files =["annotations/test.txt"]
pred_score_thresh = 0.2


roi_head_encode_weights = [10, 10, 5, 5]

pths_dir = "pths"
font_path = "fonts/wt014.ttf"
labels = ["rust"]

n_classes = len(labels) + 1  # include background

output_nms_within_class = 0.15
serve_weight_path =  "/home/raspect/nas1/tmp/joe/dsds/dsds_models/rust_cls_0003/model_epoch_50.pth"