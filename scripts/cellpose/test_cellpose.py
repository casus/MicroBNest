import numpy as np
from stardist.matching import matching

from cellpose import io, models

TRAIN_DIR = "/bigdata/casus/MLID/nips_benchmark/hela_cyto_cellpose/train"
TEST_DIR = "/bigdata/casus/MLID/nips_benchmark/hela_cyto_cellpose/test"


output = io.load_train_test_data(
    TRAIN_DIR,
    TEST_DIR,
    image_filter="_img",
    mask_filter="_masks",
    look_one_level_down=False,
)
images, labels, image_names, test_images, test_labels, image_names_test = output

model = models.CellposeModel(
    pretrained_model="/home/wyrzyk93/DeepStain/models/cyto_model_final"
)

masks_pred, flows, _ = model.eval(test_images, channels=[1, 3])

for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    acc = np.zeros(len(test_labels))
    iour = np.zeros(len(test_labels))
    for i in range(len(test_labels)):
        match = matching(np.array(test_labels[i]), np.array(masks_pred[i]), thresh=t)
        acc[i] = match.accuracy
        iour[i] = match.mean_true_score
    print(t, "acc", np.mean(acc))
    print(t, "iour", np.mean(iour))
