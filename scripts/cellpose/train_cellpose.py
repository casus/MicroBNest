from cellpose import io, models, train

io.logger_setup()


TRAIN_DIR = ""
VAL_DIR = ""

images, labels, image_names, test_images, test_labels, image_names_test = (
    io.load_train_test_data(
        TRAIN_DIR,
        VAL_DIR,
        image_filter="_img",
        mask_filter="_masks",
        look_one_level_down=False,
    )
)

model = models.CellposeModel(model_type="cyto3", gpu=True)
model_path = train.train_seg(
    model.net,
    train_data=images,
    train_labels=labels,
    channels=[1, 3],
    test_data=test_images,
    test_labels=test_labels,
    n_epochs=300,
    save_every=5,
    model_name="",
)
