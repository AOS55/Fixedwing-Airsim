default_config = dict(
    epochs=20,
    learning_rate=1e-3,
    batch_size=1,
    dataset='test_set',
    classes=3,
    run_name='test_run'
)

category_rgb_vals = {
    tuple([0, 0, 0]): 0,
    tuple([78, 53, 104]): 1,
    tuple([155, 47, 90]): 2
}

category_rgb_names = {
    (0, 0, 0): "sky",
    (78, 53, 104): "runway",
    (155, 47, 90): "ground"
}