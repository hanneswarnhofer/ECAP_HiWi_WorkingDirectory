def rm_unused_labels(feat, labels, exclude_keys):
    new_labels = {}

    for k, inp in labels.items():

        if k not in exclude_keys:
            new_labels[k] = inp

    return feat, new_labels
