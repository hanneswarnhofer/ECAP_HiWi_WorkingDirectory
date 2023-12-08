import tensorflow as tf


def default_mapping(feat, labels, drop_iact_img=True, mask_pixels=True):
    '''
        Individual definition of your own advanced preprocessing function.
        Heavy calculations (high memory consumption, computational overhead) should be done here.
        During training (on GPU), this part of the preprocessing is done on-the-fly on the CPU, which helps to
        enhance performance and keeping memory consumption low.
        This mapping will break for AixNet_v1_1_py3_8 and AixNet_v1_1.
    '''
    # t0

    if mask_pixels is True:

        for tel in ["ct1", "ct2", "ct3", "ct4"]:
            broken_pix = tf.cast(tf.random.uniform((41, 36, 1), 0, 1) < 2 * 3 / (41 * 36), tf.float32)
            feat[tel] = feat[tel] * (1 - broken_pix)

        broken_pix = tf.cast(tf.cast(tf.random.poisson((56, 56, 1), lam=2. / (56 * 56)), tf.bool), tf.float32)
        feat["ct5"] = feat["ct5"] * (1 - broken_pix)

    if drop_iact_img is True:
        x = tf.random.uniform((1,), 1, 4, dtype=tf.int32)
        is_mono = x == 1   # ~1/3% of training --> mono
        is_hessu1_stereo = x == 2   # ~1/3% of training --> HESS-IU stereo
        # is_hybrid = x == 3   # ~1/3% of training --> hybrid

        ct14_tels_to_keep = tf.cast(tf.random.uniform((4,), 0, 1) > 0.5, tf.float32) * (1 - tf.cast(is_mono, tf.float32))

        keep_ct5 = tf.cast(~is_hessu1_stereo, tf.float32)
        tels_to_keep = tf.concat([ct14_tels_to_keep, keep_ct5], axis=0)
        tels_to_remove = tf.cast(~tf.cast(tels_to_keep, tf.bool), tf.float32)

        signal = tf.stack([tf.reduce_sum(feat["ct%i" % i], keepdims=False, axis=(-1, -2, -3)) for i in range(1, 6)])
        triggered = tf.cast(signal > 0, tf.float32)
        least_single_tel_triggered = tf.cast(tf.reduce_sum(tels_to_keep * triggered, keepdims=True) > 0, tf.float32)
        tel_mask = tf.ones(5) - least_single_tel_triggered * tels_to_remove  # remove only if at least a single telesope is unmasked
        # tf.print(tel_mask)

        for i, t_mask in enumerate(tf.split(tel_mask, 5)):
            feat["ct%i" % (i + 1)] *= tf.expand_dims(tf.expand_dims(t_mask, axis=-1), axis=-1)

    return feat, labels
