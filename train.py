# -*- coding:utf-8 -*-
from absl import flags
from random import random, shuffle
from ethnicity_model import *

import numpy as np
import sys
import os
import datetime

flags.DEFINE_string("tr_img_path", "", "Training image path")

flags.DEFINE_string("tr_txt_path", "", "Training text path")

flags.DEFINE_string("te_img_path", "", "Testing image path")

flags.DEFINE_string("te_txt_path", "", "Testing text path")

flags.DEFINE_integer("img_size", 224, "Height and Width")

flags.DEFINE_integer("img_ch", 3, "Channels")

flags.DEFINE_integer("batch_size", 32, "Batch size")

flags.DEFINE_integer("epochs", 150, "Training epochs")

flags.DEFINE_float("lr", 0.001,"Learning rate")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("pre_checkpoint_path", "", "Saved checkpoint path")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_string("weight_path", "C:/Users/Yuhwan/Desktop/yuhwan/rcmalli_vggface_tf_notop_vgg16.h5", "Vggface weight")

flags.DEFINE_string("graphs", "", "")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.Adam(FLAGS.lr)

def train_func(img_path, lab_list):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, FLAGS.img_ch)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)

    lab = tf.cast(lab_list, tf.float32)
    return img, lab

def test_func(img_path, lab_list):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, FLAGS.img_ch)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    lab = tf.cast(lab_list, tf.float32)
    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, images, labels):

    with tf.GradientTape() as tape:
        logits = run_model(model, images, True)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def cal_acc(model, images, labels):

    logits = run_model(model, images, False)
    logits = tf.nn.sigmoid(logits)  # [batch, 1]
    logits = tf.squeeze(logits, 1)  # [batch]

    predict = tf.cast(tf.greater(logits, 0.5), tf.float32)
    count_acc = tf.cast(tf.equal(predict, labels), tf.float32)
    count_acc = tf.reduce_sum(count_acc)

    return count_acc

def main():
    ethnicity_MODEL = ethnic_model(input_shape=(FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch),
                                   include_top=False,
                                   pooling=None,
                                   weights="vggface",
                                   weight_path=FLAGS.weight_path)
    regularizer = tf.keras.regularizers.l2(0.00005)
    for layer in ethnicity_MODEL.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    h = ethnicity_MODEL.output  # VGGface2 로 학습된 모델을 이용해서 이 layer에 weight를 추가시켜야한다!!!!
    # https://github.com/rcmalli/keras-vggface
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(2048)(h)
    h = tf.keras.layers.Dropout(rate=0.5)(h)
    h = tf.keras.layers.Dense(1024)(h)
    h = tf.keras.layers.Dropout(rate=0.5)(h)
    h = tf.keras.layers.Dense(1)(h)
    ethnicity_MODEL = tf.keras.Model(inputs=ethnicity_MODEL.input, outputs=h)
    ethnicity_MODEL.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(ethnicity_MODEL=ethnicity_MODEL,
                                   optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored the latest checkpoint files!!")

    if FLAGS.train:
        count = 0

        tr_img = np.loadtxt(FLAGS.tr_txt_path, dtype="<U100", skiprows=0 ,usecols=0)
        tr_img = [FLAGS.tr_img_path + img for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_img = np.loadtxt(FLAGS.te_txt_path, dtype="<U100", skiprows=0 ,usecols=0)
        te_img = [FLAGS.te_img_path + img for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.shuffle(len(te_img))
        te_gener = te_gener.map(test_func)
        te_gener = te_gener.batch(FLAGS.batch_size)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        val_log_dir = FLAGS.graphs + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(FLAGS.epochs):

            TR = list(zip(tr_img, tr_lab))
            shuffle(TR)
            tr_img, tr_lab = zip(*TR)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab, dtype=np.int32)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(train_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = len(tr_img) // FLAGS.batch_size
            tr_iter = iter(tr_gener)

            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)

                loss = cal_loss(ethnic_model, batch_images, batch_labels)

                with train_summary_writer.as_default():
                    tf.summary.scalar('total loss', loss, step=count)

                if count % 10 == 0:
                    print("Epoch(s): {} [{}/{}] Total loss = {}".format(epoch, step + 1, tr_idx, loss))

                if count % 50 == 0:
                    # test
                    te_idx = len(te_img) // FLAGS.batch_size
                    te_iter = iter(te_gener)
                    count_acc = 0.
                    for i in range(te_idx):
                        te_images, te_labels = next(te_iter)
                        count_acc += cal_loss(model, te_images, te_labels)

                    ACC = (count_acc / len(te_img)) * 100.
                    print("Acc = {} for {} steps".format(ACC, count))

                    with val_summary_writer.as_default():
                        tf.summary.scalar('total loss', ACC, step=count)


                if count % 500 == 0:
                    num_ = int(count // 500)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)

                    ckpt = tf.train.Checkpoint(ethnicity_MODEL=ethnicity_MODEL,
                                               optim=optim)
                    ckpt_dir = model_dir + "/" + "ethnic_model_{}.ckpt".format(count)

                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__" :
    main()