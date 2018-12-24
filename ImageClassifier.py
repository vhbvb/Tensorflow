# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob

'''
初始化常量
'''
sources_path = "../CoreML/flower_photos/"
split_rate = 0.2
categories = glob.glob(sources_path + "*")
category_names = [x.split("/")[-1] for x in categories]
print "category_names:%s"%category_names
log_path = "logs/"
checkpoint_path = "logs/model.ckpt"

'''
从本地文件读取数据
'''
def readData():
    train_image_paths = []
    test_image_paths = []
    for path in categories:
        image_paths = glob.glob(path + "/*.jpg")
        split_count = int(len(image_paths) * (1 - split_rate))
        train_image_paths += image_paths[:split_count]
        test_image_paths += image_paths[split_count:]

    train_labels = [x.split("/")[-2] for x in train_image_paths]
    test_labels = [x.split("/")[-2] for x in test_image_paths]
    train_label_list = [category_names.index(x) for x in train_labels]
    test_labels_list = [category_names.index(x) for x in test_labels]

    train_image_paths = np.asarray(train_image_paths)
    test_image_paths = np.asarray(test_image_paths)

    train_tmp = np.array([train_image_paths, train_label_list])
    train_tmp_t = train_tmp.T
    np.random.shuffle(train_tmp_t)
    train_tmp = train_tmp_t.T
    train_image_paths = train_tmp[0]

    # 这个地方合并后会变成str类型的，导致后续无法执行
    train_label_list = [int(x) for x in train_tmp[1]]

    test_tmp = np.array([test_image_paths, test_labels_list])
    test_tmp_t = test_tmp.T
    np.random.shuffle(test_tmp_t)
    test_tmp = test_tmp_t.T
    test_image_paths = test_tmp[0]
    test_labels_list = [int(x) for x in test_tmp[1]]

    return (train_image_paths, train_label_list), (test_image_paths, test_labels_list)

'''
图片加噪点
'''
def gaussian_noise_layer(input_image, std):
    noise = tf.random_normal(shape=tf.shape(input_image), mean=0.0, stddev=std, dtype=tf.float32)
    noise_image = tf.cast(input_image, tf.float32) + noise
    noise_image = tf.clip_by_value(noise_image, 0, 1.0)
    return noise_image

'''
生成器，用于生产训练数据
Returns:
训练需要的数据
'''
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    print "get_batch:\n" + str(image.shape)

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    # image = tf.image.resize_image_with_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    #     dataset = tf.data.Dataset.from_tensor_slices(image,label)
    #     dataset.batch(batch_size)
    #     dataset.capacity(capacity)
    #     iterator =  dataset.make_initializable_iterator()
    #     image_batch,label_batch = iterator.

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=8, capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch

'''
图片预览
'''
def view_images(images,labels,count):
    with tf.Session() as sess:
        i=0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        try:
            while not coord.should_stop() and i<1:
                print "view images:\n"
                img,label = sess.run([images,labels])
                for j in np.arange(count):
                    plt.imshow(img[j,:,:,:])
                    print category_names[label[j]]
                    plt.xlabel(category_names[label[j]])
                    plt.show()
                i+=1
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)


'''
模型定义
'''
def inference(images, batch_size, n_classes):
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weight',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(0.1))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')

        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('pooling_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weight',
                                  shape=[3, 3, 16, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(0.1))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')

    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.005))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.005))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(0.005))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.005))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


'''
训练
'''
N_CLASSES = 5
IMG_W = 64
IMG_H = 64
BATCH_SIZE = 50
CAPACITY = 200.0
MAX_STEP = 2000
learning_rate = 0.01


def run_training():
    (train, train_label), (test, test_label) = readData()
    train_batch, train_label_batch = get_batch(train, train_label,
                                               IMG_W,
                                               IMG_H,
                                               BATCH_SIZE,
                                               CAPACITY)
    # 查看前10张训练集
    view_images(train_batch, train_label_batch, 10)

    train_logits = inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = losses(train_logits, train_label_batch)
    train_op = trainning(train_loss, learning_rate)
    train_acc = evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(log_path, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
            if step % 50 == 0:
                print('Step %d,train loss = %.2f,train occuracy = %.2f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

'''
开始训练
'''
run_training()