from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
import os
from PIL import Image
import tensorflow as tf



############## Функция для пикового отношения сигнал/шум (PSNR) ##########
def psnr(target, ref):

    target_data = target.astype(float)
    ref_data = ref.astype(float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return (20 * math.log10(255. / rmse))

############## Функция для среднеквадратичной ошибки (MSE) ##########
def mse(target, ref):
    target_data = target.astype(np.float32)
    ref_data = ref.astype(np.float32)
    err = np.sum((target_data - ref_data) ** 2)

    err /= np.float(target_data.shape[0] * target_data.shape[1])
    return err

############## Функция, которая объединяет все три показателя качества изображения ##########
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel=True))
    return scores


def prepare_images(path, factor):
    # Loop through the files in the directory
    for file in os.listdir(path):
        image = cv2.imread(path + '/' + file)

        # Find old and new image dimensions
        h, w, c = image.shape
        new_height = int(h / factor)
        new_width = int(w / factor)

        # Resize down the image
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Resize up the image
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Save the image
        try:
            os.listdir(path + '/../../resized')
        except:
            os.mkdir(path + '/../../resized')

        cv2.imwrite(path + '/../../resized/{}'.format(file), image)



def model_rdn(pre_model):
    SRCNN = Sequential(name='input_sr')

    SRCNN.add(pre_model)

    SRCNN.add(Conv2D(128, (9, 9),
                      activation='relu',
                      padding='same',
                      name='sr_conv1'))
    SRCNN.add(Conv2D(64, (3, 3),
                     activation='relu',
                     padding='same',
                     name='sr_conv2'))

    SRCNN.add(Conv2D(1, (5, 5),
                     activation='linear',
                     padding='same',
                     name='sr_conv3'))


     # define optimizer_
    adam = Adam(lr=0.0003)

    # compile model_
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return SRCNN


def load_main_model(name_model):
    # Загружаем модель NASNetMobile и предсказываем

    vgg16 = name_model(weights='imagenet',include_top=False, input_shape=(150, 150, 3))
    vgg16.trainable = False# Сверточная часть не обучается
    print (vgg16.summary())

    return vgg16

if __name__ == '__main__':

    prepare_images('./source/Set14', 2)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(Image.open('./source/Set14/barbara.bmp'))
    ax[0].title.set_text('Original Image')
    ax[1].imshow(Image.open('./resized/barbara.bmp'))
    ax[1].title.set_text('Resized Image')
    # plt.show()

    target = cv2.imread('./source/Set14/barbara.bmp')
    ref = cv2.imread('./resized/barbara.bmp')

    # metrics = compare_images(target, ref)
    # print("PSNR: {}".format(metrics[0]))
    # print("MSE: {}".format(metrics[1]))
    # print("SSIM: {}".format(metrics[2]))

    # Build train dataset
    import h5py

    names = sorted(os.listdir('./Train'))

    data = []
    label = []

    for name in names:
        fpath = './Train/' + name
        hr_img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
        shape = hr_img.shape


        # resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (int(shape[1] / 2), int(shape[0] / 2)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        width_range = int((shape[0] - 16 * 2) / 16)
        height_range = int((shape[1] - 16 * 2) / 16)

        for k in range(width_range):
            for j in range(height_range):
                x = k * 16
                y = j * 16

                hr_patch = hr_img[x: x + 32, y: y + 32]
                lr_patch = lr_img[x: x + 150, y: y + 150]

                hr_patch = hr_patch.astype(np.float32) / 255.
                lr_patch = lr_patch.astype(np.float32) / 255.

                hr = np.zeros((3, 4, 4), dtype=np.double)
                lr = np.zeros((3, 150, 150), dtype=np.double)

                # hr[0, :, :] = hr_patch[6:-6, 6: -6]
                # lr[0, :, :] = lr_patch[ :, :-1]

                label.append(hr)
                data.append(lr)

    data = np.array(data, dtype=np.float32)
    label = np.array(label, dtype=np.float32)
    # print(data.shape, label.shape)


    with h5py.File('train.h5', 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)
        h.create_dataset('label', data=label, shape=label.shape)

    names = sorted(os.listdir('./source/Set14'))
    nums = len(names)

    data_test = np.zeros((nums * 30, 3, 150, 150), dtype=np.double)
    label_test = np.zeros((nums * 30, 3, 4, 4), dtype=np.double)

    for i, name in enumerate(names):
        fpath = './source/Set14/' + name
        hr_img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
        shape = hr_img.shape

        # resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (int(shape[1] / 2), int(shape[0] / 2)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        # Produce random crop
        x = np.random.randint(0, min(shape[0], shape[1]) - 32, 30)
        y = np.random.randint(0, min(shape[0], shape[1]) - 32, 30)

        for j in range(30):
            lr_patch = lr_img[x[j]:x[j] + 32, y[j]:y[j] + 32]
            hr_patch = hr_img[x[j]:x[j] + 32, y[j]:y[j] + 32]

            lr_patch = lr_patch.astype(np.float32) / 255.
            hr_patch = hr_patch.astype(np.float32) / 255.

            # data_test[i * 30 + j, 0, :, :] = lr_patch
            # label_test[i * 30 + j, 0, :, :] = hr_patch[6: -6, 6: -6]

        with h5py.File('test.h5', 'w') as h:
            h.create_dataset('data', data=data_test, shape=data_test.shape)
            h.create_dataset('label', data=label_test, shape=label_test.shape)

    pre_train=load_main_model(VGG16)
    srcnn_model = model_rdn(pre_train)


    with h5py.File('./train.h5', 'r') as h:
            data = np.array(h.get('data'))
            label = np.array(h.get('label'))
            X_train = np.transpose(data, (0, 2, 3, 1))
            y_train = np.transpose(label, (0, 2, 3, 1))

    with h5py.File('./test.h5', 'r') as h:
            data = np.array(h.get('data'))
            label = np.array(h.get('label'))
            X_test = np.transpose(data, (0, 2, 3, 1))
            y_test = np.transpose(label, (0, 2, 3, 1))

    # print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)


    checkpoint_path = './srcnn/cp-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_best_only=True,
                                                    save_weights_only=True, verbose=0)
    f = open('results.txt', 'w')
    ep = [1, 5,10,50,150,200,250]
    for i in ep:
        srcnn_model.fit(X_train, y_train, batch_size=64, validation_data=(X_test, y_test),
                        callbacks=[checkpoint], epochs=i, verbose=False)

    # fig, ax = plt.subplots(figsize=(15, 10))
    # ax.imshow(Image.open('./source/Set14/barbara.bmp'))
    # ax.title.set_text("Original Image")
    # plt.show()

        try:
            os.listdir('./output')
        except:
            os.mkdir('./output')

        target = cv2.imread('./source/Set14/barbara.bmp', cv2.IMREAD_COLOR)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
        shape = target.shape

    # Resize down by scale of 2
        Y_img = cv2.resize(target[:, :, 0], (int(shape[1] / 2), int(shape[0] / 2)), cv2.INTER_CUBIC)

    # Resize up to orignal image
        Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
        target[:, :, 0] = Y_img
        target = cv2.cvtColor(target, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite('./output/input.jpg', target)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(Image.open('./output/input.jpg'))
        ax.title.set_text("Distorted Image")
    # plt.show()

        Y = np.zeros((nums * 30, 3, 150, 150), dtype=np.double)
        Y_pre = np.transpose(Y, (0, 2, 3, 1))
    # Normalize
    # Y[0, :, :, 0] = Y_img.astype(np.float32) / 255.


    # Predict
        pre = srcnn_model.predict(Y_pre, batch_size=1) * 255.

    # Post process output
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(np.uint8)

    # Copy y channel back to image and convert to BGR
        output = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
    # output[6: -6, 6: -6, 0] = pre[0, :, :, 0]
        output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)

    # Save image
        cv2.imwrite('./output/output.jpg', output)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(Image.open('./output/output.jpg'))
        ax.title.set_text("Predicted Image")
    # plt.show()

        original = cv2.imread('./source/Set14/barbara.bmp')
        distorted = cv2.imread('./output/input.jpg')
        predicted = cv2.imread('./output/output.jpg')



        metrics = compare_images(original, distorted)
        f.write(
        "Метрики оригинал/искаженное, эпоха № " + str(i) + '\n' + "PSNR: {}".format(metrics[0]) + '\n' + "MSE: {}".format(
            metrics[1]) + '\n' + "SSIM: {}".format(metrics[2]) + '\n')
        print("Метрики оригинал/искаженное")
        print("PSNR: {}".format(metrics[0]))
        print("MSE: {}".format(metrics[1]))
        print("SSIM: {}".format(metrics[2]))

        metrics = compare_images(original, predicted)
        f.write("Метрики оригинал/предсказаное, эпоха № " + str(i) + '\n' + "PSNR: {}".format(metrics[0]) + '\n' + "MSE: {}".format(
            metrics[1]) + '\n' + "SSIM: {}".format(metrics[2]) + '\n')
        print("Метрики оригинал/предсказаное")
        print("PSNR: {}".format(metrics[0]))
        print("MSE: {}".format(metrics[1]))
        print("SSIM: {}".format(metrics[2]))

    f.close()
