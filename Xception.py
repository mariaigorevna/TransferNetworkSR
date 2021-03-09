from tensorflow.keras.applications.xception import Xception
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from skimage.metrics import structural_similarity as ssim
import os
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

DATA_PATH = './Train/'
TEST_PATH = './source/Set14/'
Random_Crop = 30
Patch_size = 32
label_size = 20
conv_side = 6
scale = 2
EPOCHS = 5
BLOCK_STEP = 16
BLOCK_SIZE = 32

# Оценка сигнал/шум
def psnr(target, ref):

    target_data = target.astype(np.float32)
    ref_data = ref.astype(np.float32)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = np.sqrt(np.mean(diff ** 2.))

    return 20 * np.log10(255. / rmse)

# Среднеквадратичная ошибка (MSE)
def mse(target, ref):
    target_data = target.astype(np.float32)
    ref_data = ref.astype(np.float32)
    err = np.sum((target_data - ref_data) ** 2)

    err /= np.float(target_data.shape[0] * target_data.shape[1])
    return err

# Метрика структурного сходства
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel=True))
    return scores

# Функция изменения размера изображения
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

# Загрузка предобученной модели
def load_xception():
    input_tensor = (150, 150, 3)
    pre_model = Xception(weights='imagenet', include_top=False, input_shape=input_tensor)
    pre_model.trainable = False
    return pre_model

# Пример оригинального и измененного изображения
def example():
    from PIL import Image
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(Image.open('./source/Set14/barbara.bmp'))
    ax[0].title.set_text('Original Image')
    ax[1].imshow(Image.open('./resized/barbara.bmp'))
    ax[1].title.set_text('Resized Image')
    plt.show()

# Изменение тестового набора
def prepare_data(_path):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    data = np.zeros((nums * Random_Crop, 1, Patch_size, Patch_size), dtype=np.double)
    label = np.zeros((nums * Random_Crop, 1, label_size, label_size), dtype=np.double)

    for i in range(nums):
        name = _path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        shape = hr_img.shape

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]

        # two resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (int(shape[1] / scale), int(shape[0] / scale)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        # produce Random_Crop random coordinate to crop training img
        Points_x = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)
        Points_y = np.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)

        for j in range(Random_Crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            data[i * Random_Crop + j, 0, :, :] = lr_patch
            label[i * Random_Crop + j, 0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]
            # cv2.imshow("lr", lr_patch)
            # cv2.imshow("hr", hr_patch)
            # cv2.waitKey(0)
    return data, label

# Изменение тренировочного набора
def prepare_crop_data(_path):
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    data = []
    label = []

    for i in range(nums):
        name = _path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
        shape = hr_img.shape

        # two resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (int(shape[1] / scale), int(shape[0] / scale)))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        width_num =int((shape[0] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP)
        height_num = int((shape[1] - (BLOCK_SIZE - BLOCK_STEP) * 2) / BLOCK_STEP)

        for k in range(width_num):
            for j in range(height_num):
                x = k * BLOCK_STEP
                y = j * BLOCK_STEP
                hr_patch = hr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]
                lr_patch = lr_img[x: x + BLOCK_SIZE, y: y + BLOCK_SIZE]

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                lr = np.zeros((1, Patch_size, Patch_size), dtype=np.double)
                hr = np.zeros((1, label_size, label_size), dtype=np.double)

                lr[0, :, :] = lr_patch
                hr[0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]

                data.append(lr)
                label.append(hr)

    data = np.array(data, dtype=float)
    label = np.array(label, dtype=float)
    return data, label

# Запись в hdf5
def write_hdf5(data, labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    x = data.astype(np.float32)
    y = labels.astype(np.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        # h.create_dataset()

# Модель
def model():
    SRCNN = tf.keras.Sequential(name='SRCNN')
    SRCNN.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(9, 9),
                                     padding='VALID',
                                     use_bias=True,
                                     input_shape=(None, None, 1),
                                     kernel_initializer='glorot_uniform',
                                     activation='relu'))
    SRCNN.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                     padding='SAME',
                                     use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     activation='relu'))
    SRCNN.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 5),
                                     padding='VALID',
                                     use_bias=True,
                                     kernel_initializer='glorot_uniform',
                                     activation='linear'))
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)

    # Compile model
    SRCNN.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])

    return SRCNN

# Получение тестовых и тренировочных данных из файлов .h5
def train_test_data():
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

    return X_train, y_train, X_test, y_test

# Оригинальное измененное и предсказанное изображения
def orig_distr_pred(srcnn_model):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(Image.open('./source/Set14/barbara.bmp'))
    ax.title.set_text("Original Image")
    plt.show()

    try:
        os.listdir('./output')
    except:
        os.mkdir('./output')

    target = cv2.imread('./source/Set14/barbara.bmp', cv2.IMREAD_COLOR)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
    shape = target.shape

    # Resize down by scale of 2
    Y_img = cv2.resize(target[:, :, 0], (int(shape[1] / scale), int(shape[0] / scale)), cv2.INTER_CUBIC)

    # Resize up to orignal image
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    target[:, :, 0] = Y_img
    target = cv2.cvtColor(target, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('./output/input.jpg', target)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(Image.open('./output/input.jpg'))
    ax.title.set_text("Distorted Image")
    plt.show()

    Y = np.zeros((1, target.shape[0], target.shape[1], 1), dtype=np.float32)
    # Normalize
    Y[0, :, :, 0] = Y_img.astype(np.float32) / 255.

    # Predict
    pre = srcnn_model.predict(Y, batch_size=1) * 255.

    # Post process output
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)

    # Copy y channel back to image and convert to BGR
    output = cv2.cvtColor(target, cv2.COLOR_BGR2YCrCb)
    output[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)

    # Save image
    cv2.imwrite('./output/output.jpg', output)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(Image.open('./output/output.jpg'))
    ax.title.set_text("Predicted Image")
    plt.show()

# Получение метрик оригинального и измененного изображения и оригинального и предсказанного изображения
def metric():
    original = cv2.imread('./source/Set14/barbara.bmp')
    distorted = cv2.imread('./output/input.jpg')
    predicted = cv2.imread('./output/output.jpg')

    metrics = compare_images(original, distorted)
    print("Метрика оригинального и измененного изображения")
    print("PSNR: {}".format(metrics[0]))
    print("MSE: {}".format(metrics[1]))
    print("SSIM: {}".format(metrics[2]))

    metrics = compare_images(original, predicted)
    print("Метрика оригинального и предсказанного изображения")
    print("PSNR: {}".format(metrics[0]))
    print("MSE: {}".format(metrics[1]))
    print("SSIM: {}".format(metrics[2]))

# Обучение
def train():
    data, label = prepare_crop_data(DATA_PATH)
    write_hdf5(data, label, "train.h5")
    data, label = prepare_data(TEST_PATH)
    write_hdf5(data, label, "test.h5")

    srcnn_model = model()
    srcnn_model.summary()

    X_train, y_train, X_test, y_test = train_test_data()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    checkpoint_path = './srcnn/cp-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, save_best_only=True,
                                                    save_weights_only=True, verbose=0)

    srcnn_model.fit(X_train, y_train, batch_size=64, validation_data=(X_test, y_test),
                    callbacks=[checkpoint], shuffle=True, epochs=EPOCHS, verbose=False)
    return srcnn_model

# Предсказание
def predict(srcnn_model):
    orig_distr_pred(srcnn_model)
    metric()

if __name__ == '__main__':
    srcnn_model = train()
    predict(srcnn_model)



