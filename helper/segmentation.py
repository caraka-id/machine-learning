import cv2
import numpy as np
import tensorflow as tf


bali_labels = ['Ba', 'Ca', 'Da', 'Ga', 'Ha',
                'Ja', 'Ka', 'La', 'Ma', 'Na',
                'Nga', 'Nya', 'Pa',
                'Pengangge suara - Pepet',
                'Pengangge suara - Suku',
                'Pengangge suara - Taleng',
                'Pengangge suara - Taleng Tedong',
                'Pengangge suara - Tedong',
                'Pengangge suara - Ulu',
                'Pengangge tengenan - Adeg Adeg',
                'Pengangge tengenan - Bisah (h)',
                'Pengangge tengenan - Cecek (ng)',
                'Pengangge tengenan - Surang (r)',
                'Ra', 'Sa', 'Ta', 'Wa', 'Ya']

sunda_labels = ['a', 'ba', 'ca', 'da', 'e',
                'ee', 'eu', 'fa', 'ga', 'ha',
                'i', 'ja', 'ka', 'la', 'ma', 'na',
                'nga', 'nya', 'ou', 'pa', 'qa', 'ra',
                'sa', 'ta', 'u', 'va', 'vowels_e',
                'vowels_ee', 'vowels_eu', 'vowels_h',
                'vowels_i', 'vowels_la', 'vowels_ng',
                'vowels_o', 'vowels_r', 'vowels_ra',
                'vowels_u', 'vowels_x', 'vowels_ya',
                'wa', 'xa', 'ya', 'za']

lampung_labels = ['a', 'ai', 'au', 'ba', 'ca', 'da', 'e', 'ee', 'ga',
                  'gha', 'h', 'ha', 'i', 'ja', 'ka', 'la', 'ma', 'n',
                  'na', 'nengen', 'ng', 'nga', 'nya', 'o', 'pa', 'r',
                  'ra', 'sa', 'ta', 'u', 'wa', 'ya']


def load_model(aksara_type=None):
    if aksara_type == 'Bali':
        MODEL_PATH = '../models/model_bali_v1.h5'
    elif aksara_type == 'Sunda':
        MODEL_PATH = '../models/model_sunda.h5'
    elif aksara_type == 'Lampung':
        MODEL_PATH = '../models/model_lampung.h5'
    else:
        raise ValueError('Aksara not in our reached')

    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_input_image(image):
    resized_image = cv2.resize(image, (150, 150))
    if resized_image.shape[-1] == 1:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    return resized_image


def predict_character(model, character_image, aksara_type=None):
    input_image = preprocess_input_image(character_image)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)
    predicted_label_index = np.argmax(predictions)

    if aksara_type == 'Bali':
        return bali_labels[predicted_label_index]
    elif aksara_type == 'Sunda':
        return sunda_labels[predicted_label_index]
    elif aksara_type == 'Lampung':
        return lampung_labels[predicted_label_index]


def predict_characters(model, image_path, list_of_boxes, aksara_type):
    image = cv2.imread(image_path)

    predicted_labels = []
    for box in list_of_boxes:
        x1, y1, x2, y2 = box
        character_image = image[y1:y2, x1:x2]
        predicted_label = predict_character(model, character_image, aksara_type)
        predicted_labels.append(predicted_label)

        print(f'Bounding Box Coordinates: ({x1}, {y1}, {x2}, {y2})')

    return predicted_labels


def segment_characters(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    threshold = cv2.threshold(blur, 0.5, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    boxes.sort(key=lambda x: (x[0], x[1]))

    cv2.imshow('Image with Bounding Box', image)
    cv2.imshow('Image with Threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return boxes


def predict_words(labels, type=None):
    for id, label in enumerate(labels):
        print(f'Predicted Class {id}: {label}')

    words = []

    for id, label in enumerate(labels):
        if type == 'Bali':
            if label == 'Pengangge tengenan - Bisah (h)':
                label = 'h'

            elif label == 'Pengangge tengenan - Cecek (ng)':
                label = 'ng'

            elif label == 'Pengangge tengenan - Surang (r)':
                label = 'r'

            elif label == 'Pengangge tengenan - Adeg Adeg' and id == len(labels):
                label = labels[id - 1].replace(list(labels[id - 1])[1], '')
                words.pop()

            elif label == 'Pengangge tengenan - Adeg Adeg' and id != len(labels):
                label = 'Pengangge suara - Taleng'

            elif label == 'Pengangge suara - Suku':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'u')
                words.pop()

            elif label == 'Pengangge suara - Ulu':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'i')
                words.pop()

            elif label == 'Pengangge suara - Pepet':
                label = labels[id + 1].replace(list(labels[id + 1])[1], 'ee')
                labels[id + 1] = ''

            if label == 'Pengangge suara - Taleng':
                label = labels[id + 1].replace(list(labels[id + 1])[1], 'e')
                labels[id + 1] = ''

        if type == 'Sunda':
            if label == 'sa' and labels[id + 1] == 'sa':
                label = ''

            if label == 'ka' and labels[id + 1] == 'ka':
                label = ''

            if label == 'vowels_o':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'o')
                words.pop()
            elif label == 'vowels_e':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'e')
                label = ''
            elif label == 'vowels_ee':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'ee')
                label = ''
            elif label == 'vowels_eu':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'eu')
                label = ''
            elif label == 'vowels_i':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'i')
                label = ''
            elif label == 'vowels_u':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'u')
                words.pop()
            elif label == 'vowels_la':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'la')
                words.pop()
            elif label == 'vowels_ra':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'ra')
                words.pop()
            elif label == 'vowels_ya':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'ya')
                words.pop()
            elif label == 'vowels_x':
                label = labels[id - 1].replace(list(labels[id - 1])[1], '')
                words.pop()

            elif label == 'vowels_h':
                label = 'h'
            elif label == 'vowel_r':
                labels[id + 1] = (labels[id + 1]) + 'r'
                label = ''
            elif label == 'vowels_ng':
                labels[id + 1] = (labels[id + 1]) + 'ng'
                label = ''

        if type == 'Lampung':
            if label == 'i':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'i')
                label = ''
            elif label == 'ee':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'ee')
                label = ''
            elif label == 'e':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'e')
                label = ''
            elif label == 'n':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'n')
                label = ''
            elif label == 'ng':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'ng')
                label = ''
            elif label == 'r':
                labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'r')
                label = ''
            elif label == 'o':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'o')
                words.pop()
            elif label == 'au':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'au')
                words.pop()
            elif label == 'ai':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'i')
                words.pop()
            elif label == 'u':
                label = labels[id - 1].replace(list(labels[id - 1])[1], 'u')
                words.pop()
            elif label == 'nengen':
                label = labels[id - 1].replace(list(labels[id - 1])[1], '')
                words.pop()
            elif label == 'h':
                label = 'h'


            words.append(label)
        return ''.join(word for word in words).lower()


if __name__ == '__main__':
    TYPE = 'Bali'
    IMAGE_PATH = f'../test_images/Aksara_{TYPE}/pergi_ke_bali.png'

    model = load_model(TYPE)

    list_of_boxes = segment_characters(IMAGE_PATH)
    predicted_labels = predict_characters(model, IMAGE_PATH, list_of_boxes, TYPE)

    predicted_words = predict_words(predicted_labels, TYPE)
    print(predicted_words)
