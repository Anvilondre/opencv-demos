import time

import cv2
import telebot
from itertools import cycle
from data.Token import token

bot = telebot.TeleBot(token=token)


@bot.message_handler(content_types=['photo'])
def handle_image(message):
    global counter  # Just for naming purposes
    counter += 1
    raw = message.photo[-1].file_id

    src_path = f'data/scr_imgs/{counter}.jpg'
    res_path = f'data/res_imgs/{counter}.jpg'

    # Downloading a picture so we can precess it with OpenCV
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)

    with open(src_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    pic = process_picture(src_path)
    cv2.imwrite(res_path, pic)  # Saving the result so we can send it to user

    with open(res_path, 'rb') as f:
        bot.send_photo(message.chat.id, f)


def convert_image(img):
    """Just copies the blue channel and adds it as an alpha channel."""
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = b_channel
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def find_face_positions(img, classifier):
    """Finds all the faces on image and returns an array of (x, y, w, h) tuples."""
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    faces = classifier.detectMultiScale(image_gray)
    return faces


def overlay_image(background, foreground, x, y, w, h):
    """Scales foreground image to (w, h) size and adds it to the background image at (x, y) - top left corner."""
    replacement = cv2.resize(foreground, (w, h), cv2.INTER_CUBIC)
    img = background.copy()
    roi = img[y:y + h, x:x + w]
    img2gray = cv2.cvtColor(replacement, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img2_fg = cv2.bitwise_and(replacement, replacement, mask=mask)
    dst = cv2.add(img1_bg, img2_fg)
    img[y: y + h, x:x + w] = dst
    return img


def add_faces(img, replacements, faces):
    """Replaces all the faces if any found. Otherwise just adds a face in the bottom middle."""
    res_img = img.copy()
    if faces is None or len(faces) == 0:
        w, h = res_img.shape[1] // 5, res_img.shape[0] // 3
        x, y = res_img.shape[1] // 2 - w // 2, res_img.shape[0] - h
        res_img = overlay_image(res_img, next(replacements), x, y, w, h)
    for face in faces:
        res_img = overlay_image(res_img, next(replacements), *face)

    return res_img


def process_picture(file_name):
    # We need to make sure there are exactly 4 channels: RGB + Alpha
    source_img = y if (y := cv2.imread(file_name, cv2.IMREAD_UNCHANGED)).shape[2] == 4 else convert_image(y)

    # TODO: Add different face masks functionality
    face_mask = cv2.imread('data/replacement_imgs/Vasya_face.png', cv2.IMREAD_UNCHANGED)
    face_mask = cycle([face_mask])  # Creating an iterator so we can cycle through different face replacements in future

    # Initializing a built-in cascade classifier to detect faces
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(cv2.samples.findFile('data/haarcascades/haarcascade_frontalface_alt.xml'))
    faces = find_face_positions(source_img, face_cascade)

    # Image with added/replaced faces
    img = add_faces(source_img, face_mask, faces)

    return img


if __name__ == '__main__':

    # TODO: Refactor in OOP style, removing redundant object creation on each iteration

    global counter
    counter = 0

    # If bot encounters some unpredictable stuff just wait some time and reload
    while True:
        try:
            bot.polling()
        except Exception:
            time.sleep(5)
