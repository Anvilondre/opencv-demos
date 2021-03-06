import time

import cv2
import telebot
from itertools import cycle
from data.Token import token
from glob import glob

bot = telebot.TeleBot(token=token)


def write_counter():
    global counter
    with open('data/counter.txt', 'w') as w:
        w.write(str(counter))


@bot.message_handler(content_types=['photo'])
def handle_image(message):
    global counter, deep_face  # Counter is just for naming purposes
    counter += 1
    write_counter()
    raw = message.photo[-1].file_id

    src_path = f'data/scr_imgs/{counter}.jpg'
    res_path = f'data/res_imgs/{counter}.jpg'

    # Downloading a received picture to process it with OpenCV
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)

    with open(src_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    pic = deep_face.process_picture(src_path)
    cv2.imwrite(res_path, pic)  # Saving the result to send it back

    with open(res_path, 'rb') as f:
        bot.send_photo(message.chat.id, f)


def find_face_positions(img, classifier):
    """Finds all the faces on image and returns an array of (x, y, w, h) tuples."""
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    faces = classifier.detectMultiScale(image_gray)
    return faces


class Face_Replacer:

    def __init__(self, mask_images: iter):

        # Initializing a built-in cascade classifier to detect faces
        self.face_cascade = cv2.CascadeClassifier()
        self.face_cascade.load(cv2.samples.findFile('data/haarcascades/haarcascade_frontalface_alt.xml'))

        self.mask_images = mask_images

    def overlay_image(self, background, x, y, w, h):
        """Scales foreground image to (w, h) size and adds it to the background image at (x, y) - top left corner."""
        replacement = cv2.resize(next(self.mask_images), (w, h), cv2.INTER_CUBIC)
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

    def add_faces(self, img, faces):
        """Replaces all the faces if any found. Otherwise just adds a face in the bottom middle."""
        res_img = img.copy()
        if faces is None or len(faces) == 0:
            w, h = res_img.shape[1] // 5, res_img.shape[0] // 3
            x, y = res_img.shape[1] // 2 - w // 2, res_img.shape[0] - h
            res_img = self.overlay_image(res_img, x, y, w, h)
        for face in sorted(faces, key=lambda x: x[2] * x[3]):
            face[2], face[3] = 0.9 * face[2], 1.1 * face[3]
            face[0] += face[2] * 0.05
            res_img = self.overlay_image(res_img, *face)

        return res_img

    def process_picture(self, file_name):
        """Connects everything together."""
        # We need to make sure there are exactly 4 channels: BGR + Alpha
        source_img = y if (y := cv2.imread(file_name, cv2.IMREAD_UNCHANGED)).shape[2] == 4 \
            else cv2.cvtColor(y, cv2.COLOR_BGR2BGRA)

        # Detect faces on image
        faces = find_face_positions(source_img, self.face_cascade)

        # Image with added/replaced faces
        img = self.add_faces(source_img, faces)

        return img


if __name__ == '__main__':
    global counter, deep_face

    # Creating an iterator for replacement faces
    face_masks = cycle(cv2.imread(y, cv2.IMREAD_UNCHANGED) for y in glob('data/replacement_imgs/*.png'))
    deep_face = Face_Replacer(face_masks)

    with open('data/counter.txt', 'r') as r:
        counter = int(r.read())

    # If bot encounters some unpredictable stuff just wait some time and reload
    while True:
        try:
            bot.polling()
        except Exception:
            print(f"Something went wrong! Image number: {counter}")
            time.sleep(5)
