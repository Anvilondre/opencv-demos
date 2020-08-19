import cv2
from itertools import cycle


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


def run(as_file=False):
    f_name = 'kavo.jpg'

    source_img = y if (y := cv2.imread(f'data/scr_imgs/{f_name}', cv2.IMREAD_UNCHANGED)).shape[2] == 4 \
        else convert_image(y)  # We need to make sure there are exactly 4 channels: RGB + Alpha
    face_mask = cv2.imread('data/replacement_imgs/Vasya_face.png', cv2.IMREAD_UNCHANGED)
    face_mask = cycle([face_mask])  # Creating an iterator so we can cycle through different face replacements in future

    # Initializing a built-in cascade classifier to detect faces
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(cv2.samples.findFile('data/haarcascades/haarcascade_frontalface_alt.xml'))
    faces = find_face_positions(source_img, face_cascade)

    # Image with added/replaced faces
    img = add_faces(source_img, face_mask, faces)

    if as_file:
        cv2.imwrite(f'data/res_imgs/{f_name}', img)
    else:
        cv2.imshow(f_name, img)
        cv2.waitKey(0)


if __name__ == '__main__':
    run(as_file=False)
