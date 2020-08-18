import cv2


def convert_image(img):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = b_channel
    return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))


def find_face_positions(img, classifier):
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    faces = classifier.detectMultiScale(image_gray)
    return faces


def replace_faces(img, replacement, faces, name='test.jpg'):
    for (x, y, w, h) in faces:
        w = w
        replacement = cv2.resize(replacement, (w, h), cv2.INTER_CUBIC)
        roi = img[y:y + h, x:x + w]
        img2gray = cv2.cvtColor(replacement, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img2_fg = cv2.bitwise_and(replacement, replacement, mask=mask)
        dst = cv2.add(img1_bg, img2_fg)
        img[y: y + h, x:x + w] = dst
        cv2.imwrite(f'data/res_imgs/{name}', img)


if __name__ == '__main__':
    f_name = 'jora.jpg'
    img1 = y if (y := cv2.imread(f'data/scr_imgs/{f_name}', cv2.IMREAD_UNCHANGED)).shape[2] == 4 else convert_image(y)
    img2 = cv2.imread('data/replacement_imgs/Vasya_face.png', cv2.IMREAD_UNCHANGED)
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(cv2.samples.findFile('data/haarcascades/haarcascade_frontalface_alt.xml'))
    faces = find_face_positions(img1, face_cascade)
    replace_faces(img1, img2, faces, name=f_name)
