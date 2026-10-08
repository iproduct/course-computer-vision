import cv2

if __name__ == '__main__':
    img = cv2.imread('data/lena.bmp')
    if img is not None:
        print('image loaded')
        print(img.shape)
        print(type(img))
        print(img.dtype)
        cv2.imshow('img', img)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img_gray', img_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('no image')