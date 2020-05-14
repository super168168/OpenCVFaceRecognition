import cv2 as cv
import os

array_of_img = [] # this if for store all of the image data
# this function is for read image,the input is directory name
def read_directory(directory_name):
    # this loop is for read each image in this foder,directory_name is the foder name with library.
    for filename in os.listdir(r"./"+directory_name):
        print(filename) #just for test
        #img is used to store the image data
        img = cv.imread(directory_name + "/" + filename)
        array_of_img.append(img)
        # print(array_of_img)
    # cv.imshow("test", array_of_img[0])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    print(array_of_img[0])

def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

def generate():
    face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    # eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    count = 0
    for frame in array_of_img:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray, None, fx=0.5, fy=0.5)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv.resize(gray[y:y + h, x:x + w], (200, 200))
            mkdir('library/face_data/'+name);
            cv.imwrite('library/face_data/'+name+'/%s.jpg' % str(count), f )
            print(count)
            count += 1

if __name__ == "__main__":
    name = input();
    read_directory('library/img_source/'+name)
    generate()
