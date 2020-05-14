import cv2 as cv
import time
import os

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

def generate_video(path,name):
    temp = 0
    face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    # eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    count = 0
    for filename in os.listdir(r"./" + path + '/' + name):
        print(filename)
        camera = cv.VideoCapture('library/video_source/'+ name + '/' + filename)
        while (True):
            ret, frame = camera.read()
            if (frame is None):
                break
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                img = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                f = cv.resize(gray[y:y + h, x:x + w], (200, 200))
                if temp%36 == 0 :
                    mkdir('library/face_data/' + name);
                    cv.imwrite('library/face_data/'+name+'/%s.jpg' % str(count), f)
                    print(count)
                    count += 1
                temp += 1
                # time.sleep(1)

            cv.imshow("camera", frame)
            if cv.waitKey(5) & 0xff == ord("q"):
                break

def generate_camera(name):
    temp = 0
    face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    # eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    count = 0
    camera = cv.VideoCapture(0)
    while (True):
        ret, frame = camera.read()
        if (frame is None):
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            img = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv.resize(gray[y:y + h, x:x + w], (200, 200))
            if temp % 36 == 0:
                mkdir('library/face_data/' + name);
                cv.imwrite('library/face_data/' + name + '/%s.jpg' % str(count), f)
                print(count)
                count += 1
            temp += 1
            # time.sleep(1)

            cv.imshow("camera", frame)
            if cv.waitKey(5) & 0xff == ord("q"):
                break

    camera.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    choose = input('From Video or Camera?\n1.Video\n2.Camera\n')
    if(choose is '1'):
        name = input();
        generate_video('library/video_source/',name)
    elif(choose is '2'):
        name = input();
        generate_camera(name)