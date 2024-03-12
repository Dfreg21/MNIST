import cv2
import operator

from tensorflow import keras

model = keras.models.load_model('CNN_MNIST.h5')
#model = keras.models.load_model('CNN_FMNIST.h5')
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray= cv2.GaussianBlur(gray,(3,3),0)
    ret, img_bn = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.resize(img_bn, (28,28))
     
    patron = binary.reshape(1,28,28,1)/255
     
    result = model.predict(patron)
    predict = {'cero': result[0][0], 'uno': result[0][1], 'dos': result[0][2], 'tres': result[0][3], 'cuatro': result[0][4], 
              'cinco': result[0][5], 'seis': result[0][6], 'siete': result[0][7], 'ocho': result[0][8], 'nueve': result[0][9]}
    predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True)
    print(predict[0][0])
    
    cv2.imshow('frame', img_bn)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cam.release()
cv2.destroyAllWindows()
