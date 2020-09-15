#!/usr/bin/env python

import smtplib
from PyQt5.QtGui import*
from PyQt5.QtWidgets import*
import sys
from PyQt5.uic import loadUiType
import threading
from vehicle import vehicle
import tensorflow as tf
detection_graph = tf.Graph()
import time
import cv2
import numpy as np
import os
from PIL import Image
from utils import label_map_util
import qimage2ndarray
from utils import visualization_utils as vis_util
from openalpr import Alpr





path_root="test_video.avi"


MODEL_NAME = 'Cars'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/output_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('Cars', 'car_label_map.pbtxt')

NUM_CLASSES = 1

sdetection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

cap=cv2.VideoCapture(path_root) # 0 stands for very first webcam attach
filename="testoutput.avi"
codec=cv2.VideoWriter_fourcc('m','p','4','v')#fourcc stands for four character code
framerate=10
resolution=(640,480)

VideoFileOutput=cv2.VideoWriter(filename,codec,framerate, resolution)
ret,imgF=cap.read()
imgF=Image.fromarray(imgF)
im_width, im_height = imgF.size
xl1=0
xl2=im_width-1
yl1=im_height*0.5
yl2=yl1
ml1=(yl2-yl1)/(xl2-xl1)
intcptl1=yl1-ml1*xl1

count=0
xl3=0
xl4=im_width-1
yl3=im_height*0.25
yl4=yl3
ml2=(yl4-yl3)/(xl4-xl3)
intcptl2=yl3-ml2*xl3

xl5=0
xl6=im_width-1
yl5=im_height*0.1
yl6=yl5
ml3=(yl6-yl5)/(xl6-xl5)
intcptl3=yl5-ml3*xl5
ret=True
start=time.time()
c=0
sesser=tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

###########################################################################################################

ui, _ = loadUiType("gui.ui")
vehicles = []

import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="cardb"
)




class MainApp(QMainWindow, ui):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)


        self.label_DETECD_SERACH.hide()
        self.label_illegal.hide()
        self.bimg=[]
        self.best_candidate=[]
        self.name_eligal=[]
        self.image=[]
        self.InitUI()





    def disconnect(self):
        self.stop_event = threading.Event()
        self.ccc_thread = threading.Thread(target = self.main_sess, args = (self.stop_event, ))
        self.ccc_thread.start()
    def InitUI(self):
        MainApp.disconnect(self)

    def find_histogram(self, clt):
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        return hist

    def matchVehicles(currentFrameVehicles, im_width, im_height, image):
        if len(vehicles) == 0:
            for box, color in currentFrameVehicles:
                (y1, x1, y2, x2) = box
                (x, y, w, h) = (
                x1 * im_width, y1 * im_height, x2 * im_width - x1 * im_width, y2 * im_height - y1 * im_height)
                X = int((x + x + w) / 2)
                Y = int((y + y + h) / 2)
                if Y > yl5:
                    # cv2.circle(image,(X,Y),2,(0,255,0),4)
                    # print('Y=',Y,'  y1=',yl1)
                    vehicles.append(vehicle((x, y, w, h)))


        else:
            for i in range(len(vehicles)):
                vehicles[i].setCurrentFrameMatch(False)
                vehicles[i].predictNext()
            for box, color in currentFrameVehicles:
                (y1, x1, y2, x2) = box
                (x, y, w, h) = (
                x1 * im_width, y1 * im_height, x2 * im_width - x1 * im_width, y2 * im_height - y1 * im_height)
                # print((x1*im_width,y1*im_height,x2*im_width,y2*im_height),'\n',(x,y,w,h))
                index = 0
                ldistance = 999999999999999999999999.9
                X = int((x + x + w) / 2)
                Y = int((y + y + h) / 2)
                if Y > yl5:
                    # print('Y=',Y,'  y1=',yl1)
                    # cv2.circle(image,(X,Y),4,(0,0,255),8)
                    for i in range(len(vehicles)):
                        if vehicles[i].getTracking() == True:
                            # print(vehicles[i].getNext(),i)
                            distance = ((X - vehicles[i].getNext()[0]) ** 2 + (
                                        Y - vehicles[i].getNext()[1]) ** 2) ** 0.5

                            if distance < ldistance:
                                ldistance = distance
                                index = i

                    diagonal = vehicles[index].diagonal

                    if ldistance < diagonal:
                        vehicles[index].updatePosition((x, y, w, h))
                        vehicles[index].setCurrentFrameMatch(True)
                    else:
                        vehicles.append(vehicle((x, y, w, h)))

            for i in range(len(vehicles)):
                if vehicles[i].getCurrentFrameMatch() == False:
                    vehicles[i].increaseFrameNotFound()

    def checkRedLightCrossed(self,img):
        global count
        for v in vehicles:
            if v.crossed == False and len(v.points) >= 2:
                x1, y1 = v.points[0]
                x2, y2 = v.points[-1]
                if y1 > yl3 and y2 < yl3:
                    count += 1
                    v.crossed = True
                    bimg = img[int(v.rect[1]):int(v.rect[1] + v.rect[3]), int(v.rect[0]):int(v.rect[0] + v.rect[2])]
                    frame2 = bimg
                    self.bimg = bimg
                    img2 = Image.fromarray(frame2)
                    w, h = img2.size
                    asprto = w / h
                    frame2 = cv2.resize(frame2, (250, int(250 / asprto)))
                    cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                    alpr = Alpr("us", r"C:\OpenALPR\Agent\etc\openalpr\openalpr.conf",
                                r"C:\OpenALPR\Agent\usr\share\openalpr\runtime_data")
                    if not alpr.is_loaded():
                        print('Error loading OpenALPR')
                        sys.exit(1)
                    alpr.set_top_n(3)
                    results = alpr.recognize_ndarray(frame2)

                    ###########################################################################

                    resize = cv2.resize(cv2image2, (301, 351), interpolation=cv2.INTER_LINEAR)
                    image = qimage2ndarray.array2qimage(resize)
                    self.label_detect.setPixmap(QPixmap.fromImage(image))
                    ###########################################################################
                    for i, plate in enumerate(results['results']):
                        X1 = plate['coordinates'][0]['x']
                        Y1 = plate['coordinates'][0]['y']
                        X2 = plate['coordinates'][2]['x']
                        Y2 = plate['coordinates'][2]['y']
                        rimg = cv2image2[Y1:Y2, X1:X2]
                        frame3 = rimg
                        resize = cv2.resize(frame3, (301, 41), interpolation=cv2.INTER_LINEAR)
                        self.image = qimage2ndarray.array2qimage(resize)
                        self.label_numberplate.setPixmap(QPixmap.fromImage(self.image))

                        self.best_candidate = plate['candidates'][0]
                        # print('Plate #{}: {:7s} ({:.2f}%)'.format(i, best_candidate['plate'].upper(),
                        #                                           best_candidate['confidence']))
                        self.label_number_plate.setText('number plate:   ' + str(self.best_candidate['plate'].upper()))
                        detect=self.lineEdit_detect.text()

                        if detect == str(self.best_candidate['plate'].upper()):
                            self.label_DETECD_SERACH.show()
                            print("detect")#ayd2030
                        else:
                            self.label_DETECD_SERACH.hide()



                    ###########################################################################
                    self.name_eligal = r'detection\d- ' + self.best_candidate['plate'].upper() + '.jpg'
                    self.num_plate = r'numberplate\NP- ' + self.best_candidate['plate'].upper() + '.jpg'
                    cv2.imwrite(self.name_eligal, bimg)
                    cv2.imwrite(self.num_plate, resize)
                    MainApp.dis(self)

    def eligal(self, stop_event):
       # img = cv2.imread(self.name_eligal)
        import color
        image = Image.open(self.name_eligal)
        color.process_image(image)
        # print(self.best_candidate['plate'].upper())
        # print(color.final_des)

        hl= str(color.final_des)[1:-1]

        mycursor = mydb.cursor(buffered=True)
        mycursor.execute("SELECT * FROM carnumber  WHERE number = %s", (self.best_candidate['plate'].upper(),))
        myresult = mycursor.fetchone()
        if self.best_candidate['plate'].upper() == myresult[1] and hl==myresult[2]:
            print("legal")
            self.label_illegal.hide()

        else:
            print("eligal car move")
            MainApp.mail_thread(self)
            self.label_illegal.show()
            self.car_eligal = r'illegal\i- ' + self.best_candidate['plate'].upper() + '.jpg'
            cv2.imwrite(self.car_eligal, self.bimg)



    def dis(self):
        self.stop_event = threading.Event()
        self.ccc_thread = threading.Thread(target = self.eligal, args = (self.stop_event, ))
        self.ccc_thread.start()


    def checkSpeed(self, ftime, img):
        for v in vehicles:
            if v.speedChecked == False and len(v.points) >= 2:
                x1, y1 = v.points[0]
                x2, y2 = v.points[-1]
                if y2 < yl1 and y2 > yl3 and v.entered == False:
                    v.enterTime = ftime
                    v.entered = True
                elif y2 < yl3 and y2 > yl5 and v.exited == False:
                    v.exitTime = ftime
                    v.exited == False
                    v.speedChecked = True
                    speed = 60 / (v.exitTime - v.enterTime)
                    #print(speed)
                    bimg = img[int(v.rect[1]):int(v.rect[1] + v.rect[3]), int(v.rect[0]):int(v.rect[0] + v.rect[2])]
                    frame2 = bimg
                    img2 = Image.fromarray(frame2)
                    w, h = img2.size
                    asprto = w / h
                    frame2 = cv2.resize(frame2, (250, int(250 / asprto)))
                    cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                    resize = cv2.resize(cv2image2, (301, 351), interpolation=cv2.INTER_LINEAR)
                    image = qimage2ndarray.array2qimage(resize)
                    ##############################################################################################
                    self.label_speed.setText('speed:   '+str(speed)[:5]+ 'Km/hr')
                    if speed > 60:

                        self.label_overspeed.setPixmap(QPixmap.fromImage(image))
                        self.label_overspeed_2.setText('speed:   ' + str(speed)[:5] + 'Km/hr')
                        self.label_overspeed_plate.setText('number plate:   ' + self.best_candidate['plate'].upper())
                        self.label_numberplate_2.setPixmap(QPixmap.fromImage(self.image))
                        name_over = r'overspeed\o-' + self.best_candidate['plate'].upper() + ".jpg"  # str(time.time()) + '.jpg'
                        print(name_over)
                        cv2.imwrite(name_over,frame2)

                        MainApp.mail_thread(self)


                        ###############---------send mail----------------###########################################
                        sender = 'mail.abhayadev@gmail.com'
                        receivers = ['mail.devabhaya@gmail.com']

                        message = "detect over speed"+self.best_candidate['plate'].upper()

                        try:
                            smtpObj = smtplib.SMTP('smtp.gmail.com', 587)
                            smtpObj.starttls()
                            smtpObj.login("mail.abhayadev@gmail.com", "PASSWORD")
                            smtpObj.sendmail(sender, receivers, message)
                            print("Successfully sent email")
                            smtpObj.quit()
                        except smtplib.SMTPException as e:
                            print("Error: unable to send email", e)
                            smtpObj.quit()
 ##############################-###############################################################################################




                   #  name = r'C:\Users\abhayadev\Desktop\hackp\detect\t-' + self.best_candidate['plate'].upper() +".jpg"   #str(time.time()) + '.jpg'
                   #  print(name)
                   #  cv2.imwrite(name, bimg)
                   #  print("img write")

    def mail(self, stop_event):
        sender = 'mail.abhayadev@gmail.com'
        receivers = ['mail.devabhaya@gmail.com']

        message = "detect over speed" + self.best_candidate['plate'].upper()

        try:
            smtpObj = smtplib.SMTP('smtp.gmail.com', 587)
            smtpObj.starttls()
            smtpObj.login("mail.abhayadev@gmail.com", "abhayadev04864225426")
            smtpObj.sendmail(sender, receivers, message)
            print("Successfully sent email")
            smtpObj.quit()
        except smtplib.SMTPException as e:
            print("Error: unable to send email", e)
            smtpObj.quit()

    def mail_thread(self):
        self.stop_event = threading.Event()
        self.ccc_thread = threading.Thread(target=self.mail, args=(self.stop_event,))
        self.ccc_thread.start()
    def main1(self,sess=sesser):

        VIDEO_SOURCE = path_root
        cap = cv2.VideoCapture(VIDEO_SOURCE)


        ###-----------FOR REALTIME CAM----------#

        #cap = cv2.VideoCapture(0)
        while (1):
            fTime = time.time()
            _, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            img = image_np
            imgF, coords = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=2)
            #print("complete")

            MainApp.matchVehicles(coords, im_width, im_height, imgF)
            MainApp.checkRedLightCrossed(self,imgF)
            MainApp.checkSpeed(self, fTime, img)
            VideoFileOutput.write(image_np)
            # print('yola')
            frame = cv2.resize(image_np, (1020, 647))
            resize = cv2.resize(frame, (561, 351), interpolation=cv2.INTER_LINEAR)
            image = qimage2ndarray.array2qimage(resize)
            self.label_mainvideo.setPixmap(QPixmap.fromImage(image))
    def main_sess(self, stop_event):
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                #print("complete")
                sesser = sess
                MainApp.main1(self, sess)


def main():
    app= QApplication(sys.argv)
    window=MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()