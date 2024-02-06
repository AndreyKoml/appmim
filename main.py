from tkinter import *
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import HandTrackingModule as htm
import os


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 60)

width, height = 760, 540

cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

hands = mp.solutions.hands.Hands(max_num_hands=1)
draw_h = mp.solutions.drawing_utils


app = Tk()
app.title("MimiVision")
app.resizable(False, False)
app['bg'] = '#fefefa'
app.iconbitmap("iconapp.ico")
app.bind('<Escape>', lambda e: app.quit())

label_widget = Label(app)
label_widget.pack()

canvas = Canvas(app, width=1280, height=870, highlightthickness=0)
canvas.pack()

folderPath = "fingers"
fingerList = os.listdir(folderPath)
overlayList = []
for imgPath in fingerList:
    image = cv2.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

detector = htm.handDetector(detectionCon=0.75, maxHands=1)
totalFingers = 0

txt1 = Label(app, text="Приветствуем! Для начала работы убедитесь, что камера подключена к устройству.", bg="#fefefa", font=("Sans-serif", 14))
def open_camera():

    _, frame = cam.read()

    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    captured_image = Image.fromarray(opencv_image)

    photo_image = ImageTk.PhotoImage(image=captured_image)

    label_widget.photo_image = photo_image

    label_widget.configure(image=photo_image)

    label_widget.after(10, open_camera)

def fing_det():
    while True:

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

        success, img = cam.read()
        img = cv2.flip(img, 1)

        detector = htm.handDetector(detectionCon=0.75, maxHands=1)
        totalFingers = 0

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)

        if lmList:
            fingersUp = detector.fingersUp()
            totalFingers = fingersUp.count(1)

        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

        cv2.imshow("MimiVis_FingerCount", img)


button1 = PhotoImage(file="butn1.png")
Button(app, image=button1, highlightthickness=0, bd=0, command=open_camera).place(x=490, y=670)

button2 = PhotoImage(file="but2.png")
Button(app, image=button2, highlightthickness=0, bd=0, command=fing_det).place(x=490, y=770)

txt1.pack()

app.mainloop()




