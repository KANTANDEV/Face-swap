#importation
import sys
import cv2 as cv 

#charger le classificateur en cassacde preentrainer
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#Chargement de l'image

img = cv.imread('votreimage.jpg')

#Conversion de l'iamge en niveau de gris pour pouvoir la passer dans le classificateur

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Execution de la detection de visage 

faces = face_cascade.detectMultiScale(img_gray, 1.1, 8)


#verifie le nombre de visage 

if len(faces) != 2 :
    sys.exit('La photo doit contenir deux visages')
    

#Recuperation des dimention de chaque visages

x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]


#extraction des deux visages de l'image 

face1 = img[y1:y1+h1,x1:x1+w1]
face2 = img[y2:y2+h2,x2:x2+w2]

#Redimention des img

face2 = cv.resize(face2, (w1,h1))
face1 = cv.resize(face1, (w2,h2))

#Remplacer face2 par face1

img[y2:y2+h2,x2:x2+w2] = face1

#Remplacer face1 par face2

img[y1:y1+h1,x1:x1+w1] = face2


#afficher l'image avec les images echangee
cv.imshow('face swap', img)
cv.waitKey(0)
cv.destroyAllWindows()

