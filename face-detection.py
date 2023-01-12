#importations
import cv2 as cv

#Chargement de classificateur en cascade preentrainer 

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

#Chargement des images

img = cv.imread('votreimage.jpg')

#Conversion de l'iamge en niveau de gris pour pouvoir la passer dans le classificateur

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#Execution de la detection de visage 

#ont met en parametre le parametre gray puis notre parametre d'echalle et le parametre du nombre de voisin
#detectMultiScale(img, parametre echalle , parametre du nombre de voisin)

faces = face_cascade.detectMultiScale(gray, 1.1, 8)

#Affichage des visages

for face in faces:
    x, y, w, h = face
    
    #Dessiner le rectencle sur l'image principal 
    
    #On passe ne parametre : l'image, la detection du coin gauche , le coin en bas a droite, la couleur du rectangle, l'epaisseur de la ligne du rectangle
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#Execution detection des yeux
eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)

#Affichage des yeux
    
for (ex,ey,ew,eh) in eyes:
    
    #Dessiner le retangle autour des yeux sur l'image principale
    cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255,0,0), 1)

#on affiche notre image 
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()