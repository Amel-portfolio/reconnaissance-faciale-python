import cv2
import json
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from collections import defaultdict

#charge les visages pré-entraînés
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#limite de visages reconnus pour éviter les crash
max_faces = 10

#initialise le dictionnaire pour stocker les informations sur les visages détectés
faces_info = defaultdict(dict)

#initialise la variable pour la capture vidéo
cap = None

#fonction pour extraire les caractéristiques d'un visage
def extract_face_features(face, frame):
    #récupère la région du visage de l'image originale
    (x, y, w, h) = face
    face_region = frame[y:y+h, x:x+w]

    #convertit la région du visage en image en niveaux de gris
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    #égalisation d'histogramme pour améliorer le contraste
    gray_face = cv2.equalizeHist(gray_face)

    #détecte les yeux dans la région du visage
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #si aucun œil n'est détecté, retourne None pour indiquer que les caractéristiques ne peuvent pas être extraites
    if len(eyes) < 2:
        return None

    #détermine la couleur des yeux
    eye_colors = []
    for (ex, ey, ew, eh) in eyes:
        eye_region = face_region[ey:ey+eh, ex:ex+ew]
        avg_color_per_row = np.average(eye_region, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        eye_colors.append(avg_color)

    eye_color = np.mean(eye_colors, axis=0)

    # Détermine la couleur moyenne des cheveux (peut être amélioré)
    hair_color = [0, 0, 255]  # Rouge pour l'exemple

    #détermine la couleur moyenne de la peau (peut être amélioré)
    skin_color = [0, 255, 0]  # Vert pour l'exemple

    #calcule la luminosité moyenne de la peau
    skin_brightness = np.mean(gray_face)

    #détermine la couleur du visage en fonction de la luminosité de la peau
    if skin_brightness > 127:  #si la luminosité est élevée, c'est probablement du blanc
        face_color = "blanc"
    else:  #sinon, c'est probablement du noir
        face_color = "noir"

    #retourne les caractéristiques extraites
    return {
        'eye_color': eye_color.tolist(),
        'hair_color': hair_color,
        'skin_color': skin_color,
        'face_color': face_color,
        'width': w,
        'height': h
    }

def start_camera():
    global cap
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Erreur", "Impossible d'ouvrir la caméra.")
        return

    def update_frame():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                if len(faces_info) >= max_faces:
                    break
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face_features = extract_face_features((x, y, w, h), frame)
                if face_features is not None:
                    found_subject = False
                    for subject, features in faces_info.items():
                        if abs(features['width'] - face_features['width']) < 20 and abs(features['height'] - face_features['height']) < 20:
                            cv2.putText(frame, subject, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                            found_subject = True
                            break
                    if not found_subject:
                        new_subject = 'sujet{}'.format(len(faces_info) + 1)
                        faces_info[new_subject] = face_features
                        cv2.putText(frame, new_subject, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        update_text(new_subject, face_features)
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            label.config(image=photo)
            label.image = photo
            label.after(10, update_frame)
        else:
            messagebox.showerror("Erreur", "La caméra a rencontré un problème.")
            root.quit()

    update_frame()

def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

def update_text(subject, features):
    text.delete('1.0', tk.END)
    text.insert(tk.END, f"Sujet : {subject}\n")
    text.insert(tk.END, f"Couleur de peau : {features['skin_color']}\n")
    text.insert(tk.END, f"Couleur du visage : {features['face_color']}\n")

#fenêtre principale
root = tk.Tk()
root.title("Détection de visages")

# Création du bouton pour démarrer la caméra
start_button = tk.Button(root, text="Démarrer la caméra", command=start_camera)
start_button.pack(pady=10)

#bouton pour arrêter la caméra
stop_button = tk.Button(root, text="Arrêter la caméra", command=stop_camera)
stop_button.pack(pady=5)

#texte pour afficher la vidéo de la caméra
label = tk.Label(root)
label.pack()

#zone de texte
text = tk.Text(root, width=30, height=10)
text.pack(side=tk.LEFT, padx=10, pady=10)

#boucle principale
root.mainloop()