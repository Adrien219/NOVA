import cv2
import pytesseract
import numpy as np

def extract_text(frame):
    """
    Fonction optimisée pour extraire le texte d'une image OpenCV
    """
    try:
        # 1. Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Amélioration du contraste (Seuillage adaptatif)
        # Cela rend le texte noir sur fond blanc pur
        processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 3. OCR avec Tesseract
        # --psm 3 : Mode automatique (détecte les blocs de texte)
        config = r'--oem 3 --psm 3'
        text = pytesseract.image_to_string(processed_img, lang='fra', config=config)

        return text.strip()
    except Exception as e:
        print(f"Erreur OCR : {e}")
        return ""

# Petit test rapide si tu lances le fichier seul
if __name__ == "__main__":
    print("Test OCR activé...")
    # Charger une image de test si tu en as une, ou tester via la cam