# 🍇 Grape Leaf Disease Classifier

Ein Deep Learning-Projekt zur **Klassifikation von Weinblattkrankheiten** mit einem selbst entwickelten Convolutional Neural Network (CNN).
Die App erlaubt es, ein Foto eines Weinblattes hochzuladen und automatisch einer der vier Klassen zuzuordnen:

* **Black Rot**
* **ESCA**
* **Healthy**
* **Leaf Blight**

---

## 📁 Projektstruktur

```
grape_leaf_cnn/
├── app.py                 # Streamlit App zur Vorhersage
├── cnn_model.keras        # Trainiertes CNN-Modell
├── requirements.txt       # Abhängigkeiten für Streamlit/Hugging Face
├── README.md              # Dieses Dokument
└── example_leaf.jpg       # Optionales Testbild
```

---

## 🧠 Modell

* Architektur: 5x Conv2D + 3x MaxPooling + Dropout
* Input-Größe: 180x180x3
* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy
* Accuracy: \~**99 %**
* Framework: TensorFlow / Keras

Trainingsdaten: [Kaggle Dataset](https://www.kaggle.com/datasets/rm1000/augmented-grape-disease-detection-dataset)

---

## 💻 Streamlit App

> Ermöglicht das Hochladen eines Bildes und zeigt:
>
> * Vorhergesagte Klasse
> * Konfidenz (in %)

### 🔧 Lokaler Start:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Beispiel (Screenshot)

<img width="622" height="823" alt="image" src="https://github.com/user-attachments/assets/f65b2ff1-62fc-4024-9789-9aa234f8d5da" />


---

## ✨ Demo Features

* 📷 Bild-Upload (JPG/PNG)
* 📊 Sofortige Vorhersage mit Konfidenz
* ✅ Anzeige der Klassennamen
* 📀 Modell im `.keras`-Format gespeichert

---

## 🧠 Evaluation

| Klasse      | Precision | Recall | F1-Score |
| ----------- | --------- | ------ | -------- |
| Black Rot   | 0.99      | 0.98   | 0.99     |
| ESCA        | 0.99      | 0.99   | 0.99     |
| Healthy     | 1.00      | 0.99   | 1.00     |
| Leaf Blight | 0.99      | 1.00   | 0.99     |

**Gesamt-Accuracy**: `0.99`
Basierend auf `Confusion Matrix` mit 2400 Validierungsbildern.

---

## 📟 requirements.txt

```txt
streamlit
tensorflow
pillow
numpy
scikit-learn
matplotlib
seaborn
```

---

## 📌 Fußnote

**Made with ❤️ by Emr7y**
Modell: CNN
Datenquelle: [Kaggle Dataset](https://www.kaggle.com/datasets/rm1000/augmented-grape-disease-detection-dataset)

---

## 📬 Kontakt

> Bei Fragen oder Interesse: gerne per GitHub kontaktieren!
