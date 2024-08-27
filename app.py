from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('final_model.keras')

# Define the labels for medicinal and toxic plants
medicinal_labels = [
    "Aloevera", "Amruthaballi", "Betal", "Doddpathre", "Eucalyptus","Neem","Tulsi","Turmeric"
]

toxic_labels = [
    "Castor oil plant", "Dieffenbachia", "Foxglove", "Lilies", "Lily of the valley","Oleander","Rhubarb","Wisteria"
]

#dictionary to display the details

result_dict={
    'Aloevera':"Aloe vera soothes burns, hydrates the skin without greasiness, and aids in heartburn relief by reducing inflammation. It can act as a laxative, helping with constipation. Aloe vera gel promotes wound healing by stimulating fibroblast proliferation and collagen production.",
    'Amruthaballi':"Amruthaballi enhances the immune system, offering protection against infections. It has anti-inflammatory effects useful for conditions like arthritis and joint pain. Some studies also suggest it may help regulate blood sugar levels and improve insulin sensitivity.",
    'Betal':"Betel leaves aid digestion, freshen breath, and promote gastric health. They have antimicrobial properties that help combat oral bacteria and improve oral hygiene. Traditionally, betel leaves are used for their anti-inflammatory effects, beneficial for minor inflammations and skin conditions.",
    'Doddpathre':"Doddapatre is used traditionally for respiratory ailments like coughs, colds, and asthma, with mucolytic properties that relieve chest congestion. Its leaves have antimicrobial and antifungal effects, aiding in treating infections and skin ailments. It also reduces inflammation and swelling, benefiting conditions such as arthritis and joint pain.",
    'Eucalyptus':"Eucalyptus oil relieves respiratory issues like congestion, coughs, and sinusitis, and can aid in asthma and bronchitis. It has antiseptic and anti-inflammatory properties, making it useful for treating minor wounds, insect bites, and skin infections. Additionally, it can alleviate pain and inflammation when applied to sore muscles and joints.",
    'Neem':"Neem has powerful antimicrobial properties, treating bacterial and fungal infections due to compounds like nimbin and nimbidin. Neem oil and leaf extracts are used for skin conditions such as acne and eczema. Additionally, neem twigs serve as a natural toothbrush, promoting oral hygiene and gum health. Neem also boosts the immune system, supporting overall immune health.",
    'Tulsi':"Tulsi, an adaptogen, helps the body manage stress, anxiety, and fatigue while promoting mental balance. Its antimicrobial properties combat bacterial, viral, and fungal infections, and it enhances immune function. Additionally, tulsi may support cardiovascular health by reducing cholesterol and blood pressure, and it is believed to have cardioprotective properties.",
    'Turmeric':"Turmeric aids skincare with its antibacterial and anti-inflammatory properties, helping to treat acne, reduce scars, and enhance skin complexion. Its antioxidants neutralize free radicals, protecting against chronic diseases and aging. Traditionally, turmeric is used topically for wound healing and as an antiseptic, promoting faster healing and reducing infection risk.",
    'Castor oil plant':"The seeds of the castor oil plant (Ricinus communis) are highly toxic due to ricin, causing severe abdominal pain, vomiting, diarrhea, and potentially death. Immediate medical attention is required; treatment involves supportive care and activated charcoal, as there is no specific antidote. Avoid handling or consuming plant parts, especially the seeds, and seek professional help in case of poisoning.",
    'Dieffenbachia':"The entire plant is toxic, especially the leaves and stems, containing calcium oxalate crystals. Ingestion causes intense oral irritation, swelling, drooling, and difficulty swallowing. Immediate treatment includes rinsing the mouth, drinking milk, and seeking medical attention.",
    'Foxglove':" The entire plant is highly toxic, particularly the leaves, flowers, and seeds, containing cardiac glycosides. Ingestion can lead to nausea, vomiting, abdominal pain, irregular heartbeats, and potentially death. Immediate treatment involves activated charcoal, supportive care, and seeking urgent medical attention.",
    'Lilies':"Certain types (especially true lilies like Easter, tiger, and day lilies) are highly toxic, particularly to cats, with all parts being poisonous. Ingestion causes vomiting, lethargy, and kidney failure in cats. Immediate veterinary care is crucial; treatment includes inducing vomiting, activated charcoal, and aggressive IV fluid therapy.",
    'Lily of the valley':"The entire plant is highly toxic, containing cardiac glycosides. Ingestion can cause nausea, vomiting, diarrhea, irregular heartbeats, and potentially fatal heart issues. Immediate treatment involves activated charcoal, supportive care, and urgent medical attention.",
    'Oleander':"The entire plant is highly toxic, especially the leaves and flowers, containing cardiac glycosides. Ingestion can cause nausea, vomiting, abdominal pain, irregular heartbeats, and potentially fatal heart issues. Immediate treatment includes activated charcoal, supportive care, and seeking urgent medical attention.",
    'Rhubarb':"The leaves are highly toxic, containing oxalic acid and anthraquinone glycosides. Ingestion can cause difficulty breathing, seizures, kidney failure, and potentially death. Immediate treatment involves inducing vomiting, administering calcium, and seeking urgent medical attention.",
    'Wisteria':"The seeds and pods are highly toxic, containing lectins and glycosides. Ingestion can cause nausea, vomiting, diarrhea, and abdominal pain. Immediate treatment includes inducing vomiting, administering activated charcoal, and seeking medical attention."
}

# Combine labels
class_labels = medicinal_labels + toxic_labels

# Function to preprocess an image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)  # Load and resize image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create a batch-like effect
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to predict and get the category
def predict_category(img_path):
    img_array = preprocess_image(img_path)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Determine if the class is medicinal or toxic
    if predicted_class < 8:
        return f"Medicinal plant: {class_labels[predicted_class]}",result_dict[class_labels[predicted_class]]
    
    else:
        return f"Toxic plant: {class_labels[predicted_class]}",result_dict[class_labels[predicted_class]]
    
    

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            prediction = predict_category(file_path)[0]
            analysis=predict_category(file_path)[1]
            os.remove(file_path)  # Remove the file after prediction
            return render_template('result.html', prediction=prediction,analysis=analysis)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)