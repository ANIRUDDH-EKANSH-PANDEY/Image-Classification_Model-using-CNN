
# Image Classification Web App with Dashboard

This project is a Flask-based web application that uses a pre-trained TensorFlow model for classifying images from the CIFAR-10 dataset. Additionally, it includes a dashboard for visualizing classification metrics and managing image uploads.

## Features

- **Image Classification**: Upload an image and get the predicted class along with the confidence level.
- **Interactive Dashboard**: View metrics and results interactively through `dashboard.py`.
- **Responsive Design**: The web app has been styled with CSS for a modern and user-friendly interface.
- **Pre-Trained Model**: Utilizes a TensorFlow model trained on the CIFAR-10 dataset.

## Folder Structure

```plaintext
.
├── dashboard
│   ├── dashboard.py
├── static
│   ├── styles.css
│   ├── cifar10_enhanced_cnn_model_1.keras
│   ├── uploads/
├── templates
│   ├── index.html
│   ├── result.html
├── app.py
├── README.md
```

## Requirements

- Python 3.8 or later
- Flask
- TensorFlow
- NumPy

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-classification-web-app.git
   cd image-classification-web-app
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scriptsctivate     # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Place your TensorFlow model (`cifar10_enhanced_cnn_model_1.keras`) in the `static` folder.

## Running the Application

1. **Run the Web App**:
   ```bash
   python app.py
   ```

2. **Run the Dashboard**:
   ```bash
   python dashboard/dashboard.py
   ```

3. Open the browser and navigate to:
   - Web App: [http://127.0.0.1:5000](http://127.0.0.1:5000)
   - Dashboard: Depending on its Flask configuration.

## How to Use

1. **Image Classification**:
   - On the home page, upload an image in PNG/JPEG format.
   - Click on "Classify".
   - View the prediction result, including the predicted class, confidence score, and the uploaded image.
   - Use the "Classify Another Image" button to upload another image.

2. **Dashboard**:
   - Run `dashboard.py` to visualize additional metrics and analysis.

## Customization

- **Styles**: Modify `static/styles.css` to change the appearance.
- **Model**: Replace `cifar10_enhanced_cnn_model_1.keras` with your own trained model.
- **Background Color**: Change the `background-color` in the `body` selector inside `styles.css`.

## Screenshots

### Home Page
Upload an image for classification:
![Home Page Screenshot](./static/screenshots/home_page.png)

### Prediction Result
View the prediction details:
![Result Page Screenshot](./static/screenshots/result_page.png)

### Dashboard
Interactive metrics visualization:
![Dashboard Screenshot](./static/screenshots/dashboard.png)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
