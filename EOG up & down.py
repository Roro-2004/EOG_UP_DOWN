import numpy as np
from scipy.signal import butter, filtfilt, resample
import tkinter as tk
from tkinter import filedialog, messagebox
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats



#-------------------------------------------------------PREPROCESSING-------------------------------------------------------#
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def preprocessing(file_path, fs=176, lowcut=0.5, highcut=20):
    data = np.loadtxt(file_path)
    processed_rows = []
    
    for row in data:
        # remove outliers
        z_scores = stats.zscore(row)
        row = np.where(abs(z_scores) > 3, np.mean(row), row)
        
        # mean removal
        row = row - np.mean(row)
        
        # filter
        b, a = butter_bandpass(lowcut, highcut, fs)
        filtered_data = filtfilt(b, a, row)
        
        # normalization
        normalized_data = (filtered_data - np.min(filtered_data)) / (np.max(filtered_data) - np.min(filtered_data))

        # resampling
        resampled_data = resample(normalized_data, num=len(normalized_data) // 2)

        # removing nulls
        resampled_data = np.nan_to_num(resampled_data, nan=0.0)
        
        processed_rows.append(resampled_data)
                
    return np.array(processed_rows)
#---------------------------------------------------------------------------------------------------------------------------#



#-----------------------------------------------------FEATURE EXTRACTION----------------------------------------------------#
def extract_wavelet_features(signal):
    all_features = []
    for row in signal:
        row_features = []
        wavelet_families = ['db1', 'db2', 'db3', 'db4']
        for wavelet in wavelet_families:
            coefficients = pywt.wavedec(row, wavelet, level=4)
            for co in coefficients:
                row_features.extend([np.mean(co),np.std(co), np.max(co),np.min(co),stats.skew(co), stats.kurtosis(co), np.median(co), stats.iqr(co)])

        all_features.append(row_features)       
    return np.array(all_features)
#---------------------------------------------------------------------------------------------------------------------------#



#-----------------------------------------------------KNN CLASSIFICATION----------------------------------------------------#
class EOGClassifier:
    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, up_file, down_file):
        # up movement data
        up_data = preprocessing(up_file)
        up_features = extract_wavelet_features(up_data)
        print("UP train preprocessing and feature extraction completed")

        # down movement data
        down_data = preprocessing(down_file)
        down_features = extract_wavelet_features(down_data)
        print("DOWN train preprocessing and feature extraction completed")

        # create labels
        up_labels = np.ones(len(up_features))
        down_labels = np.zeros(len(down_features))
        
        # combine up and down data
        X_train = np.vstack([up_features, down_features])
        y_train = np.concatenate([up_labels, down_labels])
        
        # scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
                    
        # train the model
        self.knn.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate accuracy
        train_accuracy = self.knn.score(X_train_scaled, y_train)
        
        print(f"Training accuracy: {train_accuracy*100}%" )
        
        return True              
              
    def predict(self, test_file):
        if not self.is_trained:
            messagebox.showerror("Error", "Model not trained yet!")
            return None
        
        # process test data
        test_data = preprocessing(test_file)
        test_features = extract_wavelet_features(test_data)
        test_features_scaled = self.scaler.transform(test_features)

        # Get prediction probabilities
        pred_probs = self.knn.predict_proba(test_features_scaled)
        predictions = self.knn.predict(test_features_scaled)
                    
        print(predictions)
        
        # calculate confidence using mean probability
        up_prob = np.mean(pred_probs[:, 1])
        down_prob = np.mean(pred_probs[:, 0])
        confidence = max(up_prob, down_prob) * 100
        print(f"(Confidence: {confidence:.1f}%)")

        # Determine final prediction
        final_prediction = "UP" if up_prob > down_prob else "DOWN"
        print(f"Final prediction is : {final_prediction}")
        
        return f"{final_prediction}"
#---------------------------------------------------------------------------------------------------------------------------#



#------------------------------------------------------------GUI------------------------------------------------------------#
class EOGInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EOG UP & DOWN Classifier")
        self.classifier = EOGClassifier()
        self.setup_gui()

    def setup_gui(self):
        train_frame = tk.LabelFrame(self.root, text="Training Data", padx=5, pady=5)
        train_frame.pack(padx=10, pady=5, fill="x")

        tk.Label(train_frame, text="Up Movement File:").pack(anchor="w")
        self.up_entry = tk.Entry(train_frame, width=50)
        self.up_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Browse", command=lambda: self.browse_file(self.up_entry)).pack(side=tk.RIGHT)

        down_frame = tk.LabelFrame(self.root, text="Down Movement File", padx=5, pady=5)
        down_frame.pack(padx=10, pady=5, fill="x")
        self.down_entry = tk.Entry(down_frame, width=50)
        self.down_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(down_frame, text="Browse", command=lambda: self.browse_file(self.down_entry)).pack(side=tk.RIGHT)

        tk.Button(self.root, text="Train Model", command=self.train_model).pack(pady=5)

        test_frame = tk.LabelFrame(self.root, text="Test Data", padx=5, pady=5)
        test_frame.pack(padx=10, pady=5, fill="x")
        self.test_entry = tk.Entry(test_frame, width=50)
        self.test_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(test_frame, text="Browse", command=lambda: self.browse_file(self.test_entry)).pack(side=tk.RIGHT)

        tk.Button(self.root, text="Predict", command=self.predict).pack(pady=5)

    def browse_file(self, entry):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if filename:
            entry.delete(0, tk.END)
            entry.insert(0, filename)

    def train_model(self):
        up_file = self.up_entry.get()
        down_file = self.down_entry.get()

        if not up_file or not down_file:
            messagebox.showerror("Error", "Please select both training files!")
            return

        if self.classifier.train(up_file, down_file):
            messagebox.showinfo("Success", "Model trained successfully!")

    def predict(self):
        test_file = self.test_entry.get()
        if not test_file:
            messagebox.showerror("Error", "Please select a test file!")
            return

        result = self.classifier.predict(test_file)
        if result:
            messagebox.showinfo("Prediction", f"Predicted Movement: {result}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = EOGInterface()
    app.run()
#---------------------------------------------------------------------------------------------------------------------------#
