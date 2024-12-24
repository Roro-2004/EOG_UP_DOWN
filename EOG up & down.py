import numpy as np
from scipy.signal import butter, filtfilt, resample
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats

def butter_bandpass(lowcut, highcut, fs, order=4):
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def preprocessing(file_path, fs=176, lowcut=0.5, highcut=20):
    try:
        data = np.loadtxt(file_path)
        processed_rows = []
        
        for row in data:
            # Step 1: Remove outliers using z-score
            z_scores = stats.zscore(row)
            row = np.where(abs(z_scores) > 3, np.mean(row), row)
            
            # Step 2: Mean removal
            row = row - np.mean(row)
            
            # Step 3: Enhanced bandpass filtering
            b, a = butter_bandpass(lowcut, highcut, fs)
            filtered_data = filtfilt(b, a, row)
            
            # Step 3: Normalization (corrected for 1D array)
            normalized_data = (filtered_data - np.min(filtered_data)) / (
                np.max(filtered_data) - np.min(filtered_data)
            )

            # Step 4: Resampling (corrected for 1D array)
            resampled_data = resample(normalized_data, num=len(normalized_data) // 2)

            # Handle NaN values
            resampled_data = np.nan_to_num(resampled_data, nan=0.0)
            
            processed_rows.append(resampled_data)
            
        return np.array(processed_rows)
    
    
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error during signal processing: {e}")
        return None

def extract_wavelet_features(signal):
    try:
        all_features = []
        for row in signal:
            row_features = []
            
            # 1. Wavelet features with more wavelet families
            wavelet_families = ['db1', 'db2', 'db3', 'db4']
            for wavelet in wavelet_families:
                coeffs = pywt.wavedec(row, wavelet, level=4)
                for coeff in coeffs:
                    row_features.extend([np.mean(coeff),np.std(coeff), np.max(coeff),np.min(coeff),stats.skew(coeff), stats.kurtosis(coeff), np.median(coeff), stats.iqr(coeff)])

            all_features.append(row_features)
            
        return np.array(all_features)

    except Exception as e:
        messagebox.showerror("Feature Extraction Error", f"Error during feature extraction: {e}")
        return None

def plot_signals(data):
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 1, 2)
    one_oscillation = data[0:500]  # Adjust the slicing range as needed
    plt.plot(one_oscillation, label="One Oscillation of Flattened Processed Signal", color="orange")
    plt.title("One Oscillation of Flattened Processed Signal (After Feature Extraction)")
    plt.xlabel("Samples")
    plt.ylabel("Normalized Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()







class EOGClassifier:
    def __init__(self):
        self.knn = KNeighborsClassifier(
            n_neighbors=3,  # Number of neighbors to consider
            weights='distance',  # Weight points by distance
            metric='euclidean',  # Distance metric
            n_jobs=-1  # Use all available CPU cores
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, up_file, down_file):
        try:
            # Process up movement data
            up_data = preprocessing(up_file)
            if up_data is None:
                return False
            up_features = extract_wavelet_features(up_data)
            
            # Process down movement data
            down_data = preprocessing(down_file)
            if down_data is None:
                return False
            down_features = extract_wavelet_features(down_data)
            
            # Create labels
            up_labels = np.ones(len(up_features))
            down_labels = np.zeros(len(down_features))
            
            # Combine features and labels
            X_train = np.vstack([up_features, down_features])
            y_train = np.concatenate([up_labels, down_labels])
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
                        
            # Train the model
            self.knn.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Calculate and display training metrics
            train_accuracy = self.knn.score(X_train_scaled, y_train)
            
            messagebox.showinfo("Training Complete", 
                              f"Training accuracy: {train_accuracy*100:.2f}%\n")            
            return True

        except Exception as e:
            messagebox.showerror("Training Error", f"Error during model training: {e}")
            return False

    def predict(self, test_file):
        if not self.is_trained:
            messagebox.showerror("Error", "Model not trained yet!")
            return None
        
        try:
            # Process test data
            test_data = preprocessing(test_file)
            if test_data is None:
                return None
                
            # Extract features and scale
            test_features = extract_wavelet_features(test_data)
            test_features_scaled = self.scaler.transform(test_features)

            # Get prediction probabilities
            pred_probs = self.knn.predict_proba(test_features_scaled)
            predictions = self.knn.predict(test_features_scaled)
                        
            print(pred_probs)

            
            # Calculate confidence using mean probability
            up_prob = np.mean(pred_probs[:, 1])
            down_prob = np.mean(pred_probs[:, 0])
            
            # Determine final prediction
            final_prediction = "UP" if up_prob > down_prob else "DOWN"
            confidence = max(up_prob, down_prob) * 100
            
            return f"{final_prediction} (Confidence: {confidence:.1f}%)"

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {e}")
            return None





# Rest of the GUI code remains the same
class EOGInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EOG Movement Classifier")
        self.classifier = EOGClassifier()
        self.setup_gui()

    def setup_gui(self):
        # Training data section
        train_frame = tk.LabelFrame(self.root, text="Training Data", padx=5, pady=5)
        train_frame.pack(padx=10, pady=5, fill="x")

        # Up movement file
        tk.Label(train_frame, text="Up Movement File:").pack(anchor="w")
        self.up_entry = tk.Entry(train_frame, width=50)
        self.up_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Browse", command=lambda: self.browse_file(self.up_entry)).pack(side=tk.RIGHT)

        # Down movement file
        down_frame = tk.LabelFrame(self.root, text="Down Movement File", padx=5, pady=5)
        down_frame.pack(padx=10, pady=5, fill="x")
        self.down_entry = tk.Entry(down_frame, width=50)
        self.down_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(down_frame, text="Browse", command=lambda: self.browse_file(self.down_entry)).pack(side=tk.RIGHT)

        # Train button
        tk.Button(self.root, text="Train Model", command=self.train_model).pack(pady=5)

        # Test section
        test_frame = tk.LabelFrame(self.root, text="Test Data", padx=5, pady=5)
        test_frame.pack(padx=10, pady=5, fill="x")
        self.test_entry = tk.Entry(test_frame, width=50)
        self.test_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(test_frame, text="Browse", command=lambda: self.browse_file(self.test_entry)).pack(side=tk.RIGHT)

        # Predict button
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