import numpy as np
from scipy.signal import butter, filtfilt, resample
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def preprocessing(file_path, fs, lowcut, highcut):
    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        messagebox.showerror("File Error", f"Error reading file: {e}")
        return None, None

    # Step 1: Mean removal
    mean_removed_data = data - np.mean(data, axis=1, keepdims=True)

    # Step 2: Apply bandpass filter
    b, a = butter_bandpass(lowcut, highcut, fs)
    filtered_data = filtfilt(b, a, mean_removed_data, axis=1)

    # Step 3: Normalization
    normalized_data = (filtered_data - np.min(filtered_data, axis=1, keepdims=True)) / (
        np.max(filtered_data, axis=1, keepdims=True) - np.min(filtered_data, axis=1, keepdims=True)
    )

    # Step 4: Resampling
    resampled_data = resample(normalized_data, num=int(normalized_data.shape[1] / 2), axis=1)

    # Handle NaN values
    resampled_data = np.nan_to_num(resampled_data, nan=0.0)

    return data, resampled_data


def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)


def start_processing():
    global raw_data, processed_data
    file_path = file_entry.get()
    try:
        fs = float(fs_entry.get())
        lowcut = float(lowcut_entry.get())
        highcut = float(highcut_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for fs, lowcut, and highcut.")
        return

    if not file_path:
        messagebox.showerror("File Error", "Please select a file.")
        return
    
    raw_data, processed_data = preprocessing(file_path, fs, lowcut, highcut)
    print(processed_data)


def extract_wavelet_features(signal, wavelet_name="db3", levels=4):
    coeffs = pywt.wavedec(signal, wavelet_name, level=levels)
    features = []
    for coeff in coeffs:
        mean = np.nan_to_num(np.mean(coeff), nan=0.0)
        std = np.nan_to_num(np.std(coeff), nan=0.0)
        features.append(mean)
        features.append(std)
    return np.array(features)


def process_wavelet_features():
    global raw_data, processed_data, flattened_processed_data,extracted_features

    # Flatten processed data for feature extraction
    flattened_processed_data = processed_data.flatten()

    # Extract features
    extracted_features = extract_wavelet_features(flattened_processed_data, 'db3',4)

    messagebox.showinfo("Feature Extraction Complete", f"Extracted Features:\n{extracted_features}")
    return extracted_features

def plot_signals():
    global raw_data, processed_data, flattened_processed_data

    if raw_data is None or processed_data is None:
        messagebox.showerror("Plot Error", "No processed data available. Please process the signal first.")
        return

    plt.figure(figsize=(8, 8))

    # Plot raw data
    plt.subplot(2, 1, 1)
    plt.plot(raw_data[0], label="Raw Signal")
    plt.title("Raw Signal")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.legend()

    # Plot processed or flattened data
    plt.subplot(2, 1, 2)
    if 'flattened_processed_data' in globals():
        one_oscillation = flattened_processed_data[0:500]  # Adjust the slicing range as needed
        plt.plot(one_oscillation, label="One Oscillation of Flattened Processed Signal", color="orange")
        plt.title("One Oscillation of Flattened Processed Signal (After Feature Extraction)")
    else:
        plt.plot(processed_data[0], label="Processed Signal", color="orange")
        plt.title("Processed Signal")

    plt.xlabel("Samples")
    plt.ylabel("Normalized Amplitude")
    plt.legend()

    plt.tight_layout()
    plt.show()


def classify_with_knn():
    global extracted_features

    if extracted_features is None:
        messagebox.showerror("KNN Classification", "No features available. Please extract features first.")
        return

    # Simulate train and test data
    X_train = np.tile(extracted_features, (10, 1))  # Replicating the feature for training
    y_train = np.random.randint(0, 2, size=(10,))  # Simulated labels
    X_test = np.tile(extracted_features, (5, 1))   # Replicating the feature for testing
    y_test = np.random.randint(0, 2, size=(5,))    # Simulated labels

    # Ensure there are no NaN values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Train KNN classifier
    k = 3  # Choose k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Test the classifier
    y_pred = knn.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    messagebox.showinfo("KNN Classification", f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy


root = tk.Tk()
root.title("Signal Processing GUI")

# File selection
tk.Label(root, text="Select File:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
file_entry = tk.Entry(root, width=50)
file_entry.grid(row=0, column=1, padx=5, pady=5)
browse_button = tk.Button(root, text="Browse", command=lambda: browse_file(file_entry))
browse_button.grid(row=0, column=2, padx=5, pady=5)

# Sampling frequency input
tk.Label(root, text="Sampling Frequency (fs):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
fs_entry = tk.Entry(root, width=20)
fs_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

# Lowcut frequency input
tk.Label(root, text="Lowcut Frequency (Hz):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
lowcut_entry = tk.Entry(root, width=20)
lowcut_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

# Highcut frequency input
tk.Label(root, text="Highcut Frequency (Hz):").grid(row=3, column=0, padx=5, pady=5, sticky="e")
highcut_entry = tk.Entry(root, width=20)
highcut_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")

# Start processing button
process_button = tk.Button(root, text="Start Processing", command=start_processing)
process_button.grid(row=4, column=0, columnspan=3, pady=10)

# Plot signals button
plot_button = tk.Button(root, text="Plot Signals", command=plot_signals)
plot_button.grid(row=5, column=0, columnspan=3, pady=10)

# Extract features button
Extract_Features = tk.Button(root, text="Extract Features", command=process_wavelet_features)
Extract_Features.grid(row=6, column=0, columnspan=3, pady=10)

# Classify using KNN button
knn_button = tk.Button(root, text="Classify with KNN", command=classify_with_knn)
knn_button.grid(row=7, column=0, columnspan=3, pady=10)

# Variables to hold data
raw_data = None
processed_data = None
flattened_processed_data = None
extracted_features = None

# Run the GUI loop
root.mainloop()
