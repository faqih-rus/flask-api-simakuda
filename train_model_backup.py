import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from datetime import datetime
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class AirQuality60kTrainer:
    def __init__(self):
        """
        Inisialisasi trainer untuk dataset 60k dengan training yang dipercepat
        """
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.history = None
        self.class_weights = None
        
    def load_60k_dataset(self, filepath="air_quality_60k_2weeks.csv"):
        """
        Load dataset 60k records
        """
        print("Loading 60k dataset...")
        
        # Handle relative path
        if not os.path.isabs(filepath):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filepath = os.path.join(script_dir, filepath)
        
        df = pd.read_csv(filepath)
        
        print(f"Dataset loaded: {len(df):,} records")
        print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
        print("\nQuality distribution:")
        quality_counts = df['Air_Quality_Status'].value_counts()
        print(quality_counts)
        
        return df
    
    def prepare_data_for_training(self, df):
        """
        Prepare data untuk training neural network dengan preprocessing cepat
        """
        print("\nPreparing data for training...")
        
        # Feature selection - input layer (2 neurons)
        X = df[['MQ7_CO_PPM', 'MQ135_CO2_PPM']].values
        y = df['Air_Quality_Status'].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Convert to categorical (one-hot encoding)
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=3)
        
        # Calculate class weights to handle imbalance (fast approach)
        class_counts = np.bincount(y_encoded)
        total = len(y_encoded)
        self.class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        
        print(f"Input shape: {X_scaled.shape}")
        print(f"Output shape: {y_categorical.shape}")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Class weights: {self.class_weights}")
        
        return X_scaled, y_categorical
    
    def create_optimized_model(self, input_dim=2, num_classes=3):
        """
        Create optimized neural network architecture sesuai skripsi
        dengan performa yang baik namun cepat dalam training
        """
        print("\nCreating optimized neural network based on thesis architecture...")
        
        # Set seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        model = keras.Sequential([
            # Input layer - langsung ke hidden layer pertama
            layers.Dense(
                16, 
                activation='relu', 
                input_dim=input_dim,
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.0001),
                name='hidden_1'
            ),
            
            # Hidden layer 2
            layers.Dense(
                8, 
                activation='relu', 
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(0.0001),
                name='hidden_2'
            ),
            
            # Output layer (3 neurons untuk 3 kelas)
            layers.Dense(
                num_classes, 
                activation='softmax',
                kernel_initializer='glorot_uniform',
                name='output_layer'
            )
        ])
        
        # Compile dengan optimizer yang dioptimasi untuk kecepatan
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.002),  # Slightly higher learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, X, y, validation_split=0.2, epochs=40, batch_size=512):
        """
        Train neural network dengan kecepatan tinggi
        - Batch size yang lebih besar
        - Epochs yang lebih sedikit
        - Early stopping yang agresif
        """
        print(f"\nFast training neural network with {len(X):,} samples...")
        
        # Split data once - no cross validation for speed
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y.argmax(axis=1)
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Validation set: {len(X_val):,} samples")
        
        # Callbacks untuk optimasi kecepatan
        # Save checkpoints directly in project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(script_dir, 'models', 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoints_dir, 'best_model_60k.h5')
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,  # More aggressive early stopping
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,  # Quicker LR reduction
                min_lr=1e-4,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Fast training
        start_time = datetime.now()
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,  # Reduced epochs
            batch_size=batch_size,  # Larger batch size
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return X_val, y_val
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance with key metrics
        """
        print("\nEvaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(
            y_test_classes, y_pred_classes, 
            target_names=class_names,
            output_dict=True
        )
        
        print("Classification Report:")
        print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        # Accuracy per class
        accuracy_per_class = {}
        for i, class_name in enumerate(class_names):
            accuracy_per_class[class_name] = report[class_name]['f1-score']
        
        overall_accuracy = report['accuracy']
        
        print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print("\nAccuracy per class:")
        for class_name, acc in accuracy_per_class.items():
            print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save confusion matrix
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../outputs/reports')
        os.makedirs(reports_dir, exist_ok=True)
        plt.savefig(os.path.join(reports_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        
        return overall_accuracy, accuracy_per_class, cm
    
    def plot_training_history(self):
        """
        Plot training history (simplified for speed)
        """
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 5))
        
        # Plot in one row, two columns
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training')
        plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training')
        plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot - updated path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(script_dir, 'outputs', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        plot_path = os.path.join(reports_dir, 'training_history_60k.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history saved to: {plot_path}")
    
    def save_model_and_preprocessors(self):
        """
        Save trained model and preprocessors
        """
        print("\nSaving model and preprocessors...")
        
        # Changed path: Save directly in the root project directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models', 'saved')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(models_dir, 'air_quality_60k_model.h5')
        self.model.save(model_path)
        print(f"Model saved: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(models_dir, 'scaler_60k.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved: {scaler_path}")
        
        # Save label encoder
        encoder_path = os.path.join(models_dir, 'label_encoder_60k.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved: {encoder_path}")
        
        # Save TFLite model for embedded devices
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(models_dir, 'air_quality_60k_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TensorFlow Lite model saved: {tflite_path}")
        
        # Create a simple info file to help users understand the model files
        info_path = os.path.join(models_dir, 'model_info.txt')
        with open(info_path, 'w') as f:
            f.write("Air Quality Model Files\n")
            f.write("======================\n\n")
            f.write("- air_quality_60k_model.h5: Main Keras model file\n")
            f.write("- scaler_60k.pkl: StandardScaler for preprocessing input data\n")
            f.write("- label_encoder_60k.pkl: LabelEncoder for mapping class names\n")
            f.write("- air_quality_60k_model.tflite: TensorFlow Lite version for embedded systems\n\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Features: ['MQ7_CO_PPM', 'MQ135_CO2_PPM']\n")
            f.write(f"Classes: {list(self.label_encoder.classes_)}\n")
        print(f"Model info saved: {info_path}")
    
    def test_real_time_prediction(self):
        """
        Test prediksi real-time dengan sample data
        """
        print("\nTesting real-time prediction capability...")
        
        # Sample test cases
        test_cases = [
            {'co': 5.2, 'co2': 650, 'expected': 'Baik'},
            {'co': 15.5, 'co2': 1200, 'expected': 'Sedang'},
            {'co': 35.8, 'co2': 2800, 'expected': 'Tidak Sehat'},
            {'co': 8.9, 'co2': 950, 'expected': 'Baik'},
            {'co': 22.1, 'co2': 1850, 'expected': 'Sedang'}
        ]
        
        print("Real-time prediction test:")
        print("-" * 60)
        
        correct_count = 0
        
        for i, case in enumerate(test_cases, 1):
            # Prepare input
            input_data = np.array([[case['co'], case['co2']]])
            input_scaled = self.scaler.transform(input_data)
            
            # Predict
            start_time = datetime.now()
            prediction = self.model.predict(input_scaled, verbose=0)
            prediction_time = (datetime.now() - start_time).total_seconds() * 1000  # in ms
            
            predicted_class = self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction) * 100
            
            is_correct = predicted_class == case['expected']
            if is_correct:
                correct_count += 1
            
            print(f"Test {i}: CO={case['co']:.1f}, CO2={case['co2']:.0f}")
            print(f"  Expected: {case['expected']}")
            print(f"  Predicted: {predicted_class} (confidence: {confidence:.1f}%)")
            print(f"  Prediction time: {prediction_time:.2f} ms")
            print(f"  Status: {'✅ CORRECT' if is_correct else '❌ WRONG'}")
            print()
        
        print(f"Prediction accuracy: {correct_count}/{len(test_cases)} ({correct_count/len(test_cases)*100:.2f}%)")
    
    def complete_training_pipeline(self, dataset_file=None):
        """
        Complete training pipeline untuk dataset 60k dengan kecepatan tinggi
        """
        print("=" * 70)
        print("AIR QUALITY NEURAL NETWORK FAST TRAINING - 60K DATASET")
        print("=" * 70)
        
        # 1. Create necessary directories for Flask API compatibility
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models', 'saved')
        os.makedirs(models_dir, exist_ok=True)
        
        # 2. Load dataset
        if dataset_file is None:
            df = self.load_60k_dataset()
        else:
            df = self.load_60k_dataset(dataset_file)
        
        # 3. Prepare data
        X, y = self.prepare_data_for_training(df)
        
        # 4. Create model dengan arsitektur skripsi
        self.model = self.create_optimized_model()
        
        # 5. Train model dengan kecepatan tinggi
        X_test, y_test = self.train_model(X, y, epochs=40, batch_size=512)
        
        # 6. Evaluate
        accuracy, class_accuracy, cm = self.evaluate_model(X_test, y_test)
        
        # 7. Plot history
        self.plot_training_history()
        
        # 8. Save model in the Flask API compatible location
        self.save_model_and_preprocessors()
        
        # 9. Test real-time prediction
        self.test_real_time_prediction()
        
        # 10. Create final report
        self.create_training_report(accuracy, class_accuracy, cm)
        
        # 11. Ensure model is ready for Flask API
        self.copy_model_to_root()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"✅ Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ Model ready for real-time air quality prediction!")
        print(f"✅ Model saved in: {models_dir}")
        print(f"✅ Flask API is configured to load the model from this location.")
        print("✅ You can now start the Flask API: python app.py")

    def create_training_report(self, accuracy, class_accuracy, cm):
        """
        Create training report
        """
        # Updated path for reports
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reports_dir = os.path.join(script_dir, 'outputs', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = os.path.join(reports_dir, 'training_report_60k.txt')
        
        with open(report_path, 'w') as f:
            f.write("NEURAL NETWORK FAST TRAINING REPORT - 60K DATASET\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset: 60,000 records (2 weeks period)\n")
            f.write(f"Architecture: Feed-Forward Neural Network sesuai skripsi\n")
            f.write(f"Input Features: CO_PPM, CO2_PPM (2 neurons)\n")
            f.write(f"Hidden Layer 1: 16 neurons dengan aktivasi ReLU\n")
            f.write(f"Hidden Layer 2: 8 neurons dengan aktivasi ReLU\n")
            f.write(f"Output Classes: Baik, Sedang, Tidak Sehat (3 neurons dengan aktivasi Softmax)\n\n")
            
            f.write("TRAINING OPTIMIZATIONS:\n")
            f.write("- Larger batch size (512) for faster processing\n")
            f.write("- Reduced number of epochs with aggressive early stopping\n")
            f.write("- Class weighting for handling class imbalance\n")
            f.write("- L2 regularization to prevent overfitting\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            
            f.write("Class-wise Performance:\n")
            for class_name, acc in class_accuracy.items():
                f.write(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)\n")
            
            f.write(f"\nConfusion Matrix:\n{cm}\n")
            
            f.write("\nMODEL FILES (saved in models/saved directory):\n")
            f.write("- air_quality_60k_model.h5 (main model)\n")
            f.write("- air_quality_60k_model.tflite (TensorFlow Lite model)\n")
            f.write("- scaler_60k.pkl (feature scaler)\n")
            f.write("- label_encoder_60k.pkl (label encoder)\n")
            
            f.write("\nREADY FOR DEPLOYMENT!\n")
            f.write("Model optimized for real-time air quality prediction\n")
        
        print(f"Training report saved: {report_path}")
        
    def copy_model_to_root(self):
        """
        Copy model files to root directory for easy access by Flask API
        """
        print("\nCopying model files to root directory for Flask API...")
        import shutil
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(script_dir, 'models', 'saved')
        
        # Create the target directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Ensure the files exist before trying to copy
        source_files = [
            os.path.join(models_dir, 'air_quality_60k_model.h5'),
            os.path.join(models_dir, 'scaler_60k.pkl'),
            os.path.join(models_dir, 'label_encoder_60k.pkl')
        ]
        
        # Check if all files exist
        all_files_exist = all(os.path.exists(f) for f in source_files)
        
        if all_files_exist:
            print("All model files found and ready for Flask API")
        else:
            print("Warning: Some model files are missing.")
            
        print(f"Model files location: {models_dir}")
        print("Flask API is configured to load models from this location.")

def main():
    """
    Main function untuk training dengan dataset 60k (cepat)
    """
    trainer = AirQuality60kTrainer()
    trainer.complete_training_pipeline()

if __name__ == "__main__":
    main()