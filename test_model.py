import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from visualize_results import create_plots

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# 1. Load Data
data_path = os.path.join('Dataset', 'archive (1)', 'creditcard.csv')
df = pd.read_csv(data_path)
features = df.drop(['Class'], axis=1).columns.tolist()

df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))

# 2. Load Models
encoder = keras.models.load_model('models/encoder_model.keras', custom_objects={'Sampling': Sampling})
decoder = keras.models.load_model('models/decoder_model.keras')

# 3. Predict
test_data = df.drop(['Class'], axis=1).values.astype('float32')
_, _, z = encoder.predict(test_data, batch_size=512)
reconstructions = decoder.predict(z, batch_size=512)

# 4. Results
mse = np.mean(np.square(test_data - reconstructions), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse, 'true_class': df['Class'].values})
my_threshold = 3.0
error_df['prediction'] = (error_df['reconstruction_error'] > my_threshold).astype(int)

# 5. Save Final Text Report
with open('plots/final_report.txt', 'w') as f:
    f.write("CREDIT CARD FRAUD DETECTION REPORT\n")
    f.write(f"Used Threshold: {my_threshold}\n\n")
    f.write("Confusion Matrix:\n" + str(confusion_matrix(error_df['true_class'], error_df['prediction'])) + "\n\n")
    f.write("Metrics:\n" + classification_report(error_df['true_class'], error_df['prediction']))

# --- في ملف test_model.py الجزء رقم 6 ---

# تجميع الـ Encoder والـ Decoder في موديل واحد متوافق مع SHAP
full_model = keras.Sequential([
    encoder,
    keras.layers.Lambda(lambda x: x[2]), # نأخذ فقط الـ z من مخرجات الـ Encoder
    decoder
])

# بناء الموديل بإدخال عينة وهمية عشان SHAP يشوف الـ inputs
full_model.build(input_shape=(None, len(features))) 

# استدعاء الرسم والـ SHAP
create_plots(error_df, my_threshold, test_data, full_model, features)

print("✅ Everything is done! Check the 'plots' folder.")