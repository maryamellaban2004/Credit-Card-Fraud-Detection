import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Data Preparation (تجهيز البيانات)
data_path = os.path.join('Dataset', 'archive (1)', 'creditcard.csv')

try:
    df = pd.read_csv(data_path)
    # عمل Scaling للميزات الزمنية والمبلغ لتوحيد النطاق
    df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))
    
    # تدريب الـ VAE يتم على البيانات الطبيعية فقط (Unsupervised Learning)
    normal_df = df[df['Class'] == 0].drop(['Class'], axis=1)
    X_train, X_val = train_test_split(normal_df, test_size=0.2, random_state=42)
    
    X_train = X_train.values.astype('float32')
    X_val = X_val.values.astype('float32')
    print(f"✅ Data ready: Train {X_train.shape[0]}, Val {X_val.shape[0]}")

    # 2. VAE Architecture (بناء معمارية الموديل)
    input_dim = X_train.shape[1]
    latent_dim = 2

    # طبقة الـ Sampling لتحويل الأوزان إلى توزيع احتمالي
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # --- Encoder (المشفر) ---
    enc_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu")(enc_inputs)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(enc_inputs, [z_mean, z_log_var, z], name="encoder")

    # --- Decoder (فك التشفير) ---
    dec_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16, activation="relu")(dec_inputs)
    x = layers.Dense(32, activation="relu")(x)
    dec_outputs = layers.Dense(input_dim)(x)
    decoder = Model(dec_inputs, dec_outputs, name="decoder")

    # --- VAE Model Class (تجميع الموديل وحساب الـ Loss المخصص) ---
    class VAE(Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

        @property
        def metrics(self): return [self.total_loss_tracker, self.val_loss_tracker]

        def call(self, inputs):
            _, _, z = self.encoder(inputs)
            return self.decoder(z)

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                # حساب خطأ إعادة البناء (Reconstruction Loss)
                recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=1))
                # حساب الـ KL Divergence لضمان انتظام التوزيع
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = recon_loss + kl_loss
            
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            return {"loss": self.total_loss_tracker.result()}

        def test_step(self, data):
            # دالة مخصصة لحساب الـ Val_loss أثناء التدريب
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=1))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = recon_loss + kl_loss
            self.val_loss_tracker.update_state(total_loss)
            return {"loss": self.val_loss_tracker.result()}

    # 3. Training (عملية التدريب)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    print("\n--- Starting Training (30 Epochs) ---")
    history = vae.fit(X_train, validation_data=(X_val,), epochs=30, batch_size=128)

    # 4. Save (حفظ النتائج والموديلات)
    if not os.path.exists('models'): os.makedirs('models')
    
    # حفظ الموديلات بصيغة keras
    encoder.save('models/encoder_model.keras')
    decoder.save('models/decoder_model.keras')
    
    # حفظ سجل الـ Loss لمقارنة Training vs Validation
    pd.DataFrame(history.history).to_csv('models/training_history.csv', index=False)
    
    # حفظ هيكل الموديل (Summary) مع استخدام UTF-8 لتجنب أخطاء التشفير في ويندوز
    with open('models/model_summary.txt', 'w', encoding='utf-8') as f:
        f.write("--- Encoder Summary ---\n")
        encoder.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n" + "="*50 + "\n")
        f.write("--- Decoder Summary ---\n")
        decoder.summary(print_fn=lambda x: f.write(x + '\n'))
        
    print("\n✅ Training Complete, Models and Summary Saved successfully!")

except Exception as e: 
    print(f"❌ Error: {e}")