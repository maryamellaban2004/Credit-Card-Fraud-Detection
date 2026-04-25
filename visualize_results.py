import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import os
import shap

def create_plots(error_df, threshold, test_data_tensor, model_for_shap, feature_names):
    """
    دالة شاملة لإنشاء وحفظ كافة الرسومات البيانية والتقارير البصرية.
    """
    # التأكد من وجود مجلد plots لحفظ الصور
    if not os.path.exists('plots'): 
        os.makedirs('plots')

    # 1. رسمة منحنى التعلم (Training vs Validation Loss)
    # مهمة جداً لإثبات عدم وجود Overfitting
    if os.path.exists('models/training_history.csv'):
        h = pd.read_csv('models/training_history.csv')
        plt.figure(figsize=(10, 6))
        plt.plot(h['loss'], label='Training Loss (Normal Data)', color='blue', linewidth=2)
        if 'val_loss' in h.columns:
            plt.plot(h['val_loss'], label='Validation Loss (Testing)', color='red', linestyle='--', linewidth=2)
        plt.title('Model Learning Curve: Loss Convergence')
        plt.xlabel('Epochs')
        plt.ylabel('Reconstruction Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/1_learning_curve.png')
        plt.close()

    # 2. رسمة Precision-Recall Curve
    # المقياس الأهم في حالات البيانات غير المتوازنة (Imbalanced Data) مثل الاحتيال
    p, r, _ = precision_recall_curve(error_df['true_class'], error_df['reconstruction_error'])
    plt.figure(figsize=(8, 6))
    plt.plot(r, p, color='green', label=f'AUPRC = {auc(r, p):.2f}')
    plt.title('Precision-Recall Curve (Fraud Detection Performance)')
    plt.xlabel('Recall (Ability to find fraud)')
    plt.ylabel('Precision (Accuracy of fraud alarms)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/2_pr_curve.png')
    plt.close()

    # 3. رسمة Confusion Matrix (مصفوفة الارتباك)
    # توضح عدد حالات الـ False Negatives (الحرامية اللي هربوا) والـ False Positives
    preds = (error_df['reconstruction_error'] > threshold).astype(int)
    cm = confusion_matrix(error_df['true_class'], preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix at Threshold {threshold}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig('plots/3_confusion_matrix.png')
    plt.close()

    # 4. تحليل SHAP (تفسير القرارات)
    # شرح "لماذا" اعتبر الموديل هذه العمليات احتيالية
    print("🧠 Generating SHAP explanations (KernelExplainer)... This might take a moment.")
    
    # نختار عينة من العمليات التي تم تصنيفها كـ Fraud لشرحها
    fraud_indices = error_df[error_df['prediction'] == 1].index[:5]
    if len(fraud_indices) > 0:
        sample_data = test_data_tensor[fraud_indices]
        # خلفية من البيانات الطبيعية للمقارنة (50 عينة تكفي للسرعة)
        background = test_data_tensor[:50] 
        
        try:
            # KernelExplainer هو الأكثر استقراراً مع الموديلات المخصصة مثل VAE
            explainer = shap.KernelExplainer(model_for_shap.predict, background)
            shap_values = explainer.shap_values(sample_data)
            
            plt.figure(figsize=(10, 8))
            # استخراج القيم (تختلف الهيكلية حسب إصدار الموديل)
            vals = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            shap.summary_plot(vals, sample_data, feature_names=feature_names, show=False)
            plt.title(f"SHAP Feature Importance: Why cases were flagged")
            plt.savefig('plots/4_shap_summary.png', bbox_inches='tight')
            plt.close()
            print("✅ SHAP analysis saved successfully.")
        except Exception as e:
            print(f"⚠️ SHAP Warning: {e}. (تجاهلي هذا الخطأ إذا كانت باقي الصور سليمة)")
    else:
        print("⚠️ No Fraud cases detected to explain with SHAP.")

    print("\n📊 All professional plots have been updated in the 'plots/' folder.")