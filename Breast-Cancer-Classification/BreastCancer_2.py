import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. إعدادات الصفحة واللغة ---
st.set_page_config(page_title="Cancer Diagnosis", layout="wide")

# اختيار اللغة من الجنب
with st.sidebar:
    st.markdown("### 🌐 Settings / الإعدادات")
    lang = st.radio("Language / اللغة", ("English", "العربية"))
    st.divider()
    st.image("https://cdn-icons-png.flaticon.com/512/2869/2869818.png", width=70)

# قاموس النصوص للترجمة الفورية
texts = {
    "English": {
        "title": "Breast Cancer Diagnostic Tool",
        "sub": "Adjust the sliders to match laboratory results, then click Predict.",
        "btn": "Predict Result",
        "res_title": "Diagnostic Outcome:",
        "mal": "Malignant (Cancer Detected)",
        "ben": "Benign (No Cancer Detected)",
        "dir": "ltr"
    },
    "العربية": {
        "title": "أداة تشخيص سرطان الثدي",
        "sub": "قم بتحريك المؤشرات لتطابق النتائج المخبرية، ثم اضغط على تشخيص.",
        "btn": "بدء التشخيص",
        "res_title": "نتيجة التحليل:",
        "mal": "خبيث (تم اكتشاف إصابة)",
        "ben": "حميد (لا يوجد إصابة)",
        "dir": "rtl"
    }
}

T = texts[lang]

# --- 2. التحكم في اتجاه الصفحة (RTL / LTR) ---
if lang == "العربية":
    st.markdown("""
        <style>
        .main { direction: rtl; text-align: right; }
        div.stButton > button { width: 100%; background-color: #d33; color: white; }
        p, h1, h3, label { text-align: right !important; font-family: 'Cairo', sans-serif; }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main { direction: ltr; text-align: left; }
        div.stButton > button { width: 100%; background-color: #007bff; color: white; }
        </style>
        """, unsafe_allow_html=True)


# --- 3. تحميل البيانات والموديل ---
@st.cache_resource
def load_assets():
    data = pd.read_csv('data_cancer.csv')
    model = joblib.load('Breast_Cancer_model.pkl')
    scaler = joblib.load('Scaler.pkl')
    X = data.drop('diagnosis', axis=1)
    return X, model, scaler


X, model, scaler = load_assets()

# --- 4. واجهة المستخدم الرئيسية ---
st.title(T["title"])
st.write(T["sub"])
st.divider()

# عرض السلايدرز في أعمدة (شكل احترافي ومنظم)
col1, col2 = st.columns(2)
input_values = []

for i, col_name in enumerate(X.columns):
    # نوزعهم على العمودين
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        # قراءة القيم من الداتا عشان السلايدر يبدأ بمقاييس صحيحة
        val = st.slider(
            label=f"**{col_name}**",
            min_value=float(X[col_name].min()),
            max_value=float(X[col_name].max()),
            value=float(X[col_name].mean()),
            format="%.2f"
        )
        input_values.append(val)

st.divider()

# --- 5. زر التوقع والنتيجة ---
if st.button(T["btn"]):
    features = np.array(input_values).reshape(1, -1)
    features_std = scaler.transform(features)
    prediction = model.predict(features_std)

    st.subheader(T["res_title"])

    # ملحوظة: تأكد لو الموديل مطلع 0 يبقي خبيث ولا العكس حسب تدريبك
    if prediction[0] == 0:
        st.error(f"⚠️ {T['mal']}")
    else:
        st.success(f"✅ {T['ben']}")

st.caption("AI Medical Assistant | 2025")