import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import plotly.graph_objects as go
from datetime import datetime
import io
import base64 

# --- إعداد الصفحة ---
st.set_page_config(page_title="نظام التنبؤ وتوجيه المسار الأكاديمي", layout="wide", page_icon="🎓")

# --- CSS: تنسيقات وتحسينات ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa !important;
        padding: 15px;
        border-radius: 5px;
        border-right: 5px solid #2e86de;
        margin-bottom: 10px;
        color: #000000 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: right;
        direction: rtl;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. دوال قاعدة البيانات ---
def init_db():
    conn = sqlite3.connect('university_db_restored.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            student_id TEXT,
            department TEXT,
            prediction REAL,
            roadmap TEXT,
            date TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_report(name, sid, dept, pred, roadmap):
    conn = sqlite3.connect('university_db_restored.db')
    c = conn.cursor()
    c.execute("INSERT INTO reports (student_name, student_id, department, prediction, roadmap, date) VALUES (?, ?, ?, ?, ?, ?)",
              (name, sid, dept, pred, roadmap, datetime.now().strftime("%Y-%m-%d %H:%M")))
    conn.commit()
    conn.close()

init_db()

# --- 2. تحميل الموديل ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('iraqi_model.pkl')
    except:
        return None

model = load_model()

# --- 3. محرك المحاكاة ---
def simulate_improvement(current_data, model, current_score):
    scenarios = []
    # استخراج القيم بأمان
    val_attend = current_data['Attendance_Rate'].values[0] if isinstance(current_data, pd.DataFrame) else current_data['Attendance_Rate']
    val_partic = current_data['Participation_Score'].values[0] if isinstance(current_data, pd.DataFrame) else current_data['Participation_Score']

    # 1. زيادة الدراسة
    d1 = current_data.copy()
    d1['Study_Hours_Per_Week'] += 5 
    pred1 = model.predict(d1)[0]
    if pred1 > current_score:
        scenarios.append(f"تشير البيانات إلى أن تكثيف الساعات الدراسية بمعدل (5) ساعات أسبوعياً قد يرفع المعدل المتوقع إلى <span class='num-ltr'>{pred1:.1f}%</span>")
        
    # 2. تحسين الحضور
    d2 = current_data.copy()
    if val_attend < 95:
        d2['Attendance_Rate'] = 98 
        pred2 = model.predict(d2)[0]
        if pred2 > current_score:
            scenarios.append(f"الالتزام التام بحضور المحاضرات النظرية والعملية من شأنه تحسين النتيجة لتصل إلى <span class='num-ltr'>{pred2:.1f}%</span>")
            
    # 3. المشاركة
    d3 = current_data.copy()
    if val_partic < 9:
        d3['Participation_Score'] = 10
        pred3 = model.predict(d3)[0]
        if pred3 > current_score:
            scenarios.append(f"تحسين مستوى التفاعل والمشاركة الصفية إلى الحد الأقصى قد يساهم في وصول النتيجة إلى <span class='num-ltr'>{pred3:.1f}%</span>")

    return scenarios

# --- 4. دالة توليد ملف الطباعة (الحل السحري) ---
def create_downloadable_report(s_name, s_id, s_dept, current_pred, roadmap_steps, val_attend, val_study):
    status_text = "مرحلة الخطر (Critical)" if current_pred < 50 else "مرحلة الاستقرار (Stable)" if current_pred < 80 else "مرحلة التميز (Excellent)"
    rec_list_html = "".join([f"<li>{step}</li>" for step in roadmap_steps])
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>تقرير الطالب {s_name}</title>
        <style>
            body {{ font-family: 'Times New Roman', serif; padding: 40px; margin: 0; }}
            .report-container {{
                border: 2px solid #000; padding: 40px; max-width: 210mm; margin: auto; background-color: white;
            }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .header img {{ width: 80px; margin-bottom: 10px; }}
            h2, h3, h4 {{ margin: 5px 0; color: black; }}
            hr {{ border-top: 2px solid black; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            td {{ padding: 10px; font-size: 16px; vertical-align: top; }}
            .section {{ margin-bottom: 20px; border: 1px solid #000; padding: 15px; }}
            .num-ltr {{ direction: ltr; unicode-bidi: embed; display: inline-block; font-weight: bold; }}
            ul {{ padding-right: 20px; }}
            li {{ margin-bottom: 5px; font-weight: bold; }}
            @media print {{ .no-print {{ display: none; }} }}
        </style>
        <script>window.onload = function() {{ window.print(); }}</script>
    </head>
    <body>
        <div class="report-container">
            <div class="header">
                <img src="https://cdn-icons-png.flaticon.com/512/2231/2231649.png" alt="Logo">
                <h2>وزارة التعليم العالي والبحث العلمي</h2>
                <h3>الجامعة التكنولوجية - قسم {s_dept}</h3>
                <h4>تقرير تقييم الأداء الأكاديمي للطلاب</h4>
            </div>
            <hr>
            <table>
                <tr><td style="text-align: right;"><strong>اسم الطالب:</strong> {s_name}</td><td style="text-align: left;"><strong>الرقم الجامعي:</strong> <span class="num-ltr">{s_id}</span></td></tr>
                <tr><td style="text-align: right;"><strong>تاريخ الإصدار:</strong> <span class="num-ltr">{date_str}</span></td><td style="text-align: left;"><strong>المعدل المتوقع:</strong> <span class="num-ltr">{current_pred:.1f}%</span></td></tr>
            </table>
            <div class="section">
                <h4>أولاً: ملخص التحليل الفني</h4>
                <p>استناداً إلى معطيات الذكاء الاصطناعي، يُصنف الأداء الحالي للطالب ضمن <strong>{status_text}</strong>. 
                أظهر التحليل أن العوامل الأكثر تأثيراً هي نسبة الحضور (<span class="num-ltr">{val_attend}%</span>) 
                وساعات الدراسة (<span class="num-ltr">{val_study}</span> ساعة/أسبوع).</p>
            </div>
            <div class="section">
                <h4>ثانياً: التوصيات وخارطة الطريق</h4>
                <ul>{rec_list_html}</ul>
            </div>
            <br><br><br>
            <table style="text-align: center; margin-top: 50px;">
                <tr><td>____________________<br>توقيع المرشد الأكاديمي</td><td>____________________<br>ختم رئاسة القسم</td></tr>
            </table>
        </div>
    </body>
    </html>
    """
    return html_content

# --- دالة العرض ---
def show_student_details_view(s_name, s_id, s_dept, current_pred, roadmap_steps, val_attend, val_study, val_prev_avg, val_partic):
    st.divider()
    st.markdown(f"### 👤 الملف الأكاديمي: {s_name}")
    
    col_gauge, col_stats = st.columns([1.5, 2])
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_pred,
            title = {'text': "المعدل المتوقع", 'font': {'size': 20}},
            delta = {'reference': 50, 'increasing': {'color': "green"}},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#1f77b4"}, 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
        st.plotly_chart(fig, use_container_width=True)
    
    with col_stats:
            st.subheader("📊 المؤشرات")
            st.write(f"الحضور: {val_attend}%")
            st.progress(int(val_attend))
            st.write(f"المعدل السابق: {val_prev_avg}%")
            st.progress(int(val_prev_avg))
            if current_pred < 50: st.error("الحالة: حرجة 🔴")
            elif current_pred < 75: st.warning("الحالة: متوسطة 🟠")
            else: st.success("الحالة: ممتازة 🟢")

    st.markdown("---")
    st.subheader("🖨️ طباعة التقرير")
    report_html = create_downloadable_report(s_name, s_id, s_dept, current_pred, roadmap_steps, val_attend, val_study)
    st.download_button(
        label="📥 تحميل وطباعة التقرير الرسمي (Click to Print)",
        data=report_html,
        file_name=f"Report_{s_id}.html",
        mime="text/html",
        type="primary"
    )

# --- 5. القائمة الجانبية والواجهة ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3068/3068327.png", width=80)
    st.header("⚙️ لوحة التحكم")
    mode = st.radio("النمط:", ["إدخال يدوي", "استيراد ملف"])
    
    if mode == "إدخال يدوي":
        s_name = st.text_input("الاسم")
        s_id = st.text_input("الرقم")
        s_dept = st.selectbox("القسم", ["هندسة الحاسوب", "علوم الحاسوب", "IT", "AI"])
        st.divider()
        val_prev_avg = st.slider("المعدل السابق", 50, 100, 70)
        val_attend = st.slider("الحضور %", 0, 100, 80)
        val_study = st.number_input("ساعات الدراسة", 0, 60, 10)
        val_partic = st.slider("التقييم (1-10)", 1, 10, 5)
        val_fail = st.selectbox("الرسوب", [0, 1, 2, 3, 4])
        btn_analyze = st.button("🚀 تحليل", type="primary")
    else:
        uploaded_file = st.file_uploader("ملف Excel/CSV", type=['xlsx', 'csv'])
        sample_df = pd.DataFrame(columns=['Student_Name', 'Student_ID', 'Department', 'Study_Hours_Per_Week', 'Attendance_Rate', 'Previous_Average', 'Failures_History', 'Participation_Score'])
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer: sample_df.to_excel(writer, index=False)
        st.download_button("📥 قالب فارغ", buffer.getvalue(), "template.xlsx")

st.title("🎓 نظام التنبؤ الأكاديمي")

if not model: st.error("⚠️ شغل model_v2.py أولاً"); st.stop()

if mode == "إدخال يدوي" and btn_analyze:
    if s_name:
        fail = val_fail if isinstance(val_fail, int) else 4
        row = pd.DataFrame({'Study_Hours_Per_Week': [val_study], 'Attendance_Rate': [val_attend], 'Previous_Average': [val_prev_avg], 'Failures_History': [fail], 'Participation_Score': [val_partic]})
        pred = model.predict(row)[0]
        steps = simulate_improvement(row, model, pred)
        show_student_details_view(s_name, s_id, s_dept, pred, steps, val_attend, val_study, val_prev_avg, val_partic)

elif mode == "استيراد ملف" and uploaded_file:
    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
    else: df = pd.read_excel(uploaded_file)
    
    if st.button("⚡ تحليل الكل"):
        res = []
        df['Prediction'] = 0.0
        for i, r in df.iterrows():
            row = pd.DataFrame({'Study_Hours_Per_Week': [r['Study_Hours_Per_Week']], 'Attendance_Rate': [r['Attendance_Rate']], 'Previous_Average': [r['Previous_Average']], 'Failures_History': [r['Failures_History']], 'Participation_Score': [r['Participation_Score']]})
            p = model.predict(row)[0]
            df.at[i, 'Prediction'] = p
            res.append({'الاسم': r['Student_Name'], 'القسم': r['Department'], 'التوقع': round(p,1)})
        
        st.session_state['res'] = pd.DataFrame(res)
        st.session_state['full'] = df
    
    if 'res' in st.session_state:
        st.subheader("النتائج (اضغط لعرض التقرير)")
        buffer_res = io.BytesIO()
        with pd.ExcelWriter(buffer_res, engine='openpyxl') as writer: st.session_state['res'].to_excel(writer, index=False)
        st.download_button("📥 تحميل النتائج (Excel)", buffer_res.getvalue(), "results.xlsx")
        
        event = st.dataframe(st.session_state['res'], on_select="rerun", selection_mode="single-row", use_container_width=True)
        if len(event.selection.rows) > 0:
            idx = event.selection.rows[0]
            full_r = st.session_state['full'].iloc[idx]
            sim_row = pd.DataFrame({'Study_Hours_Per_Week': [full_r['Study_Hours_Per_Week']], 'Attendance_Rate': [full_r['Attendance_Rate']], 'Previous_Average': [full_r['Previous_Average']], 'Failures_History': [full_r['Failures_History']], 'Participation_Score': [full_r['Participation_Score']]})
            steps = simulate_improvement(sim_row, model, full_r['Prediction'])
            show_student_details_view(full_r['Student_Name'], str(full_r['Student_ID']), full_r['Department'], full_r['Prediction'], steps, full_r['Attendance_Rate'], full_r['Study_Hours_Per_Week'], full_r['Previous_Average'], full_r['Participation_Score'])