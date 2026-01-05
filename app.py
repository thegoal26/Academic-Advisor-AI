import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import streamlit.components.v1 as components
import base64
import os # مكتبة للتعامل مع الملفات

# --- إعداد الصفحة ---
st.set_page_config(
    page_title="نظام الذكاء الاصطناعي الأكاديمي | AUIQ",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="collapsed"
)

# --- دوال المساعدة ---
def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

logo_base64 = get_img_as_base64("alayen.png")

# --- دالة الحفظ التلقائي (الاستبيان) ---
def save_data_collection(student_name, student_id, dept, inputs_df, prediction):
    file_name = 'collected_dataset.csv'
    
    # تجهيز الصف للحفظ
    data_to_save = inputs_df.copy()
    data_to_save.insert(0, 'Prediction', prediction)
    data_to_save.insert(0, 'Department', dept)
    data_to_save.insert(0, 'Student_ID', student_id)
    data_to_save.insert(0, 'Student_Name', student_name)
    data_to_save['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # الحفظ في ملف CSV (تراكمي)
    if not os.path.isfile(file_name):
        data_to_save.to_csv(file_name, index=False)
    else:
        data_to_save.to_csv(file_name, mode='a', header=False, index=False)

# --- CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    h1, h2, h3 { color: #0d2c56; font-weight: 700; }
    .stButton button { background-color: #0d2c56; color: white; border-radius: 8px; transition: all 0.3s; }
    .stButton button:hover { background-color: #bfa362; color: white; }
    .metric-container { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-right: 5px solid #0d2c56; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: right; margin-bottom: 15px; }
    
    @media print {
        body { visibility: hidden; background-color: white !important; }
        .report-container-wrapper { visibility: visible !important; position: absolute !important; left: 0 !important; top: 0 !important; width: 100% !important; margin: 0 !important; padding: 0 !important; z-index: 9999 !important; background-color: white !important; }
        .report-container-wrapper * { visibility: visible !important; }
        .page-break { page-break-after: always; }
        .no-print { display: none !important; }
        @page { margin: 0.5cm; size: A4 portrait; }
    }
</style>
""", unsafe_allow_html=True)

# --- إدارة الجلسة ---
if 'user_type' not in st.session_state: st.session_state['user_type'] = None

# --- شاشة تسجيل الدخول ---
def login_screen():
    col_spacer1, col_logo, col_spacer2 = st.columns([1, 1, 1])
    with col_logo:
        if logo_base64:
            st.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_base64}" width="150"></div>', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #0d2c56;'>بوابة النظام الأكاديمي الذكي</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>جامعة العين العراقية - الكلية التقنية الهندسية</p>", unsafe_allow_html=True)
        st.divider()

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        tab_student, tab_admin = st.tabs(["👤 بوابة الطالب", "🔐 بوابة الإدارة"])
        with tab_student:
            st.info("الدخول متاح للطلبة للاطلاع على مؤشرات الأداء الشخصي.")
            if st.button("تسجيل الدخول كطالب", use_container_width=True):
                st.session_state['user_type'] = 'student'; st.rerun()
        with tab_admin:
            st.warning("هذه المنطقة مخصصة للكادر التدريسي والإداري فقط.")
            user = st.text_input("اسم المستخدم المعرف"); pw = st.text_input("كلمة المرور", type="password")
            if st.button("تأكيد الدخول الآمن", type="primary", use_container_width=True):
                if user == "admin" and pw == "1234":
                    st.session_state['user_type'] = 'admin'; st.rerun()
                else: st.error("بيانات الاعتماد غير صحيحة")

if st.session_state['user_type'] not in ['admin', 'student']: login_screen(); st.stop()

# ==================== قلب النظام ====================
@st.cache_resource
def load_model():
    try: return joblib.load('iraqi_model.pkl')
    except: return None
model = load_model()

# --- المحاكاة الذكية ---
def simulate_improvement(row, model, current_score):
    scenarios = []
    def get_val(col): return row[col].values[0] if isinstance(row, pd.DataFrame) else row[col]
    val_eng = get_val('English_Score'); val_attend = get_val('Attendance_Rate')
    
    if val_eng < 60:
        d = row.copy(); d['English_Score'] += 20; p = model.predict(d)[0]
        if p > current_score: scenarios.append(f"تعزيز المهارات اللغوية (English Proficiency) قد يرفع المؤشر إلى <b>{p:.1f}%</b>")
    
    d_stu = row.copy(); d_stu['Study_Hours_Per_Week'] += 5; p_stu = model.predict(d_stu)[0]
    if p_stu > current_score: scenarios.append(f"زيادة ساعات الدراسة الذاتية (5 ساعات/أسبوع) سترفع المؤشر إلى <b>{p_stu:.1f}%</b>")
    
    if val_attend < 95:
        d_att = row.copy(); d_att['Attendance_Rate'] = 98; p_att = model.predict(d_att)[0]
        if p_att > current_score: scenarios.append(f"الانتظام التام في المحاضرات النظرية والعملية سيرفع المؤشر إلى <b>{p_att:.1f}%</b>")
    return scenarios

# --- توليد التقرير الرسمي ---
def generate_single_report_body(name, sid, dept, pred, steps, attend, study, eng, married):
    status = "مستوى حرج 🔴" if pred < 50 else "مستوى مطمئن 🟢"
    m_status = "متزوج" if married == 1 else "أعزب"
    rec_html = "".join([f"<li style='margin-bottom:5px;'>{s}</li>" for s in steps])
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" style="width: 110px; margin-bottom: 5px;">' if logo_base64 else ""
    
    body = f"""
    <div class="box page-break">
        <div class="header">
            {logo_html}
            <h2 style="margin:5px 0;">جامعة العين العراقية</h2>
            <h3 style="margin:0; font-weight:normal;">الكلية التقنية الهندسية - قسم {dept}</h3>
            <hr style="border-top: 2px solid #000; margin-top:15px;">
        </div>
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <table style="width:100%;">
                <tr><td style="text-align:right;"><strong>الطالب:</strong> {name}</td><td style="text-align:left;"><strong>الرقم الجامعي:</strong> <span class="num">{sid}</span></td></tr>
                <tr><td style="text-align:right;"><strong>الحالة الاجتماعية:</strong> {m_status}</td><td style="text-align:left;"><strong>كفاءة الإنجليزية:</strong> <span class="num">{eng}%</span></td></tr>
                <tr><td style="text-align:right;"><strong>تاريخ التقرير:</strong> <span class="num">{datetime.now().strftime('%Y-%m-%d')}</span></td><td style="text-align:left;"><strong>مؤشر الأداء المتوقع:</strong> <span class="num" style="font-size:1.2em;">{pred:.1f}%</span></td></tr>
            </table>
        </div>
        <div style="margin-top:25px;">
            <h4 style="border-bottom: 1px solid #ccc; padding-bottom: 5px;">أولاً: التشخيص الأكاديمي (Academic Diagnosis)</h4>
            <p style="line-height: 1.6;">بناءً على خوارزميات الذكاء الاصطناعي، تم تصنيف وضع الطالب ضمن: <strong>{status}</strong>. تشير البيانات إلى أن الالتزام بالحضور بنسبة (<span class="num">{attend}%</span>) والمجهود الدراسي الأسبوعي (<span class="num">{study}</span> ساعة) هما العاملان الأكثر تأثيراً.</p>
        </div>
        <div style="margin-top:20px;">
            <h4 style="border-bottom: 1px solid #ccc; padding-bottom: 5px;">ثانياً: خارطة الطريق المقترحة (Recommended Roadmap)</h4>
            <ul style="line-height: 1.6;">{rec_html}</ul>
        </div>
        <div style="margin-top: 50px; display: flex; justify-content: space-between;">
            <div style="text-align: center;">____________________<br>توقيع المرشد الأكاديمي</div>
            <div style="text-align: center;">____________________<br>ختم القسم العلمي</div>
        </div>
    </div><br class="no-print">"""
    return body

def generate_full_html_document(report_bodies, auto_print=False):
    print_script = "<script>window.onload = function() { window.print(); }</script>" if auto_print else ""
    html = f"""<!DOCTYPE html><html lang="ar" dir="rtl"><head><meta charset="UTF-8">
    <style>@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    body {{ font-family: 'Cairo', 'Times New Roman'; padding: 40px; background-color: #f4f4f4; }}
    .box {{ border: 1px solid #ddd; padding: 40px; max-width: 210mm; margin: auto; background-color: white; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    .header {{ text-align: center; }} table {{ width: 100%; border-collapse: collapse; }} td {{ padding: 10px; border-bottom: 1px solid #eee; }} .num {{ direction: ltr; display: inline-block; font-weight: bold; }}
    @media print {{ .no-print {{ display: none; }} body {{ background-color: white; padding: 0; }} .box {{ border: none; margin: 0; width: 100%; max-width: 100%; box-shadow: none; }} .page-break {{ page-break-after: always; }} }}
    </style>{print_script}</head><body><div class="report-container-wrapper">{report_bodies}</div></body></html>"""
    return html

# --- وظيفة عرض الداشبورد ---
def display_student_dashboard(name, sid, dept, pred, steps, attend, study, eng, married, part, att_val):
    t1, t2 = st.tabs(["لوحة المؤشرات البيانية", "التقرير الرسمي والطباعة"])
    with t1:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"<div class='metric-container'><h5>المعدل المتوقع</h5><h2 style='color:#2e86de'>{pred:.1f}%</h2></div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='metric-container'><h5>مستوى الإنجليزية</h5><h2 style='color:#10ac84'>{eng}%</h2></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='metric-container'><h5>نسبة الحضور</h5><h2 style='color:#ff9f43'>{attend}%</h2></div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='metric-container'><h5>ساعات الدراسة</h5><h2 style='color:#5f27cd'>{study}</h2></div>", unsafe_allow_html=True)
        
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=pred, title={'text':"مؤشر الأداء العام"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#0d2c56"}, 'steps':[{'range':[0,50],'color':'#ff7675'},{'range':[75,100],'color':'#55efc4'}]}))
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.subheader("📉 تحليل الفجوة (Gap Analysis)")
            categories = ['المعدل المتوقع', 'اللغة الإنجليزية', 'نسبة الحضور']
            student_vals = [pred, eng, attend]; target_vals = [85, 90, 95]
            fig_bar = go.Figure(data=[go.Bar(name='مستواك الحالي', x=categories, y=student_vals, marker_color='#0d2c56'), go.Bar(name='المستوى المستهدف', x=categories, y=target_vals, marker_color='#dfe6e9')])
            fig_bar.update_layout(barmode='group', height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.info("💡 **خارطة الطريق والتوصيات الذكية:**")
        for s in steps: st.markdown(f"<li style='direction: rtl; font-size:1.1em;'>{s}</li>", unsafe_allow_html=True)

    with t2:
        body = generate_single_report_body(name, sid, dept, pred, steps, attend, study, eng, married)
        html_dl = generate_full_html_document(body, auto_print=True)
        html_prev = generate_full_html_document(body, auto_print=False)
        components.html(html_prev, height=600, scrolling=True)
        st.download_button("🖨️ طباعة الوثيقة الرسمية", data=html_dl, file_name=f"Official_Report_{sid}.html", mime="text/html", type="primary")


# --- الواجهة الرئيسية ---
col_h1, col_h2 = st.columns([1, 4])
with col_h1:
    if logo_base64: st.markdown(f'<img src="data:image/png;base64,{logo_base64}" style="width: 100%;">', unsafe_allow_html=True)
with col_h2:
    st.title("النظام الجامعي الذكي للتنبؤ وتطوير الأداء")
    st.markdown("**جامعة العين العراقية - الكلية التقنية الهندسية**")
st.divider()

with st.sidebar:
    st.header("⚙️ الإعدادات")
    # ميزة المسؤول الجديدة: تحميل الداتا
    if st.session_state['user_type'] == 'admin':
        st.markdown("### 📥 بيانات الاستبيان")
        if os.path.isfile('collected_dataset.csv'):
            with open('collected_dataset.csv', 'rb') as f:
                st.download_button("تحميل قاعدة البيانات المجمعة", f, file_name="Students_Dataset.csv", mime='text/csv')
        else:
            st.caption("لا توجد بيانات محفوظة بعد.")
            
    if st.button("🚪 تسجيل الخروج", use_container_width=True): st.session_state['user_type']=None; st.rerun()

if st.session_state['user_type'] == 'admin':
    selected_mode = st.radio("اختر نمط العمل:", ["📥 إدخال بيانات فردي", "📂 استيراد ملف دفعة كاملة (Excel)"], horizontal=True)
else:
    selected_mode = "📥 إدخال بيانات فردي"

# --- نمط الإدخال الفردي (يعمل كاستبيان أيضاً) ---
if "إدخال بيانات فردي" in selected_mode:
    with st.expander("📝 بيانات الطالب الأكاديمية", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            s_name = st.text_input("الاسم الرباعي"); s_id = st.text_input("الرقم الجامعي"); s_dept = st.selectbox("القسم العلمي", ["هندسة الحاسوب", "هندسة تقنيات الحاسوب", "هندسة الأجهزة الطبية", "AI"])
        with c2:
            val_prev = st.slider("المعدل السابق (%)", 50, 100, 70); s_eng = st.slider("مستوى اللغة الإنجليزية (%)", 0, 100, 50); val_stu = st.number_input("ساعات الدراسة", 1, 50, 10)
        with c3:
            val_att = st.slider("نسبة الحضور (%)", 0, 100, 80); val_part = st.slider("التفاعل (1-10)", 1, 10, 5); val_fail = st.selectbox("الرسوب", [0, 1, 2, 3])
            s_married_opt = st.radio("الحالة الاجتماعية", ["أعزب", "متزوج"], horizontal=True); val_married = 1 if s_married_opt == "متزوج" else 0
        analyze_btn = st.button("🚀 إجراء التحليل الذكي", type="primary", use_container_width=True)

    if analyze_btn and s_name:
        row = pd.DataFrame({'Study_Hours_Per_Week': [val_stu], 'Attendance_Rate': [val_att], 'Previous_Average': [val_prev], 'Failures_History': [val_fail], 'Participation_Score': [val_part], 'Marital_Status': [val_married], 'English_Score': [s_eng]})
        pred = model.predict(row)[0]
        steps = simulate_improvement(row, model, pred)
        
        # --- هنا يتم حفظ البيانات سراً (الاستبيان) ---
        save_data_collection(s_name, s_id, s_dept, row, pred)
        
        st.markdown("---")
        st.subheader(f"📊 نتائج التحليل للطالب: {s_name}")
        display_student_dashboard(s_name, s_id, s_dept, pred, steps, val_att, val_stu, s_eng, val_married, val_part, val_att)

# --- نمط استيراد الملف ---
elif "استيراد ملف" in selected_mode:
    st.info("يرجى رفع ملف Excel يحتوي على بيانات الطلاب.")
    up_file = st.file_uploader("اختر الملف", type=['xlsx', 'csv'])
    
    if up_file:
        if up_file.name.endswith('.csv'): df_upload = pd.read_csv(up_file)
        else: df_upload = pd.read_excel(up_file)
        
        if st.button("⚡ بدء معالجة الدفعة"):
            if 'Marital_Status' not in df_upload.columns: df_upload['Marital_Status'] = 0
            if 'English_Score' not in df_upload.columns: df_upload['English_Score'] = 50
            preds = []
            for i, r in df_upload.iterrows():
                row = pd.DataFrame({'Study_Hours_Per_Week': [r['Study_Hours_Per_Week']], 'Attendance_Rate': [r['Attendance_Rate']], 'Previous_Average': [r['Previous_Average']], 'Failures_History': [r['Failures_History']], 'Participation_Score': [r['Participation_Score']], 'Marital_Status': [r['Marital_Status']], 'English_Score': [r['English_Score']]})
                preds.append(model.predict(row)[0])
            df_upload['Prediction'] = preds
            df_upload['Status'] = df_upload['Prediction'].apply(lambda x: 'مستوى مطمئن' if x >= 50 else 'مستوى حرج')
            st.session_state['batch_df'] = df_upload
            st.success("تمت المعالجة بنجاح!")

        if 'batch_df' in st.session_state:
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                total = len(st.session_state['batch_df']); critical = len(st.session_state['batch_df'][st.session_state['batch_df']['Status']=='مستوى حرج'])
                st.metric("إجمالي الطلاب", total); st.metric("في دائرة الخطر", critical, delta_color="inverse")
            with c2:
                fig_pie = px.pie(st.session_state['batch_df'], names='Status', title='توزيع حالة الدفعة', color='Status', color_discrete_map={'مستوى مطمئن':'#00b894', 'مستوى حرج':'#d63031'})
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("### 📋 سجل الطلاب (حدد طالباً واحداً للمعاينة، أو مجموعة للطباعة)")
            event = st.dataframe(st.session_state['batch_df'][['Student_Name', 'Department', 'Prediction', 'Status']], on_select="rerun", selection_mode="multi-row", use_container_width=True)
            sel_idx = event.selection.rows
            
            if len(sel_idx) == 0:
                st.info("👆 قم باختيار طالب من الجدول لعرض تفاصيله.")
            elif len(sel_idx) == 1:
                idx = sel_idx[0]; r = st.session_state['batch_df'].iloc[idx]
                sim_row = pd.DataFrame({'Study_Hours_Per_Week': [r['Study_Hours_Per_Week']], 'Attendance_Rate': [r['Attendance_Rate']], 'Previous_Average': [r['Previous_Average']], 'Failures_History': [r['Failures_History']], 'Participation_Score': [r['Participation_Score']], 'Marital_Status': [r['Marital_Status']], 'English_Score': [r['English_Score']]})
                steps = simulate_improvement(sim_row, model, r['Prediction'])
                st.markdown("---")
                st.subheader(f"🔍 التفاصيل الفردية للطالب: {r['Student_Name']}")
                display_student_dashboard(r['Student_Name'], str(r['Student_ID']), r['Department'], r['Prediction'], steps, r['Attendance_Rate'], r['Study_Hours_Per_Week'], r['English_Score'], r['Marital_Status'], r['Participation_Score'], r['Attendance_Rate'])
            else:
                st.success(f"✅ تم تحديد {len(sel_idx)} طالباً للطباعة الجماعية.")
                bodies = ""
                for idx in sel_idx:
                    r = st.session_state['batch_df'].iloc[idx]
                    sim_row = pd.DataFrame({'Study_Hours_Per_Week': [r['Study_Hours_Per_Week']], 'Attendance_Rate': [r['Attendance_Rate']], 'Previous_Average': [r['Previous_Average']], 'Failures_History': [r['Failures_History']], 'Participation_Score': [r['Participation_Score']], 'Marital_Status': [r['Marital_Status']], 'English_Score': [r['English_Score']]})
                    steps = simulate_improvement(sim_row, model, r['Prediction'])
                    bodies += generate_single_report_body(r['Student_Name'], str(r['Student_ID']), r['Department'], r['Prediction'], steps, r['Attendance_Rate'], r['Study_Hours_Per_Week'], r['English_Score'], r['Marital_Status'])
                final_html = generate_full_html_document(bodies, auto_print=True)
                st.download_button("🖨️ تحميل التقارير المجمعة (ملف واحد)", final_html, "Batch_Reports.html", "text/html", type="primary")

            with st.expander("خيارات متقدمة"):
                 if st.button("🖨️ طباعة تقارير الدفعة بالكامل"):
                    bodies = ""
                    for i, r in st.session_state['batch_df'].iterrows():
                        sim_row = pd.DataFrame({'Study_Hours_Per_Week': [r['Study_Hours_Per_Week']], 'Attendance_Rate': [r['Attendance_Rate']], 'Previous_Average': [r['Previous_Average']], 'Failures_History': [r['Failures_History']], 'Participation_Score': [r['Participation_Score']], 'Marital_Status': [r['Marital_Status']], 'English_Score': [r['English_Score']]})
                        steps = simulate_improvement(sim_row, model, r['Prediction'])
                        bodies += generate_single_report_body(r['Student_Name'], str(r['Student_ID']), r['Department'], r['Prediction'], steps, r['Attendance_Rate'], r['Study_Hours_Per_Week'], r['English_Score'], r['Marital_Status'])
                    final_html = generate_full_html_document(bodies, auto_print=True)
                    st.download_button("📥 تحميل الملف الشامل للدفعة", final_html, "Full_Batch.html", "text/html")
