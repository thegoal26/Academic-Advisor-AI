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

# --- إعداد الصفحة ---
st.set_page_config(page_title="النظام الجامعي الذكي", layout="wide", page_icon="🎓")

# --- CSS ---
st.markdown("""
<style>
    @media print {
        body { visibility: hidden; background-color: white !important; }
        header, footer, .stSidebar, .stButton, .stApp > header, .stApp > footer, .stTabs { display: none !important; }
        .no-print { display: none !important; }
        .report-container-wrapper {
            visibility: visible !important; position: absolute !important; left: 0 !important; top: 0 !important;
            width: 100% !important; margin: 0 !important; padding: 0 !important; z-index: 9999 !important; background-color: white !important;
        }
        .report-container-wrapper * { visibility: visible !important; }
        .page-break { page-break-after: always; }
        @page { margin: 0.5cm; size: A4 portrait; }
    }
    .metric-card {
        background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-right: 5px solid #2e86de;
        margin-bottom: 10px; color: #000; font-weight: bold; text-align: right; direction: rtl; font-size: 16px;
    }
    .num-ltr { direction: ltr; unicode-bidi: embed; display: inline-block; color: #d63031; }
</style>
""", unsafe_allow_html=True)

# --- تسجيل الدخول ---
if 'user_type' not in st.session_state: st.session_state['user_type'] = None
def login_screen():
    st.markdown("<br><br><h1 style='text-align: center; color: #1f77b4;'>🎓 بوابة النظام الأكاديمي الذكي</h1>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        col_st, col_ad = st.columns(2)
        with col_st:
            if st.button("👤 أنا طالب", use_container_width=True, type="primary"): st.session_state['user_type'] = 'student'; st.rerun()
        with col_ad:
            if st.button("🔐 دخول مسؤولين", use_container_width=True): st.session_state['user_type'] = 'login_attempt'; st.rerun()  
    if st.session_state['user_type'] == 'login_attempt':
        with c2:
            st.info("المنطقة الإدارية")
            user = st.text_input("اسم المستخدم"); pw = st.text_input("كلمة المرور", type="password")
            if st.button("تأكيد", type="secondary", use_container_width=True):
                if user == "admin" and pw == "1234": st.session_state['user_type'] = 'admin'; st.rerun()
                else: st.error("خطأ")
if st.session_state['user_type'] not in ['admin', 'student']: login_screen(); st.stop()

# ==================== النظام ====================
@st.cache_resource
def load_model():
    try: return joblib.load('iraqi_model.pkl')
    except: return None
model = load_model()

# --- المحاكاة ---
def simulate_improvement(row, model, current_score):
    scenarios = []
    def get_val(col): return row[col].values[0] if isinstance(row, pd.DataFrame) else row[col]
    val_eng = get_val('English_Score'); val_attend = get_val('Attendance_Rate')
    if val_eng < 60:
        d = row.copy(); d['English_Score'] += 20; p = model.predict(d)[0]
        if p > current_score: scenarios.append(f"دورة تقوية بالإنجليزية سترفع المعدل إلى <span class='num-ltr'>{p:.1f}%</span>")
    d_stu = row.copy(); d_stu['Study_Hours_Per_Week'] += 5; p_stu = model.predict(d_stu)[0]
    if p_stu > current_score: scenarios.append(f"زيادة الدراسة (5 ساعات) سترفع المعدل إلى <span class='num-ltr'>{p_stu:.1f}%</span>")
    if val_attend < 95:
        d_att = row.copy(); d_att['Attendance_Rate'] = 98; p_att = model.predict(d_att)[0]
        if p_att > current_score: scenarios.append(f"الالتزام بالدوام سيرفع النتيجة إلى <span class='num-ltr'>{p_att:.1f}%</span>")
    return scenarios

# --- HTML Generator (جسم التقرير فقط) ---
def generate_single_report_body(name, sid, dept, pred, steps, attend, study, eng, married):
    status = "خطر 🔴" if pred < 50 else "جيد 🟢"
    m_status = "متزوج" if married == 1 else "أعزب"
    rec_html = "".join([f"<li>{s}</li>" for s in steps])
    
    body = f"""
    <div class="box page-break">
        <div class="header">
            <img src="https://cdn-icons-png.flaticon.com/512/2231/2231649.png">
            <h2>وزارة التعليم العالي</h2><h3>القسم: {dept}</h3>
        </div>
        <hr>
        <table>
            <tr><td><strong>الطالب:</strong> {name}</td><td><strong>الرقم:</strong> <span class="num">{sid}</span></td></tr>
            <tr><td><strong>الحالة:</strong> {m_status}</td><td><strong>اللغة الإنجليزية:</strong> <span class="num">{eng}%</span></td></tr>
            <tr><td><strong>التاريخ:</strong> <span class="num">{datetime.now().strftime('%Y-%m-%d')}</span></td><td><strong>المعدل المتوقع:</strong> <span class="num">{pred:.1f}%</span></td></tr>
        </table>
        <br>
        <div style="border:1px solid #000; padding:15px;">
            <h4>📊 ملخص التحليل:</h4>
            <p>حالة الطالب: <strong>{status}</strong>. يعتمد ذلك على الحضور (<span class="num">{attend}%</span>) والدراسة (<span class="num">{study}</span> ساعة).</p>
        </div>
        <div style="border:1px solid #000; padding:15px; margin-top:10px;">
            <h4>🚀 التوصيات:</h4><ul>{rec_html}</ul>
        </div>
        <br><br><center>ختم وتوقيع القسم</center>
    </div>
    <br class="no-print">
    """
    return body

# --- المجمع النهائي (تم التعديل للتحكم بالطباعة) ---
def generate_full_html_document(report_bodies, auto_print=False):
    # إذا كانت auto_print صحيحة، نضيف كود الجافا سكربت. إذا خطأ، لا نضيفه.
    print_script = "<script>window.onload = function() { window.print(); }</script>" if auto_print else ""
    
    html = f"""
    <!DOCTYPE html><html lang="ar" dir="rtl"><head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: 'Times New Roman'; padding: 40px; background-color: #eee; }}
        .box {{ border: 2px solid #000; padding: 30px; max-width: 210mm; margin: auto; background-color: white; margin-bottom: 20px; }}
        .header {{ text-align: center; }} .header img {{ width: 80px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        .num {{ direction: ltr; display: inline-block; font-weight: bold; }}
        @media print {{ 
            .no-print {{ display: none; }} 
            body {{ background-color: white; padding: 0; }}
            .box {{ border: none; margin: 0; width: 100%; max-width: 100%; }}
            .page-break {{ page-break-after: always; }}
        }}
    </style>
    {print_script}
    </head><body>
    <div class="report-container-wrapper">
        {report_bodies}
    </div>
    </body></html>
    """
    return html

# --- عرض التفاصيل (تم الإصلاح هنا) ---
def show_single_student_dashboard(name, sid, dept, pred, steps, attend, study, prev, partic, eng, married):
    st.divider()
    t1, t2 = st.tabs(["📊 لوحة التحليل", "📄 معاينة التقرير"])
    with t1:
        st.markdown(f"### 👤 {name}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("التوقع", f"{pred:.1f}%", delta=f"{pred-50:.1f}")
        c2.metric("الإنجليزية", f"{eng}%"); c3.metric("الحالة", "متزوج" if married else "أعزب")
        c4.metric("الحضور", f"{attend}%"); c5.metric("الدراسة", f"{study}")
        col_g, col_c = st.columns([1, 2])
        with col_g:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=pred, title={'text':"المؤشر"}, gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#1f77b4"}, 'steps':[{'range':[0,50],'color':'#ffcbcb'},{'range':[75,100],'color':'#d9ead3'}]}))
            st.plotly_chart(fig, use_container_width=True)
        with col_c:
            cats = ['التوقع', 'الإنجليزية', 'الحضور']; vals = [pred, eng, attend]; target = [85, 90, 100]
            fig_b = go.Figure(data=[go.Bar(name='أنت', x=cats, y=vals, marker_color='#1f77b4'), go.Bar(name='الهدف', x=cats, y=target, marker_color='#d63031')])
            st.plotly_chart(fig_b, use_container_width=True)
        st.subheader("💡 التوصيات")
        if steps:
            for s in steps: st.markdown(f'<div class="metric-card">✅ {s}</div>', unsafe_allow_html=True)
        else: st.success("ممتاز!")
    
    with t2:
        # جسم التقرير
        body = generate_single_report_body(name, sid, dept, pred, steps, attend, study, eng, married)
        
        # 1. للمعانية: نولد HTML بدون سكربت الطباعة
        html_preview = generate_full_html_document(body, auto_print=False)
        components.html(html_preview, height=600, scrolling=True)
        
        # 2. للتحميل: نولد HTML مع سكربت الطباعة
        html_download = generate_full_html_document(body, auto_print=True)
        st.download_button("🖨️ تحميل للطباعة", data=html_download, file_name=f"Report_{sid}.html", mime="text/html", type="primary")

# --- القائمة الجانبية ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3068/3068327.png", width=80)
    if st.button("🚪 خروج"): st.session_state['user_type']=None; st.rerun()
    st.divider()
    if st.session_state['user_type'] == 'admin':
        st.info("وضع المسؤول")
        mode = st.radio("الوضع:", ["إدخال يدوي", "استيراد ملف"])
    else:
        st.success("وضع الطالب")
        mode = "إدخال يدوي"

    if mode == "إدخال يدوي":
        s_name = st.text_input("الاسم"); s_id = st.text_input("الرقم")
        s_dept = st.selectbox("القسم", ["هندسة الحاسوب", "علوم الحاسوب", "IT", "AI"])
        s_married = st.radio("الحالة:", ["أعزب", "متزوج"], horizontal=True)
        val_married = 1 if s_married == "متزوج" else 0
        s_eng = st.slider("الإنجليزية", 0, 100, 50)
        with st.expander("تفاصيل الأداء"):
            val_prev = st.slider("المعدل السابق", 50, 100, 70); val_att = st.slider("الحضور", 0, 100, 80)
            val_stu = st.number_input("ساعات الدراسة", 1, 50, 10); val_part = st.slider("التفاعل", 1, 10, 5)
            val_fail = st.selectbox("الرسوب", [0, 1, 2, 3])
        btn = st.button("🚀 تحليل", type="primary")
    elif mode == "استيراد ملف":
        up_file = st.file_uploader("ملف Excel", type=['xlsx', 'csv'])
        sample = pd.DataFrame(columns=['Student_Name', 'Student_ID', 'Department', 'Study_Hours_Per_Week', 'Attendance_Rate', 'Previous_Average', 'Failures_History', 'Participation_Score', 'Marital_Status', 'English_Score'])
        buf = io.BytesIO(); 
        with pd.ExcelWriter(buf, engine='openpyxl') as w: sample.to_excel(w, index=False)
        st.download_button("📥 قالب (V27)", buf.getvalue(), "template_v27.xlsx")

# --- التشغيل ---
st.title("🎓 النظام الجامعي الذكي")
if not model: st.error("⚠️ الموديل غير موجود"); st.stop()

if mode == "إدخال يدوي" and btn:
    if s_name:
        row = pd.DataFrame({'Study_Hours_Per_Week': [val_stu], 'Attendance_Rate': [val_att], 'Previous_Average': [val_prev], 'Failures_History': [val_fail], 'Participation_Score': [val_part], 'Marital_Status': [val_married], 'English_Score': [s_eng]})
        pred = model.predict(row)[0]
        steps = simulate_improvement(row, model, pred)
        show_single_student_dashboard(s_name, s_id, s_dept, pred, steps, val_att, val_stu, val_prev, val_part, s_eng, val_married)

elif mode == "استيراد ملف" and up_file:
    if up_file.name.endswith('.csv'): df_upload = pd.read_csv(up_file)
    else: df_upload = pd.read_excel(up_file)
    
    if st.button("⚡ تحليل الدفعة كاملة"):
        if 'Marital_Status' not in df_upload.columns: df_upload['Marital_Status'] = 0
        if 'English_Score' not in df_upload.columns: df_upload['English_Score'] = 50
        preds = []
        for i, r in df_upload.iterrows():
            row = pd.DataFrame({'Study_Hours_Per_Week': [r['Study_Hours_Per_Week']], 'Attendance_Rate': [r['Attendance_Rate']], 'Previous_Average': [r['Previous_Average']], 'Failures_History': [r['Failures_History']], 'Participation_Score': [r['Participation_Score']], 'Marital_Status': [r['Marital_Status']], 'English_Score': [r['English_Score']]})
            preds.append(model.predict(row)[0])
        df_upload['Prediction'] = preds
        df_upload['Status'] = df_upload['Prediction'].apply(lambda x: 'ناجح' if x >= 50 else 'في خطر')
        st.session_state['batch_df'] = df_upload
        st.success("تم التحليل!")

    if 'batch_df' in st.session_state:
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            total = len(st.session_state['batch_df'])
            failed = len(st.session_state['batch_df'][st.session_state['batch_df']['Status']=='في خطر'])
            st.metric("إجمالي الطلاب", total); st.metric("في دائرة الخطر", failed, delta_color="inverse")
        with c2:
            fig_pie = px.pie(st.session_state['batch_df'], names='Status', title='نسبة الخطر في الدفعة', color='Status', color_discrete_map={'ناجح':'#2ecc71', 'في خطر':'#e74c3c'})
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.divider()
        st.subheader("📋 قائمة الطلاب")
        
        df_display = st.session_state['batch_df'][['Student_Name', 'Department', 'Prediction', 'Status']]
        event = st.dataframe(df_display, on_select="rerun", selection_mode="multi-row", use_container_width=True)
        selected_indices = event.selection.rows
        
        if len(selected_indices) == 0:
            st.info("👆 اختر طالباً من الجدول.")
        elif len(selected_indices) == 1:
            idx = selected_indices[0]; full_r = st.session_state['batch_df'].iloc[idx]
            sim_row = pd.DataFrame({'Study_Hours_Per_Week': [full_r['Study_Hours_Per_Week']], 'Attendance_Rate': [full_r['Attendance_Rate']], 'Previous_Average': [full_r['Previous_Average']], 'Failures_History': [full_r['Failures_History']], 'Participation_Score': [full_r['Participation_Score']], 'Marital_Status': [full_r['Marital_Status']], 'English_Score': [full_r['English_Score']]})
            steps = simulate_improvement(sim_row, model, full_r['Prediction'])
            show_single_student_dashboard(full_r['Student_Name'], str(full_r['Student_ID']), full_r['Department'], full_r['Prediction'], steps, full_r['Attendance_Rate'], full_r['Study_Hours_Per_Week'], full_r['Previous_Average'], full_r['Participation_Score'], full_r['English_Score'], full_r['Marital_Status'])
        else:
            st.success(f"✅ تم تحديد {len(selected_indices)} طالباً.")
            all_reports_body = ""
            for idx in selected_indices:
                full_r = st.session_state['batch_df'].iloc[idx]
                sim_row = pd.DataFrame({'Study_Hours_Per_Week': [full_r['Study_Hours_Per_Week']], 'Attendance_Rate': [full_r['Attendance_Rate']], 'Previous_Average': [full_r['Previous_Average']], 'Failures_History': [full_r['Failures_History']], 'Participation_Score': [full_r['Participation_Score']], 'Marital_Status': [full_r['Marital_Status']], 'English_Score': [full_r['English_Score']]})
                steps = simulate_improvement(sim_row, model, full_r['Prediction'])
                all_reports_body += generate_single_report_body(full_r['Student_Name'], str(full_r['Student_ID']), full_r['Department'], full_r['Prediction'], steps, full_r['Attendance_Rate'], full_r['Study_Hours_Per_Week'], full_r['English_Score'], full_r['Marital_Status'])
            # هنا نستخدم auto_print=True لأن هذا زر تحميل للطباعة
            final_html = generate_full_html_document(all_reports_body, auto_print=True)
            st.download_button(label=f"📥 تحميل وطباعة {len(selected_indices)} تقرير", data=final_html, file_name="Batch_Reports.html", mime="text/html", type="primary")
        
        st.markdown("---")
        with st.expander("خيارات إضافية"):
            if st.button("🖨️ طباعة الدفعة كاملة"):
                all_reports_body = ""
                for idx, full_r in st.session_state['batch_df'].iterrows():
                    sim_row = pd.DataFrame({'Study_Hours_Per_Week': [full_r['Study_Hours_Per_Week']], 'Attendance_Rate': [full_r['Attendance_Rate']], 'Previous_Average': [full_r['Previous_Average']], 'Failures_History': [full_r['Failures_History']], 'Participation_Score': [full_r['Participation_Score']], 'Marital_Status': [full_r['Marital_Status']], 'English_Score': [full_r['English_Score']]})
                    steps = simulate_improvement(sim_row, model, full_r['Prediction'])
                    all_reports_body += generate_single_report_body(full_r['Student_Name'], str(full_r['Student_ID']), full_r['Department'], full_r['Prediction'], steps, full_r['Attendance_Rate'], full_r['Study_Hours_Per_Week'], full_r['English_Score'], full_r['Marital_Status'])
                # طباعة الكل -> طباعة تلقائية
                final_html = generate_full_html_document(all_reports_body, auto_print=True)
                st.download_button("📥 تحميل ملف الدفعة الكامل", data=final_html, file_name="Full_Batch_Reports.html", mime="text/html")
