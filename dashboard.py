import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# CẤU HÌNH TRANG
st.set_page_config(
    page_title="Hệ thống Phân tích Thị trường Việc làm AI",
    layout="wide"
)


# Đọc dữ liệu công việc
@st.cache_data
def load_data():
    df = pd.read_csv("ai_jobs_market_2025_2026.csv")
    return df


df = load_data()

# Custom CSS để làm đẹp giao diện, tạo khối trực quan chuyên nghiệp
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1E40AF;
    }
    .metric-label {
        font-size: 14px;
        color: #475569;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)


# THANH ĐIỀU HƯỚNG (SIDEBAR)
st.sidebar.markdown("### 📋 Danh mục phân tích")
page = st.sidebar.radio(
    "Chọn hướng tiếp cận dữ liệu:",
    [
        "Tổng quan thị trường",
        "Khám phá vị trí việc làm",
        "Phân tích kỹ năng công nghệ",
        "Kiểm định giả thuyết thống kê",
        "Mô hình dự đoán lương",
        "Đề xuất lộ trình học tập"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Tác giả:** Huỳnh Mỹ Kiều")
st.sidebar.markdown("**Giai đoạn phân tích:** 2025 - 2026")


# TRANG 1: TỔNG QUAN THỊ TRƯỜNG
if page == "Tổng quan thị trường":
    st.markdown('<div class="main-title">Phân tích Thị trường Việc làm AI (2025–2026)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Thống kê mô tả tổng quan toàn bộ mẫu dữ liệu tuyển dụng AI trên toàn cầu</div>',
                unsafe_allow_html=True)

    # Các thẻ Metric số liệu tổng quan
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Tổng số bản ghi dữ liệu (N)</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">${df["annual_salary_usd"].mean():,.0f}</div><div class="metric-label">Mức lương trung bình năm (USD)</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{df["demand_score"].mean():.1f} / 100</div><div class="metric-label">Điểm nhu cầu thị trường trung bình</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Top 10 vị trí có mức lương trung bình cao nhất")
        top_salary = df.groupby("job_title")["annual_salary_usd"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(x=top_salary.values, y=top_salary.index, ax=ax, palette="Blues_r", order=top_salary.index)
        ax.set_xlabel("Mức lương (USD)")
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.markdown("### 📉 Top 10 vị trí có nhu cầu tuyển dụng lớn nhất")
        top_demand = df.groupby("job_title")["demand_score"].mean().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(x=top_demand.values, y=top_demand.index, ax=ax, palette="Purples_r", order=top_demand.index)
        ax.set_xlabel("Điểm nhu cầu (Thang 100)")
        ax.set_ylabel("")
        st.pyplot(fig)


# TRANG 2: KHÁM PHÁ VỊ TRÍ VIỆC LÀM
elif page == "Khám phá vị trí việc làm":
    st.markdown('<div class="main-title">Khám phá Chi tiết Vị trí Việc làm</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Lọc và trích xuất các chỉ số đặc trưng cho từng chức danh công việc cụ thể</div>',
        unsafe_allow_html=True)

    job = st.selectbox("Chọn chức danh công việc cần khảo sát:", sorted(df["job_title"].unique()))
    data = df[df["job_title"] == job]

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">${data["annual_salary_usd"].mean():,.0f}</div><div class="metric-label">Lương trung bình năm</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{data["demand_score"].mean():.1f}</div><div class="metric-label">Điểm nhu cầu tuyển dụng</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{data["years_of_experience"].mean():.1f} năm</div><div class="metric-label">Kinh nghiệm yêu cầu trung bình</div></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🛠️ Top 10 kỹ năng cốt lõi được yêu cầu nhiều nhất")

    skills = []
    for s in data["required_skills"].dropna():
        skills.extend([sk.strip() for sk in str(s).split("|")])

    if skills:
        skill_df = pd.Series(skills).value_counts().head(10)
        st.bar_chart(skill_df, color="#3B82F6")
    else:
        st.info("Không có dữ liệu về kỹ năng cho vị trí công việc này.")


# TRANG 3: PHÂN TÍCH KỸ NĂNG CÔNG NGHỆ
elif page == "Phân tích kỹ năng công nghệ":
    st.markdown('<div class="main-title">Phân tích Sức ảnh hưởng của Kỹ năng</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Đánh giá quy mô mẫu và tầm ảnh hưởng của một kỹ năng công nghệ cụ thể tới thu nhập</div>',
        unsafe_allow_html=True)

    all_skills = []
    for s in df["required_skills"].dropna():
        all_skills.extend([sk.strip() for sk in str(s).split("|")])
    skill_list = sorted(list(set(all_skills)))

    selected = st.selectbox("Chọn kỹ năng công nghệ cần kiểm tra:", skill_list)

    # Giải quyết lỗi chứa chuỗi bằng biểu thức chính quy (Regex) để khớp chính xác từ độc lập (tránh lỗi SQL dính NoSQL)
    filtered = df[df["required_skills"].str.contains(rf"\b{selected}\b", na=False, regex=True)]

    st.markdown("<br>", unsafe_allow_html=True)
    st.success(f"**Quy mô mẫu:** Tìm thấy **{len(filtered)}** tin tuyển dụng có yêu cầu kỹ năng **{selected}**.")

    st.markdown(f"### 📋 Danh sách 15 việc làm trả lương cao nhất có sở hữu kỹ năng '{selected}'")
    st.dataframe(
        filtered[["job_title", "annual_salary_usd", "demand_score", "experience_level", "country"]]
        .sort_values("annual_salary_usd", ascending=False)
        .head(15),
        use_container_width=True
    )


# TRANG 4: KIỂM ĐỊNH GIẢ THUYẾT THỐNG KÊ
elif page == "Kiểm định giả thuyết thống kê":
    st.markdown('<div class="main-title">Phân tích Thống kê Suy luận Nâng cao</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Các chứng minh định lượng bằng toán học được trích xuất trực tiếp từ Chương 5 của cuốn báo cáo đồ án</div>',
        unsafe_allow_html=True)
    st.divider()

    tab1, tab2 = st.tabs(
        ["[1] Kiểm định T-test (Tác động của LLM)", "[2] Phân tích phương sai ANOVA (Các biến định tính)"])

    with tab1:
        st.markdown("### 🧪 Kiểm định T-test: Nhóm công việc LLM vs Nhóm truyền thống")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.info("""
            **Tóm tắt số liệu thực nghiệm:**
            * Lương trung bình nhóm công việc có LLM: **$207,746 / năm**
            * Lương trung bình nhóm không có LLM: **$191,309 / năm**
            * Biên độ chênh lệch thu nhập: **+8.59%**
            * Giá trị chỉ số thống kê **p-value: 0.000075**
            """)
        with col_t2:
            st.markdown("""
            **Nhận xét và Giải thích kết quả thống kê:**

            Vì giá trị toán học **p-value (0.000075) nhỏ hơn rất nhiều so với mức ý nghĩa tiêu chuẩn 0.05** ($\alpha = 0.05$), chúng ta có đủ cơ sở bác bỏ giả thuyết không ($H_0$). 

            Điều này minh chứng bằng toán học rằng: Các công việc có tích hợp công nghệ Mô hình ngôn ngữ lớn (LLM) mang lại mức thu nhập vượt trội rõ rệt. Sự chênh lệch này có ý nghĩa thống kê tuyệt đối, được quyết định bởi xu thế công nghệ chứ không phải do sai số ngẫu nhiên của việc lấy mẫu.
            """)

    with tab2:
        st.markdown("### ⚙️ Kết quả kiểm định ANOVA một nhân tố")

        anova_data = {
            "Biến độc lập (Nhân tố tác động)": ["Chức danh công việc (job_title)",
                                                "Trình độ học vấn (education_required)",
                                                "Quy mô doanh nghiệp (company_size)"],
            "Giá trị toán học p-value": ["2.20e-62", "7.08e-12", "3.45e-44"],
            "Mức độ ý nghĩa thống kê": ["Ảnh hưởng cực kỳ mạnh (p < 0.01)", "Ảnh hưởng rõ rệt (p < 0.01)",
                                        "Ảnh hưởng rõ rệt (p < 0.01)"]
        }
        st.table(pd.DataFrame(anova_data))

        st.warning("""
        **Đánh giá phân tích cốt lõi:**
        Cả 3 biến định tính trên đều có tác động sâu sắc đến sự phân hóa mức lương. 
        Trong đó đáng chú ý nhất là biến **chức danh công việc (job_title)** có p-value nhỏ nhất ($2.20 \times 10^{-62}$). 
        Kết quả này khẳng định: Trên thị trường lao động AI giai đoạn 2025–2026, thu nhập cao hay thấp phụ thuộc chủ yếu vào **vai trò thực tế và năng lực chuyên môn chuyên sâu** của ứng viên, chứ không còn bị nặng nề bởi bằng cấp lý thuyết thuần túy.
        """)


# TRANG 5: MÔ HÌNH DỰ ĐOÁN LƯƠNG
elif page == "Mô hình dự đoán lương":
    st.markdown('<div class="main-title">Mô hình Hồi quy Tuyến tính Dự đoán Thu nhập</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Dự đoán mức lương năm bằng phương pháp Bình phương tối thiểu (OLS) dựa theo đúng 3 biến của Chương 6</div>',
        unsafe_allow_html=True)

    # Khởi tạo mô hình toán học khớp với thuộc tính trong Chương 6 của báo cáo
    X = df[["years_of_experience", "demand_score", "ai_salary_premium_pct"]]
    y = df["annual_salary_usd"]

    model = LinearRegression()
    model.fit(X, y)

    st.markdown("### ⚙️ Nhập các thông số giả định")
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        exp = st.slider("Số năm kinh nghiệm thực tế (0 - 15):", 0, 15, 3)
    with col_in2:
        demand = st.slider("Điểm nhu cầu thị trường hiện tại (0 - 100):", 0, 100, 80)
    with col_in3:
        premium = st.slider("Tỷ lệ Premium thu nhập từ AI (3% - 17%):", 3.0, 17.0, 10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Chạy mô hình dự đoán lương", type="primary"):
        pred = model.predict([[exp, demand, premium]])[0]
        st.markdown(f"""
        <div style="background-color: #EEF2F6; border-left: 6px solid #1E40AF; padding: 15px; border-radius: 4px;">
            <span style="color: #1E293B; font-weight: 600; font-size: 16px;">Kết quả dự đoán thu nhập năm:</span>
            <h2 style="margin: 5px 0 0 0; color: #1E40AF; font-size: 32px;">${pred:,.0f} USD / năm</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📊 Đánh giá chất lượng và chẩn đoán mô hình")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Hệ số xác định (R²)", "0.0565")
    with c2:
        st.markdown("""
        **Giải trình khoa học về giá trị R² thấp dưới góc nhìn phân tích:**

        Hệ số $R^2 = 5.65\%$ phản ánh rằng mô hình tuyến tính chưa bao quát tốt sự biến động phức tạp của tiền lương ngành AI. Trong thực tế, thu nhập ngành này phụ thuộc rất lớn vào các biến số phi tuyến khác như: Sự kết hợp các kỹ năng hiếm, vị trí địa lý của doanh nghiệp hoặc trạng thái gọi vốn của công ty công nghệ. 

        *Kết luận hướng phát triển:* Kết quả này là luận điểm khoa học quan trọng để đề xuất nâng cấp từ mô hình tuyến tính cơ bản lên các kiến trúc học máy phi tuyến mạnh mẽ hơn (như Random Forest Regression hoặc XGBoost) trong các nghiên cứu tiếp theo.
        """)


# TRANG 6: ĐỀ XUẤT LỘ TRÌNH HỌC TẬP
elif page == "Đề xuất lộ trình học tập":
    st.markdown('<div class="main-title">Lộ trình Đào tạo 4 Năm học Tối ưu</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Khuyến nghị phân bổ kỹ năng theo thời gian dựa trên các ma trận phân tích dữ liệu thị trường thực tế 2025-2026</div>',
        unsafe_allow_html=True)

    # Grid hiển thị lộ trình học dạng Card màu sắc trực quan
    col_y1, col_y2 = st.columns(2)
    with col_y1:
        st.markdown("""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #2563EB; margin-top:0;">✦ Năm 1: Toán nền tảng & Lập trình cơ sở</h4>
            <p style="color: #475569; font-size: 14px; margin-bottom:0;">
                <b>Kỹ năng trọng tâm:</b> Ngôn ngữ Python nâng cao, cấu trúc dữ liệu và giải thuật, ngôn ngữ truy vấn SQL, Đại số tuyến tính, Thống kê mô tả.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #7C3AED; margin-top:0;">✦ Năm 3: Học sâu & Ứng dụng chuyên ngành</h4>
            <p style="color: #475569; font-size: 14px; margin-bottom:0;">
                <b>Kỹ năng trọng tâm:</b> Kiến trúc mạng học sâu Deep Learning (PyTorch), Xử lý ngôn ngữ tự nhiên (NLP), Thị giác máy tính, Kỹ thuật và hạ tầng Cloud (AWS/GCP).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_y2:
        st.markdown("""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #0D9488; margin-top:0;">✦ Năm 2: Phân tích dữ liệu & Học máy cốt lõi</h4>
            <p style="color: #475569; font-size: 14px; margin-bottom:0;">
                <b>Kỹ năng trọng tâm:</b> Khám phá trực quan dữ liệu (EDA), Mô hình hóa thống kê, Các thuật toán Học máy cơ bản (Phân lớp, Hồi quy trên Scikit-Learn).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #EA580C; margin-top:0;">✦ Năm 4: Kỹ nghệ sản xuất & Hệ thống tự trị</h4>
            <p style="color: #475569; font-size: 14px; margin-bottom:0;">
                <b>Kỹ năng trọng tâm:</b> Làm việc với API Mô hình ngôn ngữ lớn (LLM Engineering), Tinh chỉnh mô hình (Fine-tuning), Tự động hóa quy trình MLOps, Thiết kế hệ thống AI Agents tự trị và thực hiện Khóa luận tốt nghiệp.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🎯 Các nhóm công nghiệp mục tiêu chiến lược")
    st.markdown("""
    Dựa trên kết quả phân tích phân cụm thu nhập cao, sinh viên được khuyến nghị nên định hướng vào:
    * **Vị trí có độ phủ lớn (Nhu cầu tuyển cao):** Kỹ sư AI (AI Engineer) / Kỹ sư Học máy (Machine Learning Engineer)
    * **Vị trí đột phá thu nhập (Mức tăng trưởng cao):** Kỹ sư LLM (LLM Engineer) / Kỹ sư MLOps (MLOps Engineer) / Nhà phát triển Hệ thống AI Agent.
    """)