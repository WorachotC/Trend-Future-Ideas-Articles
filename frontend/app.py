import streamlit as st
import requests
import os

# อ่าน URL Backend จาก Environment (รองรับ Docker)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ตั้งค่าหน้าจอ
st.set_page_config(
    page_title="Jenosize Trend Generator",
    page_icon="🚀",
    layout="wide"
)

# Header
st.title("🚀 Jenosize Future Ideas Generator")
st.markdown("Create insightful articles about trends and future ideas for businesses.")

# แบ่งหน้าจอเป็น 2 ฝั่ง (Left: Input, Right: Output)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Configuration")
    
    # 1. Inputs ตามโจทย์เป๊ะๆ 
    topic = st.text_input("Topic / Keyword", placeholder="e.g. AI in Healthcare 2025")
    
    industry = st.selectbox(
        "Industry",
        ["Technology", "Marketing", "Finance", "Healthcare", "Retail", "Other"]
    )
    
    target_audience = st.selectbox(
        "Target Audience",
        ["Business Owners", "Tech Enthusiasts", "Investors", "General Public"]
    )
    
    tone = st.select_slider(
        "Tone & Style",
        options=["Casual", "Professional", "Visionary", "Urgent"]
    )
    
    # Feature เด็ด: RAG Input
    source_url = st.text_input("Source URL (Optional)", placeholder="https://techcrunch.com/...")
    
    generate_btn = st.button("✨ Generate Article", use_container_width=True, type="primary")

with col2:
    st.subheader("📝 Generated Article")
    
    if generate_btn:
        if not topic:
            st.warning("Please enter a topic first.")
        else:
            with st.spinner("🤖 AI is researching and writing... (this may may take a moment)"):
                try:
                    # ยิง Request ไปหา Backend (FastAPI)
                    payload = {
                        "topic": topic,
                        "industry": industry,
                        "target_audience": target_audience,
                        "tone": tone,
                        "source_url": source_url
                    }
                    
                    response = requests.post(f"{API_URL}/generate", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        article_content = data.get("article", "")
                        
                        # แสดงผลแบบ Markdown สวยๆ
                        st.markdown(article_content)
                        
                        # ปุ่ม Download ไฟล์
                        st.download_button(
                            label="📥 Download as Markdown",
                            data=article_content,
                            file_name=f"{topic}_jenosize_article.md",
                            mime="text/markdown"
                        )
                    else:
                        st.error(f"Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"Connection Error: {e}")
                    st.info("Ensure Backend is running and reachable.")

# Footer
st.markdown("---")
st.caption("Powered by Jenosize AI Model | Designed for Test Assignment Option 1")