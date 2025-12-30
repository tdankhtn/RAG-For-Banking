#!/bin/bash
# Script hướng dẫn upload dự án lên GitHub

echo "==================================="
echo "HƯỚNG DẪN UPLOAD DỰ ÁN LÊN GITHUB"
echo "==================================="

echo ""
echo "📋 CHUẨN BỊ:"
echo "1. Tạo repository trên GitHub (https://github.com/new)"
echo "   - Tên repo: VD: 'rag-vietnamese-banking'"
echo "   - Public hoặc Private"
echo "   - KHÔNG tích 'Initialize with README' (vì đã có sẵn)"
echo ""

echo "🔐 LƯU Ý QUAN TRỌNG:"
echo "- File .env (chứa API keys) đã được .gitignore"
echo "- Thư mục storage/ (vector DB) không upload"
echo "- Thư mục .venv/ (môi trường ảo) không upload"
echo ""

read -p "Đã tạo repo trên GitHub? (y/n) " -n 1 -r
echo
if [[ ! $REPONSE =~ ^[Yy]$ ]]; then
    echo "Hãy tạo repo trước, sau đó chạy lại script này!"
    exit 1
fi

echo ""
echo "📝 Nhập thông tin repo:"
read -p "GitHub username: " username
read -p "Tên repository: " reponame

echo ""
echo "🚀 Đang khởi tạo Git..."

# Init git nếu chưa có
if [ ! -d ".git" ]; then
    git init
    echo "✅ Git initialized"
else
    echo "⚠️  Git đã được init từ trước"
fi

# Add remote
git remote remove origin 2>/dev/null
git remote add origin "https://github.com/$username/$reponame.git"
echo "✅ Remote added: https://github.com/$username/$reponame.git"

echo ""
echo "📦 Đang add files..."
git add .
echo "✅ Files added"

echo ""
echo "💬 Commit..."
git commit -m "Initial commit: RAG Vietnamese Banking with JSON chunks"
echo "✅ Committed"

echo ""
echo "⬆️  Đang push lên GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "======================================"
echo "✅ HOÀN TẤT!"
echo "======================================"
echo ""
echo "🌐 Repo của bạn:"
echo "   https://github.com/$username/$reponame"
echo ""
echo "📚 Bước tiếp theo:"
echo "   1. Vào GitHub repo của bạn"
echo "   2. Thêm description"
echo "   3. Add topics: rag, langchain, vietnamese, nlp, streamlit"
echo "   4. Cập nhật README.md với hướng dẫn chi tiết"
echo ""
