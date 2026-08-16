#!/bin/bash
# 一键下载 W2 公开监管语料（在仓库根目录运行：bash starter/download_corpus.sh）
# 全部为官方/权威公开发布版本，仅用于个人学习 RAG 项目
mkdir -p data
cd data

echo "↓ 下载 个人金融信息保护技术规范 (JR/T 0171—2020, 央行官方PDF)..."
curl -L -o "个人金融信息保护技术规范.pdf" "https://www.pbc.gov.cn/zhengwugongkai/4081330/4406346/4693549/4085091/2020030414554980731.pdf"

echo "↓ 下载 银行保险机构消费者权益保护管理办法 (金融监管总局官方PDF)..."
curl -L -o "银行保险机构消费者权益保护管理办法.pdf" "https://www.nfra.gov.cn/chinese/docfile/2023/e3a74db42cf44727bc233c02b4a7fe0d.pdf"

echo "↓ 下载 商业银行理财业务监督管理办法 (官方正文第三方镜像PDF)..."
curl -L -o "商业银行理财业务监督管理办法.pdf" "https://www.suyinwealth.com/upload/upload/file/202008/14/tnuGXrSpeSERBDBD.pdf"

echo "✅ 完成。请检查 data/ 下是否生成 3 个 PDF，大小非 0。"
