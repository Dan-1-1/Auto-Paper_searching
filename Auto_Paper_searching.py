import os
import time
import random
import pandas as pd
from datetime import datetime
import requests
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
from tqdm import tqdm
import re
import zipfile
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ==================== 配置区 ====================
SERPAPI_KEY = "*******************************" # 替换为你的 SerpAPI Key
QUERIES = {
    # ------------------- Google Scholar -------------------
    "scholar": [
        # 1. 核心关键词组合（必须出现）
        '"ICESat-2" ("point cloud" OR "photon cloud" OR lidar OR "laser altimetry")',
        '"ICESat-2" "deep learning" (classification OR segmentation OR registration OR denoising OR bathymetry)',

        # 2. 光子点云 + 深度学习
        '"photon point cloud" "deep learning" (classification OR segmentation OR denoising OR bathymetry)',

        # 3. 卫星激光点云处理
        '"satellite lidar" OR "spaceborne lidar" ("point cloud" OR photon) "deep learning"',

        # 4. 测深（bathymetry）相关
        '"ICESat-2" (bathymetry OR "water depth" OR "underwater topography") "deep learning"',

        # 5. 经典网络在 ICESat‑2 上的应用
        '"ICESat-2" ("PointNet" OR "PointNet++" OR "RandLA-Net" OR "Point Transformer" OR DGCNN OR KPConv)',

        # 6. 噪声去除 / 信号提取
        '"ICESat-2" (denoising OR "signal extraction" OR "photon classification") "deep learning"'
    ],

    # ------------------- arXiv -------------------
    "arxiv": [
        # 1. 基础关键词
        "ICESat-2 point cloud OR photon cloud OR lidar",
        "ICESat-2 deep learning classification OR segmentation OR bathymetry",

        # 2. 光子点云
        "photon point cloud deep learning",

        # 3. 卫星激光雷达
        "satellite lidar OR spaceborne lidar point cloud deep learning",

        # 4. 测深
        "ICESat-2 bathymetry OR water depth deep learning",

        # 5. 网络模型
        "ICESat-2 PointNet OR PointNet++ OR RandLA-Net OR Point Transformer",

        # 6. 去噪 / 分类
        "ICESat-2 denoising OR photon classification deep learning"
    ]
}
MIN_YEAR = 2018                  # ICESat-2 发射年份
MIN_CITATIONS_SCHOLAR = 10       # 放宽引用要求（新领域论文少）
MAX_PAGES = {"scholar": 5, "arxiv": 5}
TOP_N = 100                      # 保留更多候选

# 邮件配置（163 邮箱为例，推荐使用）
EMAIL_SENDER = "********************"          # 替换成你的新 QQ 邮箱
EMAIL_PASSWORD = "********************"        # 替换成你的授权码
EMAIL_RECEIVER = "********************"        # 替换成接收邮箱
# SMTP_SERVER = "smtp.163.com"
# SMTP_PORT = 465

OUTPUT_DIR = "retrieved_papers"
PDF_DIR = os.path.join(OUTPUT_DIR, "pdfs")
ZIP_PATH = os.path.join(OUTPUT_DIR, "pdfs.zip")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# ===============================================

def random_sleep(min_sec=2, max_sec=5):
    time.sleep(random.uniform(min_sec, max_sec))

def safe_filename(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:100]

def download_pdf(url, filepath, retries=3):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=30, stream=True)
            if resp.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in resp.iter_content(1024*1024):
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"  [重试 {attempt+1}] {e}")
            time.sleep(2)
    return False

# ------------------- Google Scholar -------------------
def crawl_google_scholar_api(query, max_pages=3, api_key=None):
    if not api_key: return []
    papers = []
    base_url = "https://serpapi.com/search"
    for page in range(max_pages):
        params = {
            "engine": "google_scholar", "q": query, "hl": "en",
            "start": page * 10, "num": 10, "as_ylo": MIN_YEAR, "api_key": api_key
        }
        try:
            data = requests.get(base_url, params=params, timeout=30).json()
            if "organic_results" not in data: break
            for item in data["organic_results"]:
                try:
                    title = item.get("title", "Unknown")
                    link = item.get("link", "")
                    snippet = item.get("snippet", "")
                    cited = 0
                    if item.get("inline_links", {}).get("cited_by"):
                        txt = item["inline_links"]["cited_by"].get("total", "0")
                        cited = int("".join(filter(str.isdigit, str(txt)))) if txt else 0
                    year = None
                    pub_info = item.get("publication_info", {}).get("summary", "")
                    if isinstance(pub_info, (int, float)): pub_info = str(int(pub_info))
                    elif not isinstance(pub_info, str): pub_info = str(pub_info)
                    for token in pub_info.replace(",", " ").split():
                        if token.isdigit() and 1900 < int(token) < 2100:
                            year = int(token); break
                    if year and year >= MIN_YEAR and cited >= MIN_CITATIONS_SCHOLAR:
                        parts = pub_info.split(" - ", 1)
                        authors = parts[0].strip() if len(parts) > 0 else "Unknown"
                        venue = parts[1].strip() if len(parts) > 1 else ""
                        papers.append({
                            "title": title, "authors": authors, "year": year, "venue": venue,
                            "citations": cited, "abstract": snippet[:500], "pdf_link": link,
                            "source": "Google Scholar"
                        })
                except: continue
            print(f"  Scholar 第 {page+1} 页累计 {len(papers)} 篇")
            time.sleep(2)
        except: break
    return papers

# ------------------- arXiv -------------------
def crawl_arxiv_api(query, max_pages=3, api_key=None):
    if not api_key: return []
    papers = []
    base_url = "https://serpapi.com/search"
    for page in range(max_pages):
        params = {
            "engine": "google_scholar", "q": f"{query} source:arxiv", "hl": "en",
            "start": page * 10, "num": 10, "as_ylo": MIN_YEAR, "api_key": api_key
        }
        try:
            data = requests.get(base_url, params=params, timeout=30).json()
            if "organic_results" not in data: break
            for item in data["organic_results"]:
                try:
                    title = item.get("title", "Unknown")
                    link = item.get("link", "")
                    snippet = item.get("snippet", "")
                    cited = 0
                    if item.get("inline_links", {}).get("cited_by"):
                        txt = item["inline_links"]["cited_by"].get("total", "0")
                        cited = int("".join(filter(str.isdigit, str(txt)))) if txt else 0
                    year = None
                    pub_info = item.get("publication_info", {}).get("summary", "")
                    if isinstance(pub_info, (int, float)): pub_info = str(int(pub_info))
                    elif not isinstance(pub_info, str): pub_info = str(pub_info)
                    for token in pub_info.replace(",", " ").split():
                        if token.isdigit() and 1900 < int(token) < 2100:
                            year = int(token); break
                    if year and year >= MIN_YEAR and cited >= MIN_CITATIONS_SCHOLAR:
                        authors = "Unknown"
                        if pub_info:
                            parts = pub_info.split(" - ", 1)
                            authors = parts[0].strip() if len(parts) > 0 else "Unknown"
                        pdf_link = ""
                        if "arxiv.org/abs/" in link:
                            arxiv_id = link.split("/abs/")[-1].split("?")[0].split("#")[0]
                            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                        elif "arxiv.org/pdf/" in link:
                            pdf_link = link
                        papers.append({
                            "title": title, "authors": authors, "year": year, "venue": "arXiv",
                            "citations": cited, "abstract": snippet[:500], "pdf_link": pdf_link or link,
                            "source": "arXiv"
                        })
                except: continue
            print(f"  arXiv 第 {page+1} 页累计 {len(papers)} 篇")
            time.sleep(2)
        except: break
    return papers

# ------------------- 邮件发送 -------------------
def send_email_with_attachments(csv_path, bib_path, zip_path):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"【点云论文】Top {TOP_N} 高被引论文- {datetime.now().strftime('%Y%m%d')}"

    body = f"附件为 Top {TOP_N} 篇高被引论文。"
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    for file_path in [csv_path, bib_path, zip_path]:
        if not os.path.exists(file_path): continue
        with open(file_path, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(file_path)}')
            msg.attach(part)

    for attempt in range(3):
        try:
            print(f"正在使用 QQ 邮箱发送（第 {attempt+1} 次）...")
            server = smtplib.SMTP("smtp.qq.com", 587, timeout=20)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
            server.quit()
            print(f"邮件发送成功！")
            return
        except Exception as e:
            print(f"失败: {e}")
            time.sleep(3)
    print("发送失败！请检查 QQ 邮箱授权码")

# ------------------- 主函数 -------------------
def main():
    all_papers = []
    print("开始检索 Google Scholar...")
    for q in QUERIES["scholar"]:
        print(f"  查询: {q}")
        all_papers.extend(crawl_google_scholar_api(q, MAX_PAGES["scholar"], SERPAPI_KEY))

    print("\n开始检索 arXiv...")
    for q in QUERIES["arxiv"]:
        print(f"  查询: {q}")
        all_papers.extend(crawl_arxiv_api(q, MAX_PAGES["arxiv"], SERPAPI_KEY))

    clean = []
    for p in all_papers:
        try:
            clean.append({
                "title": str(p.get("title", "Unknown")),
                "authors": str(p.get("authors", "Unknown")),
                "year": int(p.get("year") or 0),
                "venue": str(p.get("venue", "Unknown")),
                "citations": int(p.get("citations") or 0),
                "abstract": str(p.get("abstract", ""))[:500],
                "pdf_link": str(p.get("pdf_link", "")),
                "source": str(p.get("source", "Unknown"))
            })
        except: continue

    if not clean:
        print("未抓到论文！")
        return

    df = pd.DataFrame(clean).drop_duplicates(subset=["title"])
    df = df[df["year"] >= MIN_YEAR]
    df.sort_values(by=["citations", "year"], ascending=[False, False], inplace=True)
    df_top = df.head(TOP_N).copy()

    ts = datetime.now().strftime("%Y%m%d")
    csv_path = os.path.join(OUTPUT_DIR, f"pointcloud_dl_top{TOP_N}_{ts}.csv")
    bib_path = os.path.join(OUTPUT_DIR, f"pointcloud_dl_top{TOP_N}_{ts}.bib")
    df_top.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # BibTeX
    db = BibDatabase()
    for _, row in df_top.iterrows():
        author_str = row["authors"].replace(", ", " and ")
        first = "".join(c for c in row["authors"].split(",")[0].lower() if c.isalnum()) or "unknown"
        entry = {
            "ENTRYTYPE": "misc", "ID": f"{first}{row['year']}", "title": row["title"],
            "author": author_str, "year": str(row["year"]), "url": row["pdf_link"],
            "note": f"[{row['source']}] Citations: {row['citations']}"
        }
        if "arxiv" in row["source"].lower():
            entry["howpublished"] = f"\\url{{{row['pdf_link']}}}"
        else:
            entry["journal"] = row["venue"]
        db.entries.append(entry)
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(BibTexWriter().write(db))

    # 下载 PDF
    print(f"\n开始下载 Top {TOP_N} 论文 PDF...")
    success = 0
    for _, row in tqdm(df_top.iterrows(), total=len(df_top), desc="下载 PDF"):
        url = row["pdf_link"]
        if not url: continue
        safe_title = safe_filename(row["title"])
        safe_auth = safe_filename(row["authors"].split(",")[0])
        filename = f"{row['year']}_{safe_auth}_{safe_title}.pdf"
        filepath = os.path.join(PDF_DIR, filename)
        if os.path.exists(filepath):
            success += 1
            continue
        if download_pdf(url, filepath):
            success += 1
        time.sleep(1)

    # # 打包 PDF
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in os.listdir(PDF_DIR):
            zf.write(os.path.join(PDF_DIR, file), file)

    # print(f"PDF 下载完成：{success}/{len(df_top)} 篇")

    # 发送邮件
    print(f"\n正在发送邮件到 {EMAIL_RECEIVER}...")
    send_email_with_attachments(csv_path, bib_path, ZIP_PATH)

if __name__ == "__main__":
    main()
