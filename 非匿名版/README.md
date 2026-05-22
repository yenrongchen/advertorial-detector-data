## 檔案結構 (僅列出重要檔案)

```text
非匿名版/
├── author.py                  ← 爬取作者所有貼文程式
├── author_info.py             ← 整理作者行為特徵程式
├── comment.py                 ← 爬取文章留言程式
├── comment_info.py            ← 整理文章留言特徵程式
├── crawl.py                   ← 爬蟲主程式 (針對非匿名文章)
├── csv_info.py                ← 彙整文章所有資訊程式
├── Features.md                ← 文章特徵表
├── id_mapping_name.json       ← 文章原始 ID 與檔名 ID 的對照表
├── post_info.py               ← 整理文章基本特徵程式
├── post_txt.py                ← 將文章存為文字檔程式
├── README.md                  ← 說明文件 (You are here)
├── outputs/                   
│   ├── data.csv               ← 整合後的特徵資料 (模型輸入資料)
│   ├── data_new.csv           ← 新版整合後的特徵資料 (新版模型輸入資料)
│   └── dcard_name.csv         ← 文章資訊彙整
├── posts/                     ← 存放文章文字檔的資料夾
├── raw_data/                  ← 存放爬取的原始資料
└── record/                    ← 存放爬取進度與相關 ID 紀錄的資料夾
```

<br>

## CSV 欄位解釋

### `dcard_name.csv`

| 欄位名稱 | 說明 |
| :--- | :--- |
| `id` | Dcard 原始文章 ID |
| `articleId` | `posts/` 資料夾下的檔名 ID |
| `title` | 文章標題 |
| `content` | 文章內文 |
| `createdAt` | 發表時間 |

### `data.csv`
請見 [Features.md](Features.md)

<br>

## 查看原始貼文方式

1. posts 資料夾裡面所有文章的檔名都是一個整數 ID (以下稱為「文章檔名 ID」)
2. 前往 [id_mapping_name.json](id_mapping_name.json)，裡面紀錄的格式如下： 
   ```json
   {
      "文章原始ID": 文章檔名 ID,
   }
   ```
3. 用文章檔名找到文章原始 ID
4. 原始貼文的網址就會是 https://www.dcard.tw/f/travel/p/ 加上「文章原始 ID」

### 範例：
* 根據 id_mapping.json，1.txt 對應到的原始文章 ID 是 261406898
* 原始貼文的網址就會是 https://www.dcard.tw/f/travel/p/261406898

<br>

## 爬蟲執行步驟

1. 終端機輸入 
   ```bash
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome-debug"
   ```
2. 在自動開啟的瀏覽器進入 Dcard，登入 (如果未登入)，再前往旅遊板 (或任意想爬的板)
3. 執行
   ```
   python crawl.py
   ```
