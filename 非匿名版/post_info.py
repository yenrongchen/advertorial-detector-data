import json
import re
import emoji
from datetime import datetime, timezone, timedelta

POST_FILE = "./raw_data/dcard_name_raw.json"
POST_INFO_FILE = "./outputs/post_info.json"
FORUM = "travel"

def main():
    try:
        with open(POST_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"處理 {len(data)} 篇文章中...")

    article_id = 1
    result = []
    
    url_pattern = re.compile(r'^(https?://[^\s]+)(\s+https?://[^\s]+)*$')
    link_pattern = re.compile(r'https?://[^\s。，！？；：「」『』（）(),]+')
    sus_utm = {
        "utm_source=", "utm_medium=", "utm_campaign=", "utm_content=",
        "aff=", "aff_id=", "affiliate=",
        "ref=", "referral=", "referer=", 
        "partner=", "partner_id=", "tag=", 
        "coupon=", "code=", "invite=", "promo=", 
        "tracking_id=", "click_id=", "campaign_id=", 
        "subid=", "sid=", "fbclid=", "gclid=", "igshid=",
    }

    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    dcard_domains = [
        'megapx-assets.dcard.tw',
        'sticker-assets.dcard.tw',
        'megapx.dcard.tw'
    ]

    for post in data:
        post_id = post.get("id")
        forum = post.get("forumAlias")
        if forum is None:
            print(f"找到一篇缺少論壇別的文章，ID: {post_id}")
            continue
        elif forum != FORUM:
            print(f"找到一篇論壇錯誤的文章，ID: {post_id}, 論壇: {forum}")
            continue

        # 萃取需要的欄位
        item = {
            "id": post_id,
            "articleId": article_id,
            "title": post.get("title", ""),
            "edited": int(post.get("edited", False)),
            "likeCount": int(post.get("likeCount", 0)),
            "collectionCount": int(post.get("collectionCount", 0)),
            "shareCount": int(post.get("shareCount", 0)),
            "forumName": post.get("forumName", ""),
            "forumAlias": forum,
        }

        # 處理 createdAt 時間格式，轉換為台灣時間
        created_at = post.get("createdAt")
        if created_at:
            created_at = str(created_at)
            utc_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            tz_taiwan = timezone(timedelta(hours=8))
            local_dt = utc_dt.astimezone(tz_taiwan)
            item["createdAt"] = local_dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            item["createdAt"] = ""

        # 處理內文
        true_content = ""
        content_raw = post.get("content")
        if content_raw is None or content_raw == "":
            print(f"找到沒有實質內文的文章，ID: {post_id}")
            continue
        elif url_pattern.match(content_raw.strip()):
            meta = post.get("meta", {})
            if "annotation" in meta and meta["annotation"].strip() and not url_pattern.match(meta["annotation"].strip()):
                true_content = meta["annotation"].strip()
                item["content"] = true_content
            else:
                print(f"找到內文只有網址的文章，ID: {post_id}")
                continue
        else:
            true_content = content_raw
            item["content"] = true_content

        # 文本資訊
        text_without_whitespace = re.sub(r"\s+", "", true_content)
        text_len = len(text_without_whitespace)
        if text_len == 0:
            print(f"找到去除空白後沒有實質內文的文章，ID: {post_id}")
            continue
        item["wordCount"] = text_len
        item["lfFreq"] = true_content.count("\n") / text_len
        emoji_count = len(emoji.emoji_list(true_content))
        item["emojiCount"] = emoji_count
        item["emojiPerWord"] = emoji_count / text_len

        # 計算 image 和 video 數量
        media_meta = post.get("mediaMeta")
        unique_images = set()
        unique_videos = set()
        for m in media_meta:
            m_id = m.get("id")
            m_type = m.get("type", "")
            if m_id:
                if "image" in m_type:
                    unique_images.add(m_id)
                elif "video" in m_type:
                    unique_videos.add(m_id)

        image_count = len(unique_images)
        video_count = len(unique_videos)
        
        item["withImages"] = 1 if image_count > 0 else 0
        item["withVideos"] = 1 if video_count > 0 else 0
        item["imageCount"] = image_count
        item["videoCount"] = video_count

        item["imageTextRatio"] = image_count / text_len * 1000
        item["videoTextRatio"] = video_count / text_len * 1000
        item["mediaTextRatio"] = (image_count + video_count) / text_len * 1000

        # 計算內文中的連結數量，排除 Dcard 內部圖片連結和 Imgur 圖片連結
        link_count = 0
        utm_link_count = 0
        all_urls = link_pattern.findall(true_content)

        for url in all_urls:
            if any(domain in url for domain in dcard_domains):
                continue
            url_without_query = url.split('?')[0].lower()
            if any(url_without_query.endswith(ext) for ext in extensions):
                continue
            if "i.imgur.com" in url:
                continue
            
            link_count += 1
            if any(f"?{utm}" in url or f"&{utm}" in url for utm in sus_utm):
                utm_link_count += 1

        item["linksCount"] = link_count
        item["utmLinksCount"] = utm_link_count
        item["linkPerWord"] = link_count / text_len
        item["utmLinkRatio"] = utm_link_count / link_count if link_count > 0 else 0

        # 擷取作者資訊
        author = post.get("author", {})
        item["authorName"] = author.get("displayName")
        item["authorSubtitle"] = author.get("subtitle", "")
        item["authorUseNickname"] = int(post.get("withNickname"))
        item["authorSuspicious"] = int(author.get("isSuspiciousAccount", False))
        item["authorHasCreatorBadge"] = int(post.get("creatorBadge", False))
        item["authorHasOfficialCreatorBadge"] = int(post.get("officialCreatorBadge", False))

        result.append(item)
        article_id += 1

    with open(POST_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已處理完 {len(result)} 篇文章的資訊")


if __name__ == "__main__":
    main()
