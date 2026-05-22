import re
import json
from datetime import datetime

POST_FILE = "./raw_data/dcard_name_raw.json"
COMMENT_FILE_1 = "./raw_data/comments_1.json"
COMMENT_FILE_2 = "./raw_data/comments_2.json"
COMMENT_INFO_FILE = "./outputs/comments_info.json"

url_pattern = re.compile(r'https?://[^\s。，！？；：「」『』（）(),]+')
extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
dcard_domains = [
    'megapx-assets.dcard.tw',
    'sticker-assets.dcard.tw',
    'megapx.dcard.tw'
]
sus_utm = {
    "utm_source=", "utm_medium=", "utm_campaign=", "utm_content=",
    "aff=", "aff_id=", "affiliate=",
    "ref=", "referral=", "referer=", 
    "partner=", "partner_id=", "tag=", 
    "coupon=", "code=", "invite=", "promo=", 
    "tracking_id=", "click_id=", "campaign_id=", 
    "subid=", "sid=", "fbclid=", "gclid=", "igshid=",
}

def fetch_post_author():
    with open(POST_FILE, "r", encoding="utf-8") as f:
        posts = json.load(f)

    mapping = {}

    for post in posts:
        post_id = post.get("id")
        author = post.get("author", {})
        name = author.get("displayName")
        uid = author.get("subtitle")
        post_time = post.get("createdAt")
        mapping[str(post_id)] = (name, uid, post_time)

    return mapping

def count_link(content, author_reply_links, utm_links):
    if not content:
        return author_reply_links, utm_links
        
    all_urls = url_pattern.findall(content)

    for url in all_urls:
        if any(domain in url for domain in dcard_domains):
            continue
        if "i.imgur.com" in url:
            continue
        url_without_query = url.split('?')[0].lower()
        if any(url_without_query.endswith(ext) for ext in extensions):
            continue
            
        author_reply_links += 1
        if any(f"?{utm}" in url or f"&{utm}" in url for utm in sus_utm):
            utm_links += 1

    return author_reply_links, utm_links

def get_author_info(author_name, author_uid, comment, author_replies, author_reply_links, utm_links):
    host = comment.get("host")
    is_author = False

    if host is True:
        is_author = True
    elif host is False:
        is_author = False
    else:
        # 當 host 欄位缺失 (None) 時，執行備援比對
        com_author = comment.get("author", {})
        com_name = com_author.get("displayName")
        com_uid = com_author.get("subtitle")

        if com_uid is not None:
            if com_name == author_name and com_uid == author_uid:
                is_author = True
        else:
            if com_name == author_name and com_name is not None:
                is_author = True

    if is_author:
        author_replies += 1
        author_reply_links, utm_links = count_link(
            comment.get("content", ""), author_reply_links, utm_links
        )

    return author_replies, author_reply_links, utm_links

def parse_time(t_str):
    if not t_str:
        return None
    try:
        return datetime.fromisoformat(t_str.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        return None

def main():
    with open(COMMENT_FILE_1, "r", encoding="utf-8") as f:
        comments = json.load(f)

    with open(COMMENT_FILE_2, "r", encoding="utf-8") as f:
        comments.extend(json.load(f))

    mapping = fetch_post_author()

    all_comments_info = []

    for post_entry in comments:
        for post_id, com_list in post_entry.items():
            author_data = mapping.get(post_id)
            if not author_data:
                continue
            
            author_name, author_uid, post_time_str = author_data
            dt_post = parse_time(post_time_str)

            # 統計數據初始化
            dir_com_count = 0  # 有效主留言總數
            sub_com_count = 0  # 有效子留言總數
            author_replies = 0  # 作者回覆數量
            author_reply_links = 0  # 作者回覆的連結數量
            author_reply_utmlinks = 0  # 作者回覆的 UTM 連結數量
            
            first_main_comment_time = None
            comments_within_30_mins = 0

            for com in com_list:
                if com.get("content") is None or com.get("content") == "":
                    continue

                dir_com_count += 1
                
                # 時間統計
                com_time_str = com.get("createdAt")
                dt_com = parse_time(com_time_str)
                if dt_com:
                    # 僅紀錄第一個主留言的時間差
                    if first_main_comment_time is None or dt_com < first_main_comment_time:
                        first_main_comment_time = dt_com
                    
                    # 主留言 30 分鐘內
                    if dt_post:
                        diff = (dt_com - dt_post).total_seconds()
                        if 0 <= diff <= 1800:
                            comments_within_30_mins += 1

                # 計算作者回覆數量 (主留言)
                author_replies, author_reply_links, author_reply_utmlinks = get_author_info(
                    author_name, author_uid, com, author_replies, author_reply_links, author_reply_utmlinks
                )

                for subcom in com.get("subComments", []):
                    if subcom.get("content") is None or subcom.get("content") == "":
                        continue
                    
                    sub_com_count += 1
                    
                    # 子留言 30 分鐘內
                    subcom_time_str = subcom.get("createdAt")
                    dt_subcom = parse_time(subcom_time_str)
                    if dt_subcom and dt_post:
                        diff = (dt_subcom - dt_post).total_seconds()
                        if 0 <= diff <= 1800:
                            comments_within_30_mins += 1
                    
                    # 計算作者回覆數量 (子留言)
                    author_replies, author_reply_links, author_reply_utmlinks = get_author_info(
                        author_name, author_uid, subcom, author_replies, author_reply_links, author_reply_utmlinks
                    )

            total_comment_count = dir_com_count + sub_com_count

            if first_main_comment_time and dt_post:
                time_diff = (first_main_comment_time - dt_post).total_seconds() / 60
            else:
                time_diff = None

            if total_comment_count > 0:
                sub_depth = sub_com_count / total_comment_count
                ratio_30m = comments_within_30_mins / total_comment_count
                author_reply_ratio = author_replies / total_comment_count
            else:
                sub_depth = 0
                ratio_30m = 0
                author_reply_ratio = 0

            author_reply_link_ratio = (
                author_reply_links / author_replies if author_replies > 0 else 0
            )
            author_reply_utm_link_ratio = (
                author_reply_utmlinks / author_reply_links if author_reply_links > 0 else 0
            )

            all_comments_info.append({
                "id": post_id,
                "commentCount": dir_com_count,
                "totalCommentCount": total_comment_count,
                "subcomDepth": sub_depth,
                "authorReplyCount": author_replies,
                "authorReplyLinkCount": author_reply_links, 
                "authorReplyUTMLinkCount": author_reply_utmlinks,
                "authorReplyRatio": author_reply_ratio,
                "authorReplyLinkRatio": author_reply_link_ratio,
                "authorReplyUTMLinkRatio": author_reply_utm_link_ratio,
                "firstCommentTimeDiff": time_diff,
                "first30MinCommentRatio": ratio_30m,
            })

            break # 處理完一個 post_id 後跳出，因為 post_entry 只包含一個 key

    with open(COMMENT_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(all_comments_info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
