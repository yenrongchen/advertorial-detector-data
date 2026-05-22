import json
from datetime import datetime, timedelta
import statistics
import math

AUTHOR_FILE = "./raw_data/author_posts.json"
AUTHORS_INFO_FILE = "./outputs/authors_info.json"

def main():
    with open(AUTHOR_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    authors_info = {}

    for author in data:
        uid = author.get("uid")
        posts = author.get("posts", [])
        
        if not posts:
            continue

        likes = 0
        travel_post_count = 0
        forum_counts = {}
            
        timestamps = []
        for post in posts:
            created_at = post.get("createdAt")
            if created_at:
                try:
                    # 原始格式為 UTC (Z)，將其解析為 datetime 物件
                    dt_utc = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    # 轉換為 UTC+8 (台灣時間)
                    dt_tw = dt_utc + timedelta(hours=8)
                    timestamps.append(dt_tw)

                    # 讚數計數
                    likes += post.get("likeCount", 0)

                    # 旅遊版文章計數
                    forum_alias = post.get("forumAlias", "unknown")
                    if forum_alias == "travel":
                        travel_post_count += 1
                    
                    # 統計各看板的發文數
                    forum_counts[forum_alias] = forum_counts.get(forum_alias, 0) + 1

                except ValueError:
                    continue
        
        if timestamps:
            post_count = len(timestamps)

            # 確保時間戳依先後順序排列
            timestamps.sort()
            earliest = timestamps[0]
            latest = timestamps[-1]
            diff = latest - earliest

            diff_days = diff.days + diff.seconds / 86400
            
            # 計算同一天發文數量的最大值 (UTC+8)
            day_counts = {}
            # 計算上班時間發文比率 (09:00~18:00 UTC+8)
            business_posts = 0
            # 計算周末發文比率 (週六與週日 UTC+8)
            weekend_posts = 0
            
            for dt in timestamps:
                # 統計各日期的發文數
                date_str = dt.date().isoformat()
                day_counts[date_str] = day_counts.get(date_str, 0) + 1
                
                # 判斷是否為上班時間 (平日 09:00 至 17:59)
                if dt.weekday() < 5 and 9 <= dt.hour < 18:
                    business_posts += 1
                
                # 判斷是否為周末 (weekday 為 5 代表週六，6 代表週日)
                if dt.weekday() >= 5:
                    weekend_posts += 1
            
            max_posts_per_day = max(day_counts.values()) if day_counts else 0
            
            # 計算最大12小時內發文數
            max_posts_per_12h = 0
            for i in range(post_count):
                count_12h = 1
                for j in range(i + 1, post_count):
                    if (timestamps[j] - timestamps[i]).total_seconds() <= 12 * 3600:
                        count_12h += 1
                    else:
                        break
                if count_12h > max_posts_per_12h:
                    max_posts_per_12h = count_12h
            
            # 計算發文時間間隔的標準差 (單位：天)
            intervals = []
            for i in range(1, post_count):
                # 計算相鄰兩篇貼文之間的時間差（天）
                interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 86400
                intervals.append(interval)
            
            interval_std = 0
            if len(intervals) >= 2:
                # 使用 statistics 模組計算樣本標準差
                interval_std = statistics.stdev(intervals)

            # 計算跨看板分散度
            forum_variety = math.log2(len(forum_counts) + 1)

            # 計算看板分布均勻度 (Pielou's Evenness)
            forum_entropy = 0
            for count in forum_counts.values():
                p = count / post_count
                if p > 0:
                    forum_entropy -= p * math.log2(p)
            
            # 標準化 Entropy 到 0~1 (若 unique_forum 為 1，則 Entropy 必為 0，直接設為 0)
            unique_forum = len(forum_counts)
            normalized_entropy = 0
            if unique_forum > 1:
                normalized_entropy = forum_entropy / math.log2(unique_forum)

            authors_info[uid] = {
                "postCount": post_count,
                "timeDiffDays": diff_days,
                "postFreq": post_count / diff_days if diff_days > 0 else post_count,
                "maxPostsPerDay": max_posts_per_day,
                "maxPostsPer12h": max_posts_per_12h,
                "postIntervalStd": interval_std,
                "businessHourRatio": business_posts / post_count,
                "weekendRatio": weekend_posts / post_count,
                "travelPostRatio": travel_post_count / post_count,
                "forumVariety": forum_variety,
                "forumEvenness": normalized_entropy,
                "avgLikes": likes / post_count,
            }

    with open(AUTHORS_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(authors_info, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
