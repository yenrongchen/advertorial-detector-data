## 文章特徵

### 社交互動資訊
* `edited`: 是否編輯過
* `commentCount`: 直接回覆數量
* `totalCommentCount`: 總回覆數量 (含子留言)
* `likeCount`: 按讚數
* `collectionCount`: 收藏數
* `shareCount`: 分享數


### 文本資訊
* `wordCount`: 文章字數
* `lfFreq`: 換行符號密度
* `linksCount`: 外部連結數量
* `utmLinksCount`: 帶有 UTM 參數的外部連結數量
* `emojiCount`: Emoji 頻率
* `linkPerWord`: 外部連結數量 / 文章字數
* `utmLinkRatio`: 帶有 UTM 參數的外部連結數量 / 外部連結數量
* `emojiPerWord`: Emoji 數量 / 文章字數
* `withImages`: 是否包含圖片
* `withVideos`: 是否包含影片
* `imageCount`: 圖片數量
* `videoCount`: 影片數量
* `imageTextRatio`: 圖片文字比例 (每千字)
* `videoTextRatio`: 影片文字比例 (每千字)
* `mediaTextRatio`: 媒體文字比例 (每千字)
* `text_length`: 文字長度
* `expected_sentiment_value`: 情緒期望值
* `sentiment_volatility`: 情緒波動度


### 促購詞與語意特徵
* `cat_freq_cta_click`: CTA、導流或要求讀者採取行動的詞彙頻率
* `cat_freq_promotion`: 優惠、折扣、團購或促銷相關詞彙頻率
* `cat_freq_urgency`: 強烈推薦、急迫感或購買誘因相關詞彙頻率
* `cat_freq_disclosure`: 合作、贊助、體驗或商業揭露相關詞彙頻率
* `cat_freq_platform_brand`: 旅遊平台、電商平台、品牌或商業服務相關詞彙頻率
* `cat_freq_recommendation`: 明確推薦語氣相關詞彙頻率
* `cat_freq_lottery`: 抽獎或互動增粉相關詞彙頻率


### 留言特徵
* `authorReplyCount`: 作者回覆數量 (含主留言及子留言)
* `authorReplyLinkCount`: 作者回覆內的外部連結數量
* `authorReplyUTMLinkCount`: 作者回覆內的帶有 UTM 參數的外部連結數量
* `authorReplyRatio`: 作者回覆數量 / 總留言數
* `authorReplyLinkRatio`: 作者回覆內的外部連結數量 / 作者回覆數量
* `authorReplyUTMLinkRatio`: 作者回覆內帶有 UTM 參數的外部連結數量 / 作者回覆內的外部連結數量
* `subcomDepth`: 平均子留言深度 (子留言數量 / 主留言數量)
* `firstCommentTimeDiff`: 第一個主留言的時間和發文時間的時間差
* `first30MinCommentRatio`: 前 30 分鐘內的主、子留言佔總留言比例


### 作者資訊
* `authorUseNickname`: 發文者是否為匿名 (待定) (理論上非匿名版全都是 true，會沒有意義)
* `authorHasCreatorBadge`: 發文者是否有創作者勳章
* `authorHasOfficialCreatorBadge`: 發文者是否有官方創作者勳章
* `authorSuspicious`: 是不是可疑帳號


### 作者資訊及行為特徵
* `timeDiffDays`: 最早與最晚的貼文時間差 (天)
* `postCount`: 歷史總發文數
* `postFreq`: 歷史發文頻率 (篇/天)
* `maxPostsPerDay`: 最大單日發文數
* `maxPostsPer12h`: 最大 12 小時內發文數
* `postIntervalStd`: 發文時間間隔標準差 (天)
* `businessHourRatio`: 平日上班時間發文比率
* `weekendRatio`: 周末發文比率
* `travelPostRatio`: 旅遊版文章佔所有貼文的比例
* `forumVariety`: 所有貼文的看板之分散程度
* `forumEvenness`: 所有貼文的看板之分布均勻度
* `avgLikes`: 貼文平均讚數


### 文字資料 (需特殊處理)
* 文章標題 (`title`)
* 文章內文 (`content`)


#### 註：`edited`、`authorUseNickname`、`withImages`、`withVideos`、`authorHasCreatorBadge`、`authorHasOfficialCreatorBadge` 六個欄位原為布林值，CSV 檔中已被轉換為整數，1 代表 True，0 代表 False
