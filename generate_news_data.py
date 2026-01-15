import os

# 目标文件夹
output_dir = "docs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 新闻与博客数据集 ---
# 这里的 Key 是文件名（包含英文关键词以便分类），Value 是文章内容
news_data = {
    # ==========================================
    # 类别 1: AI (关键词: learning, neural, intelligence, gpt, python, data, cloud)
    # ==========================================
    "tech_blog_gpt_update.txt": """【科技快讯】OpenAI发布更强大的GPT-5预览版
    OpenAI今天宣布了其生成式AI（Generative AI）模型的最新迭代。新的GPT模型在推理能力和代码生成方面有了显著提升。许多Python开发者表示，这极大地提高了生产力。""",

    "cloud_data_center_news.txt": """【行业分析】云计算（Cloud Computing）基础设施面临新的数据挑战
    随着大数据（Big Data）时代的到来，传统的数据中心正在经历转型。亚马逊AWS和微软Azure正在扩建新的服务器集群，以应对AI训练带来的巨大算力需求。""",

    "python_learning_tutorial.txt": """【博主分享】为什么Python是2026年最值得学习的语言？
    对于初学者来说，Python的语法简洁，生态丰富。无论是从事数据分析（Data Analysis）还是机器学习（Machine Learning），Python都是绕不开的工具。""",

    "deep_learning_breakthrough.txt": """【学术前沿】深度学习（Deep Learning）在医疗影像中的新突破
    研究人员开发了一种新的神经网络（Neural Network）架构，能够比人类医生更早地发现肺部病变。这一人工智能（Intelligence）应用有望挽救数百万人的生命。""",

    "neural_link_update.txt": """【未来科技】脑机接口与神经（Neural）科学的结合
    埃隆·马斯克的Neuralink公司展示了最新的植入式设备。该设备试图通过读取神经信号来实现人脑与计算机的直接交互，这被认为是人工智能（Intelligence）的终极形态之一。""",
    
    "machine_learning_ops.txt": """【DevOps】MLOps：机器学习（Machine Learning）运维的最佳实践
    随着企业将AI模型投入生产环境，如何管理数据（Data）流和模型版本变得至关重要。""",
    
    "data_science_trends.txt": """【职场观察】2026年数据科学（Data Science）就业趋势
    企业对能够处理非结构化数据和构建预测模型的专家需求激增。掌握Python和SQL依然是入行的硬门槛。""",

    "intelligent_robotics.txt": """【机器人】具身智能（Embodied Intelligence）是下一个风口吗？
    将大模型植入机器人大脑，让它们能够理解复杂的自然语言指令，是目前AI领域最热门的研究方向。""",

    # ==========================================
    # 类别 2: 金融科技 (关键词: blockchain, bitcoin, payment, finance, wallet, economy, bank)
    # 这里的关键词需要我们在 app.py 里同步更新！
    # ==========================================
    "fintech_blockchain_revolution.txt": """【金融观察】区块链（Blockchain）技术如何重塑跨境支付？
    传统的SWIFT系统转账慢、费用高。基于区块链技术的分布式账本方案，正在让跨境金融（Finance）交易变得像发邮件一样即时和低成本。""",

    "bitcoin_market_analysis.txt": """【加密货币】比特币（Bitcoin）再次突破历史新高
    受机构投资者入场影响，数字货币市场迎来新一轮牛市。分析师认为，去中心化金融（DeFi）正在成为全球经济（Economy）中不可忽视的力量。""",

    "digital_wallet_payment.txt": """【移动支付】电子钱包（Digital Wallet）正在消灭现金
    在亚洲和非洲，移动支付（Payment）的普及率已经超过了信用卡。手机即银行（Bank）的概念正在改变人们的消费习惯。""",

    "finance_ai_trading.txt": """【量化交易】AI算法正在接管华尔街金融（Finance）市场
    高频交易公司正在利用深度学习预测股票走势。在现代经济（Economy）体系中，毫秒级的速度优势就意味着巨大的利润。""",

    "central_bank_digital_currency.txt": """【政策解读】央行（Bank）数字货币（CBDC）的试点进展
    多国中央银行正在加速研发法定数字货币。这将对现有的商业银行体系和第三方支付（Payment）平台产生深远影响。""",

    "defi_economy_report.txt": """【Web3】去中心化金融与创作者经济（Economy）
    通过智能合约，创作者可以直接获得版税收入，而无需经过中介。这是区块链（Blockchain）技术对传统商业模式的降维打击。""",

    "mobile_payment_security.txt": """【安全警示】如何保护你的电子钱包（Wallet）安全？
    随着移动支付（Payment）的普及，新型诈骗手段层出不穷。专家建议开启生物识别验证并定期更换密码。""",
    
    "global_economy_trends.txt": """【宏观经济】全球经济（Economy）复苏面临的挑战
    通货膨胀和供应链危机依然困扰着各大经济体。金融（Finance）政策的调整将直接影响股市和楼市的走向。""",

    # ==========================================
    # 类别 3: 人文常识 (关键词: history, culture, art, philosophy, literature, civilization, museum)
    # ==========================================
    "history_renaissance_art.txt": """【艺术史】文艺复兴时期的艺术（Art）与科学
    达芬奇不仅是画家，更是科学家。这一时期，人类历史（History）迎来了思想的解放，透视法的发明彻底改变了绘画的表现形式。""",

    "museum_guide_paris.txt": """【旅行指南】卢浮宫博物馆（Museum）必看的镇馆之宝
    除了蒙娜丽莎，卢浮宫还收藏了大量古希腊和古埃及的文物。这些藏品见证了人类文明（Civilization）的辉煌历程。""",

    "philosophy_of_life.txt": """【读书笔记】存在主义哲学（Philosophy）在当代的意义
    在焦虑的现代社会，阅读萨特和加缪的作品或许能给我们带来内心的平静。思考“存在的意义”并非无用之功。""",

    "chinese_culture_tea.txt": """【文化杂谈】中国茶文化（Culture）的演变
    从唐代的煎茶到宋代的点茶，再到明清的泡茶。茶不仅是一种饮品，更蕴含了东方哲学（Philosophy）中“天人合一”的思想。""",

    "literature_nobel_prize.txt": """【文学评论】今年的诺贝尔文学（Literature）奖颁给了谁？
    评委会表彰了这位作家在探索人类苦难与希望方面的深刻洞察力。好的文学作品能够跨越国界和文化（Culture）的隔阂。""",

    "ancient_civilization_egypt.txt": """【考古发现】古埃及文明（Civilization）的未解之谜
    金字塔是如何建造的？法老的诅咒真的存在吗？历史（History）学家和考古学家正在利用新技术揭开这些千古谜题。""",

    "modern_art_exhibition.txt": """【展览回顾】当现代艺术（Art）遇上科技
    这场在当代艺术博物馆（Museum）举办的展览，利用VR技术让观众走进梵高的画作中，体验前所未有的视觉冲击。""",

    "history_silk_road.txt": """【历史回顾】丝绸之路上的文化（Culture）交流
    这条古老的商路不仅运输丝绸和香料，更是东西方哲学（Philosophy）、宗教和艺术交汇的桥梁。""",
}

# 写入文件
count = 0
for filename, content in news_data.items():
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(content)
    count += 1
    print(f"✅ 已发布文章: {filename}")

print(f"\n🎉 成功生成 {count} 篇 博客/新闻 文章！")