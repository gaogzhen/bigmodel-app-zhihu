import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置显示选项
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 200)

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据集
try:
    df = pd.read_csv('Seattle_Hotels.csv', encoding="UTF-8")
    print(f"成功读取数据集，共 {len(df)} 条记录")
    print(f"数据列：{list(df.columns)}")
except FileNotFoundError:
    print("错误：未找到 'Seattle_Hotels.csv' 文件")
    print("请确保文件在当前目录下或提供正确的文件路径")
    exit(1)

# 检查必需列是否存在
required_columns = ['name', 'desc']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"错误：数据集中缺少必要的列：{missing_columns}")
    exit(1)

# 显示数据基本信息
print("\n=== 数据集基本信息 ===")
print(f"数据集形状：{df.shape}")
print(f"前5条数据：")
print(df.head())
print(f"\n数据列信息：")
print(df.info())

# 创建英文停用词列表
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don',
    "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
    "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn',
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}


# 1. 分析函数：得到酒店描述中n-gram特征中的topK个
def get_top_n_words(corpus, n=1, k=20, title="词频分析"):
    """
    获取语料中最高频的n-gram词组

    参数:
    corpus: 文本语料
    n: n-gram的n值（1=单词，2=双词，3=三词）
    k: 返回的前k个高频词
    title: 图表标题
    """
    if corpus.empty:
        print("错误：语料为空")
        return pd.DataFrame()

    try:
        # 统计ngram词频矩阵，使用自定义停用词列表
        vec = CountVectorizer(ngram_range=(n, n), stop_words=list(ENGLISH_STOPWORDS)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        # 按照词频大小从大到小排序
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        # 创建可视化
        if words_freq:
            top_k_words = words_freq[:k]
            df_words = pd.DataFrame(top_k_words, columns=['word', 'count'])

            # 绘制水平条形图
            plt.figure(figsize=(10, 8))
            df_words.sort_values('count').plot(kind='barh', x='word', y='count',
                                               title=f'{title} - Top {k}个{n}-gram词组')
            plt.xlabel('词频')
            plt.tight_layout()
            plt.show()

            # 打印结果
            print(f"\n=== Top {k}个{n}-gram词组 ===")
            print(df_words)

            return df_words
        else:
            print(f"警告：未找到有效的{n}-gram词组")
            return pd.DataFrame()

    except Exception as e:
        print(f"分析n-gram时出错: {str(e)}")
        return pd.DataFrame()


# 执行n-gram分析
print("\n=== 执行N-gram分析 ===")
for n_value in [1, 2, 3]:
    result_df = get_top_n_words(df['desc'], n=n_value, k=15,
                                title=f"酒店描述{n_value}-gram分析")
    if not result_df.empty:
        print(f"1-{n_value}元语法分析完成")


# 文本预处理
def clean_text(text):
    """
    清洗文本数据

    参数:
    text: 原始文本

    返回:
    清洗后的文本
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""

    try:
        # 全部小写
        text = text.lower()

        # 替换或移除特殊字符的正则表达式
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

        # 空格替换特殊符号，比如标点
        text = REPLACE_BY_SPACE_RE.sub(' ', text)

        # 移除坏掉的字符
        text = BAD_SYMBOLS_RE.sub('', text)

        # 文本去掉停用词
        text = ' '.join(word for word in text.split() if word not in ENGLISH_STOPWORDS)

        return text.strip()
    except Exception as e:
        print(f"清洗文本时出错: {str(e)}")
        return ""


# 对desc标签清洗
print("\n=== 文本清洗 ===")
df['desc_cleaned'] = df['desc'].apply(clean_text)

# 检查清洗结果
print(f"原始描述示例: {df['desc'].iloc[0][:200]}...")
print(f"清洗后示例: {df['desc_cleaned'].iloc[0][:200]}...")

# 设置酒店名称为索引（关键步骤）
print("\n=== 设置索引 ===")
if 'name' in df.columns:
    original_index = df.index.tolist()
    df.set_index('name', inplace=True)
    print(f"已将'name'列设置为索引，原索引已保存")
    print(f"索引设置后DataFrame形状: {df.shape}")
else:
    print("错误：数据集中没有'name'列")
    exit(1)


# 2. 建模和特征提取
class HotelRecommender:
    """酒店推荐系统类"""

    def __init__(self, df, text_column='desc_cleaned'):
        """
        初始化推荐系统

        参数:
        df: 包含酒店信息的DataFrame
        text_column: 用于提取特征的文本列
        """
        self.df = df.copy()
        self.text_column = text_column
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_similarities = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, ngram_range=(1, 3), min_df=0.01):
        """
        训练推荐模型

        参数:
        ngram_range: n-gram范围
        min_df: 最小文档频率（用于过滤稀有词）
        """
        print("\n=== 训练推荐模型 ===")

        if self.text_column not in self.df.columns:
            print(f"错误：文本列'{self.text_column}'不存在")
            return False

        # 检查文本数据
        if self.df[self.text_column].isna().any():
            print("警告：文本列中存在空值，将用空字符串填充")
            self.df[self.text_column] = self.df[self.text_column].fillna('')

        # 使用TF-IDF提取文本特征
        print(f"使用TF-IDF提取文本特征...")
        print(f"参数: ngram_range={ngram_range}, min_df={min_df}")

        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=0.95,  # 避免过于常见的词
            stop_words=list(ENGLISH_STOPWORDS)
        )

        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df[self.text_column])
            self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

            print(f"TF-IDF特征提取完成")
            print(f"特征数量: {len(self.feature_names)}")
            print(f"特征矩阵形状: {self.tfidf_matrix.shape}")

            # 计算酒店之间的余弦相似度
            print("计算余弦相似度矩阵...")
            self.cosine_similarities = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
            print(f"相似度矩阵形状: {self.cosine_similarities.shape}")

            self.is_fitted = True
            print("模型训练完成！")

            # 可视化相似度矩阵
            self.visualize_similarity_matrix()

            return True

        except Exception as e:
            print(f"训练模型时出错: {str(e)}")
            return False

    def visualize_similarity_matrix(self, sample_size=20):
        """可视化相似度矩阵（采样显示）"""
        if self.cosine_similarities is None:
            print("相似度矩阵尚未计算")
            return

        try:
            # 随机选择一部分酒店进行可视化
            if len(self.df) > sample_size:
                import random
                indices = random.sample(range(len(self.df)), min(sample_size, len(self.df)))
                similarity_sample = self.cosine_similarities[np.ix_(indices, indices)]
                hotel_names = [self.df.index[i] for i in indices]
            else:
                similarity_sample = self.cosine_similarities
                hotel_names = self.df.index.tolist()

            plt.figure(figsize=(12, 10))
            sns.heatmap(similarity_sample,
                        xticklabels=hotel_names,
                        yticklabels=hotel_names,
                        cmap='YlOrRd',
                        annot=True if len(hotel_names) <= 15 else False,
                        fmt='.2f',
                        cbar_kws={'label': '相似度'})
            plt.title('酒店相似度矩阵热力图', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"可视化相似度矩阵时出错: {str(e)}")

    def get_hotel_features(self, hotel_name, top_n=10):
        """
        获取酒店的主要TF-IDF特征

        参数:
        hotel_name: 酒店名称
        top_n: 返回前N个最重要的特征

        返回:
        重要特征列表
        """
        if not self.is_fitted:
            print("错误：模型尚未训练")
            return []

        if hotel_name not in self.df.index:
            print(f"错误：酒店 '{hotel_name}' 不存在")
            return []

        try:
            hotel_idx = self.df.index.get_loc(hotel_name)
            feature_array = self.tfidf_matrix[hotel_idx].toarray().flatten()

            # 获取非零特征及其权重
            non_zero_indices = np.where(feature_array > 0)[0]
            if len(non_zero_indices) == 0:
                print(f"酒店 '{hotel_name}' 没有可用的文本特征")
                return []

            # 按权重排序
            features = [(self.feature_names[idx], feature_array[idx])
                        for idx in non_zero_indices]
            features.sort(key=lambda x: x[1], reverse=True)

            print(f"\n=== 酒店 '{hotel_name}' 的主要特征 ===")
            for i, (feature, weight) in enumerate(features[:top_n], 1):
                print(f"{i}. {feature}: {weight:.4f}")

            return features[:top_n]

        except Exception as e:
            print(f"获取酒店特征时出错: {str(e)}")
            return []

    def find_similar_hotels(self, hotel_name, top_n=10, min_similarity=0.0,
                            exclude_self=True, return_scores=True):
        """
        查找与指定酒店相似的酒店

        参数:
        hotel_name: 查询酒店名称
        top_n: 返回的推荐数量
        min_similarity: 最小相似度阈值
        exclude_self: 是否排除自己
        return_scores: 是否返回相似度分数

        返回:
        推荐酒店列表（含相似度分数）
        """
        if not self.is_fitted:
            print("错误：模型尚未训练")
            return [] if not return_scores else []

        if hotel_name not in self.df.index:
            print(f"错误：酒店 '{hotel_name}' 不存在于数据集中")
            print(f"可用酒店数量: {len(self.df)}")
            print("前10个酒店名称:")
            for i, name in enumerate(self.df.index[:10], 1):
                print(f"  {i}. {name}")
            return [] if not return_scores else []

        try:
            # 找到查询酒店的索引
            idx = self.df.index.get_loc(hotel_name)

            # 获取相似度分数
            similarity_scores = list(enumerate(self.cosine_similarities[idx]))

            # 按相似度排序（降序）
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

            # 如果需要排除自己
            start_idx = 1 if exclude_self else 0

            # 筛选结果
            results = []
            for i, (hotel_idx, score) in enumerate(similarity_scores[start_idx:], start=1):
                if score < min_similarity:
                    break

                result_hotel_name = self.df.index[hotel_idx]

                # 获取原始描述片段
                original_desc = ""
                if 'desc' in self.df.columns:
                    desc = self.df.iloc[hotel_idx]['desc']
                    if isinstance(desc, str):
                        original_desc = desc[:150] + "..." if len(desc) > 150 else desc

                if return_scores:
                    results.append({
                        'rank': i,
                        'hotel_name': result_hotel_name,
                        'similarity_score': round(score, 4),
                        'description': original_desc
                    })
                else:
                    results.append(result_hotel_name)

                if len(results) >= top_n:
                    break

            return results

        except Exception as e:
            print(f"查找相似酒店时出错: {str(e)}")
            return [] if not return_scores else []

    def recommend_hotels(self, hotel_name, top_n=10, show_details=True):
        """
        推荐与指定酒店相似的酒店（主要推荐函数）

        参数:
        hotel_name: 查询酒店名称
        top_n: 推荐数量
        show_details: 是否显示详细信息

        返回:
        推荐结果DataFrame
        """
        results = self.find_similar_hotels(hotel_name, top_n, return_scores=True)

        if not results:
            if show_details:
                print(f"\n未找到与 '{hotel_name}' 相似的酒店")
            return pd.DataFrame()

        if show_details:
            print(f"\n{'=' * 60}")
            print(f"推荐结果: 与 '{hotel_name}' 最相似的 {len(results)} 家酒店")
            print(f"{'=' * 60}")

            # 显示查询酒店信息
            if hotel_name in self.df.index:
                print(f"\n查询酒店: {hotel_name}")
                if 'desc' in self.df.columns:
                    desc = self.df.loc[hotel_name, 'desc']
                    if isinstance(desc, str):
                        print(f"描述: {desc[:200]}..." if len(desc) > 200 else f"描述: {desc}")

            print(f"\n推荐列表:")
            for result in results:
                print(f"\n{result['rank']}. {result['hotel_name']}")
                print(f"   相似度: {result['similarity_score']:.2%}")
                if result['description']:
                    print(f"   描述: {result['description']}")

        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.set_index('rank', inplace=True)

        return results_df

    def get_recommendation_reasons(self, hotel1, hotel2, top_features=5):
        """
        获取两个酒店相似的原因（共同特征）

        参数:
        hotel1: 酒店1名称
        hotel2: 酒店2名称

        返回:
        共同特征列表
        """
        if not self.is_fitted:
            print("错误：模型尚未训练")
            return []

        if hotel1 not in self.df.index or hotel2 not in self.df.index:
            print("错误：指定的酒店不存在")
            return []

        try:
            idx1 = self.df.index.get_loc(hotel1)
            idx2 = self.df.index.get_loc(hotel2)

            # 获取两个酒店的TF-IDF向量
            vec1 = self.tfidf_matrix[idx1].toarray().flatten()
            vec2 = self.tfidf_matrix[idx2].toarray().flatten()

            # 找出共同的特征
            common_features = []
            for i in range(len(self.feature_names)):
                if vec1[i] > 0 and vec2[i] > 0:
                    # 使用平均权重
                    avg_weight = (vec1[i] + vec2[i]) / 2
                    common_features.append((self.feature_names[i], avg_weight))

            # 按权重排序
            common_features.sort(key=lambda x: x[1], reverse=True)

            if common_features:
                print(f"\n=== '{hotel1}' 和 '{hotel2}' 的相似特征 ===")
                for i, (feature, weight) in enumerate(common_features[:top_features], 1):
                    print(f"{i}. {feature}: {weight:.4f}")

            return common_features[:top_features]

        except Exception as e:
            print(f"获取推荐原因时出错: {str(e)}")
            return []


# 3. 创建推荐系统实例并训练模型
print("\n" + "=" * 60)
print("创建酒店推荐系统")
print("=" * 60)

recommender = HotelRecommender(df, text_column='desc_cleaned')

# 训练模型
if not recommender.fit(ngram_range=(1, 3), min_df=0.01):
    print("模型训练失败，请检查数据和参数")
    exit(1)

# 4. 测试推荐系统
print("\n" + "=" * 60)
print("测试推荐系统")
print("=" * 60)

# 获取可用酒店列表
available_hotels = df.index.tolist()
print(f"可用酒店数量: {len(available_hotels)}")
print(f"前5个酒店: {available_hotels[:5]}")

# 测试用例
test_cases = [
    "Hilton Seattle Airport & Conference Center",
    "The Bacon Mansion Bed and Breakfast",
    "The Maxwell Hotel"
]

for test_hotel in test_cases:
    if test_hotel in available_hotels:
        print(f"\n{'=' * 60}")
        print(f"测试酒店: {test_hotel}")

        # 获取酒店特征
        recommender.get_hotel_features(test_hotel, top_n=8)

        # 获取推荐
        recommendations_df = recommender.recommend_hotels(test_hotel, top_n=5)

        # 如果找到推荐，显示推荐原因
        if not recommendations_df.empty and len(recommendations_df) > 0:
            top_recommendation = recommendations_df.iloc[0]['hotel_name']
            recommender.get_recommendation_reasons(test_hotel, top_recommendation, top_features=3)
    else:
        print(f"\n测试酒店 '{test_hotel}' 不存在于数据集中")


# 5. 交互式推荐界面
def interactive_recommendation():
    """交互式推荐界面"""
    print("\n" + "=" * 60)
    print("酒店推荐系统 - 交互模式")
    print("=" * 60)
    print("输入酒店名称获取推荐（输入 'quit' 退出）")
    print("输入 'list' 查看所有酒店列表")
    print("输入 'random' 随机选择一个酒店进行推荐")

    while True:
        print("\n" + "-" * 40)
        user_input = input("\n请输入酒店名称: ").strip()

        if user_input.lower() == 'quit':
            print("感谢使用酒店推荐系统！")
            break

        elif user_input.lower() == 'list':
            print(f"\n所有酒店列表（共{len(available_hotels)}家）:")
            for i, hotel in enumerate(available_hotels, 1):
                print(f"{i:3d}. {hotel}")
                if i % 20 == 0:
                    cont = input("显示更多？(按Enter继续，输入'q'返回): ")
                    if cont.lower() == 'q':
                        break

        elif user_input.lower() == 'random':
            import random
            random_hotel = random.choice(available_hotels)
            print(f"\n随机选择的酒店: {random_hotel}")
            recommendations_df = recommender.recommend_hotels(random_hotel, top_n=8)

            if not recommendations_df.empty:
                # 可视化相似度
                fig, ax = plt.subplots(figsize=(10, 6))
                hotels = [random_hotel] + recommendations_df['hotel_name'].tolist()
                scores = [1.0] + recommendations_df['similarity_score'].tolist()

                bars = ax.barh(range(len(hotels)), scores, color='skyblue')
                ax.set_yticks(range(len(hotels)))
                ax.set_yticklabels(hotels)
                ax.set_xlabel('相似度分数')
                ax.set_title(f"与 '{random_hotel}' 的相似度")

                # 在条形图上添加数值标签
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{score:.2%}', va='center')

                plt.tight_layout()
                plt.show()

        else:
            # 查找匹配的酒店
            matching_hotels = [hotel for hotel in available_hotels
                               if user_input.lower() in hotel.lower()]

            if not matching_hotels:
                print(f"未找到包含 '{user_input}' 的酒店")
                print("请尝试输入完整的酒店名称或使用 'list' 查看所有酒店")
                continue

            if len(matching_hotels) == 1:
                target_hotel = matching_hotels[0]
                print(f"找到酒店: {target_hotel}")

                # 显示酒店信息
                if target_hotel in df.index:
                    print(f"\n酒店信息:")
                    for col in ['address', 'desc']:
                        if col in df.columns:
                            value = df.loc[target_hotel, col]
                            if isinstance(value, str):
                                print(f"{col}: {value[:200]}..." if len(value) > 200 else f"{col}: {value}")

                # 获取推荐
                recommendations_df = recommender.recommend_hotels(target_hotel, top_n=10)

                # 保存推荐结果到CSV
                if not recommendations_df.empty:
                    filename = f"recommendations_{target_hotel.replace(' ', '_').replace('/', '_')}.csv"
                    recommendations_df.to_csv(filename, encoding='utf-8')
                    print(f"\n推荐结果已保存到: {filename}")
            else:
                print(f"找到 {len(matching_hotels)} 个匹配的酒店:")
                for i, hotel in enumerate(matching_hotels[:10], 1):
                    print(f"{i}. {hotel}")

                if len(matching_hotels) > 10:
                    print(f"... 还有 {len(matching_hotels) - 10} 个匹配项")

                choice = input("\n请输入编号选择酒店，或输入 'all' 显示所有匹配项: ")
                if choice.isdigit() and 1 <= int(choice) <= len(matching_hotels):
                    target_hotel = matching_hotels[int(choice) - 1]
                    recommendations_df = recommender.recommend_hotels(target_hotel, top_n=8)
                elif choice.lower() == 'all':
                    for i, hotel in enumerate(matching_hotels, 1):
                        print(f"\n{i}. {hotel}")
                        recommender.recommend_hotels(hotel, top_n=3, show_details=True)


# 启动交互式界面
interactive_recommendation()

print("\n=== 酒店推荐系统运行结束 ===")