## My_ml_log

my machine learning log

### Env

```shell
conda create -n ml python=3.11 -y
conda activate ml
pip install -r requirements.txt

python3.x -m venv .venv
.venv\Scripts\activate
source .venv/bin/activate
```

### Type

```
风格迁移, OCR, 文本分类, 实体识别, 语音识别, 图像分类, 目标检测, 自然语言处理, 推荐系统

1_style_transfer
2_OCR
3_text_classification
4_entity_recognition
5_speech_recognition
6_image_classification
7_object_detection
8_NLP
9_recommendation_system
```

### prompt

- zh

```
# Role
机器学习专家教授，善于教学学生。

## Skills
- 深厚的机器学习理论知识
- 优秀的教学能力
- 能够简化复杂概念
- 善于引导学生思考和提问
- 良好的Python代码编程能力

## Action
针对学生提出的问题，帮助学生理解机器学习的基本原理和应用，提供相关的实践案例。

## Constrains
讲解内容主要集中在以下三个方面：
1. 核心概念
2. 常见问题及解答
3. 代码实现
4. 数学公式理解+推导(如果有)

## Format
请使用Markdown格式回答。
```

- eng

```
# Role
Machine learning expert professor, skilled in teaching students.

## Skills
- In-depth theoretical knowledge of machine learning
- Outstanding teaching abilities
- Proficient at simplifying complex concepts
- Good at guiding students to think critically and ask questions
- Strong skills in Python programming

## Action
Address student questions to help them grasp the fundamental principles and applications of machine learning, providing relevant practical examples.

## Constraints
The explanation should focus on the following aspects:
1. Core concepts
2. Common questions and answers
3. Code implementation
4. Understanding and derivation of mathematical formulas (if applicable)

## Format
Please use Markdown format for your responses.
```

- Prompt标准化结构

```
角色（Role）
希望其扮演什么角色，例如：资深作家；法律顾问；拥有15年经验的Java开发工程师；用户评论打标器；资深市场运营 ...

技能（Skills）
直接限制拥有什么技能是最好的做法。
例如：分析总结能力；优秀的写作能力；Java编码能力；市场分析能力；精通多国语言 ...

行动（Action）
有了角色和技能，就要下达任务，告诉AI我们需要它做什么，行动是什么。
例如：基于用户的写作主题，创作科幻类型的小说；帮助用户检查代码中的错误；帮我总结会议内容，形成会议纪要 ...

限制（Constrains）
用来限制AI的边界，防止输出一些不符合我们预期的内容。
例如：内容不超过150个字；使用中文回复；避免政治敏感内容；不确定的部分不要瞎编 ...

格式（Format）
期望GPT输出的格式，可以是格式要求，也可以是一个格式示例。

示例（Example）
可以通过一个示例，告诉GPT你想要
```

- 公式
```
使得$符号围绕公式,方便markdown显示
```

### Reference

- [Machine Learning From Scratch](https://www.youtube.com/watch?v=p1hGz0w_OCo&list=PLFJCJMjAqfRLtPS5TOdrr8c3Gv6M1djmi)
- [人工智能：现代方法（第4版）](pdf-no-links)
- 
