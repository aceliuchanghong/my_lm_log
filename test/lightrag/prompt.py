GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["组织架构", "人物", "地区", "事件"]

PROMPTS[
    "entity_extraction"
] = """
{{
  "Goal": "给定一个可能与该活动相关的文本文档和一个实体类型列表,从文本中识别出所有这些类型的实体以及它们之间的关系。",
  "Steps": [
    {{
      "Step": 1,
      "Description": "识别所有实体。对于每个识别出的实体,提取以下信息:",
      "Details": [
        {{
          "entity_name": "实体名称,使用与输入文本相同的语言。如果是英文,则首字母大写。",
          "entity_type": "以下类型之一:[{entity_types}]",
          "entity_description": "实体属性和活动的全面描述"
        }}
      ],
      "Format": "将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)"
    }},
    {{
      "Step": 2,
      "Description": "从步骤1中识别的实体中,识别所有明显相关的(source_entity, target_entity)对。",
      "Details": [
        {{
          "source_entity": "源实体的名称,如步骤1中所识别",
          "target_entity": "目标实体的名称,如步骤1中所识别",
          "relationship_description": "解释为什么你认为源实体和目标实体是相关的",
          "relationship_strength": "表示源实体和目标实体之间关系强度的数值分数",
          "relationship_keywords": "一个或多个高层次的关键词,总结关系的总体性质,重点关注概念或主题,而不是具体细节"
        }}
      ],
      "Format": "将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)"
    }},
    {{
      "Step": 3,
      "Description": "识别总结整个文本主要概念、主题或话题的高层次关键词。这些应该捕捉文档中存在的总体思想。",
      "Format": "将内容级别的关键词格式化为 ("content_keywords"{tuple_delimiter}<high_level_keywords>)"
    }},
    {{
      "Step": 4,
      "Description": "以英文返回输出,作为步骤1和步骤2中识别的所有实体和关系的单一列表。使用 **{record_delimiter}** 作为列表分隔符。",
      "Format": "将输出格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>){record_delimiter}("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)"
    }},
    {{
      "Step": 5,
      "Description": "完成后,输出 {completion_delimiter}"
    }}
  ],
  "Examples": [
    {{
      "Example": 1,
      "Entity_types": ["person", "technology", "mission", "organization", "location"],
      "Text": "while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.",
      "Output": [
        ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter},
        ("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter},
        ("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter},
        ("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter},
        ("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter},
        ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter},
        ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter},
        ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter},
        ("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter},
        ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter},
        ("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter},
      ]
    }},
    {{
      "Example": 2,
      "Entity_types": ["person", "technology", "mission", "organization", "location"],
      "Text": "They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.",
      "Output": [
        ("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter},
        ("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter},
        ("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter},
        ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter},
        ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter},
        ("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter},
      ]
    }},
    {{
      "Example": 3,
      "Entity_types": ["person", "role", "technology", "organization", "event", "location", "concept"],
      "Text": "their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.",
      "Output": [
        ("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter},
        ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter},
        ("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter},
        ("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter},
        ("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter},
        ("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter},
        ("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter},
        ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter},
        ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter},
        ("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter},
        ("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter},
      ]
    }}
  ],
  "Real Data": {{
    "Entity_types": "{entity_types}",
    "Text": "{input_text}"
  }}
}}
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """
{{
    "Role": "你是一个有帮助的助手,负责生成下面提供数据的全面总结。",
    "Goal": "给定一个或两个实体,以及与之相关的描述列表。请将所有这些内容合并成一个全面的描述。确保包含所有描述中的信息。如果提供的描述存在矛盾,请解决这些矛盾并提供一个连贯的总结。确保以第三人称书写,并包含实体名称以便我们了解完整上下文。",
    "Data Format": {{
      "实体": "{entity_name}",
      "描述列表": "{description_list}"
    }},
    "Instruction": "生成一个全面的总结。"
}}
"""


PROMPTS[
    "entiti_continue_extraction"
] = """在上次提取中遗漏了许多实体。请使用相同的格式将它们添加到下面:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """如果还有需要添加的实体,请回答YES或NO。
"""

PROMPTS["fail_response"] = "对不起,我无法回答这个问题。"

PROMPTS[
    "rag_response"
] = """
{{
  "Role": "你是一个有帮助的助手,负责回答关于提供表格中数据的问题。",
  "Goal": "生成符合目标长度和格式的响应,回答用户的问题,总结输入数据表中适合响应长度和格式的所有信息,并结合任何相关的常识。如果你不知道答案,直接说不知道。不要编造任何内容。不要包含没有提供支持证据的信息。",
  "Format": "{response_type}",
  "Data Tables": "{context_data}",
  "Others": "根据长度和格式,在响应中适当添加章节和评论。使用Markdown样式。"
}}
"""

PROMPTS[
    "keywords_extraction"
] = """
{{
  "Role": "你是一个有帮助的助手,任务是识别用户查询中的高层次和低层次关键词。",
  "Goal": "根据查询,列出高层次和低层次关键词。高层次关键词关注总体概念或主题,而低层次关键词关注具体实体、细节或具体术语。",
  "Instructions": [
    "以JSON格式输出关键词。",
    "JSON应包含两个键:",
    "- 'high_level_keywords' 用于总体概念或主题。",
    "- 'low_level_keywords' 用于具体实体或细节。"
  ],
  "Examples": [
    {{
      "Example 1": {{
        "Query": "国际贸易如何影响全球经济稳定？",
        "Output": {{
          "high_level_keywords": ["国际贸易", "全球经济稳定", "经济影响"],
          "low_level_keywords": ["贸易协议", "关税", "货币兑换", "进口", "出口"]
        }}
      }}
    }},
    {{
      "Example 2": {{
        "Query": "森林砍伐对生物多样性有哪些环境影响？",
        "Output": {{
          "high_level_keywords": ["环境影响", "森林砍伐", "生物多样性丧失"],
          "low_level_keywords": ["物种灭绝", "栖息地破坏", "碳排放", "雨林", "生态系统"]
        }}
      }}
    }},
    {{
      "Example 3": {{
        "Query": "教育在减少贫困中扮演什么角色？",
        "Output": {{
          "high_level_keywords": ["教育", "贫困减少", "社会经济发展"],
          "low_level_keywords": ["学校入学", "识字率", "职业培训", "收入不平等"]
        }}
      }}
    }}
  ],
  "Real Data":{{
    "Query": "{query}",
    "Output": ""
  }}
}}
"""

PROMPTS[
    "naive_rag_response"
] = """
{{
  "Role": "你是一个有帮助的助手,负责回答关于提供文档的问题。",
  "Goal": "生成一个符合目标长度和格式的响应,回答用户的问题,总结输入数据表中适合响应长度和格式的所有信息,并结合任何相关的常识。如果你不知道答案,直接说不知道。不要编造任何内容。不要包含支持证据未提供的信息。",
  "Format": "{response_type}",
  "Documents": "{content_data}",
  "Others": "根据长度和格式,在响应中适当添加章节和评论。使用Markdown格式回答输出。"
}}
"""
