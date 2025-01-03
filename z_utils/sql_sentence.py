create_rule_table_sql = """
CREATE TABLE IF NOT EXISTS rule_info (
  rule_name TEXT, 
  entity_name TEXT, 
  entity_format TEXT, 
  entity_regex_pattern TEXT, 
  entity_order TEXT, 
  rule_state TEXT, 
  latest_modified_insert TEXT, 
  remark TEXT
)
"""

select_rule_sql = """
SELECT 
  entity_name,
  entity_format,
  entity_regex_pattern,
  entity_order,
  rule_state,
  latest_modified_insert,
  remark 
FROM rule_info
WHERE 
  rule_name = ? 
  order by latest_modified_insert desc
"""

insert_rule_sql = """
INSERT INTO rule_info (
  rule_name, 
  entity_name, 
  entity_format, 
  entity_regex_pattern, 
  entity_order,
  rule_state, 
  latest_modified_insert, 
  remark
) 
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

delete_rule_sql = """
DELETE from rule_info 
WHERE rule_name = ?
"""

select_all_rule_name_sql = """
SELECT distinct
  rule_name
FROM rule_info
"""

create_entity_info_sql = """
CREATE TABLE IF NOT EXISTS entity_info (
  rule_name TEXT, 
  original_file_name TEXT, 
  new_file_name TEXT, 
  entity_name TEXT, 
  result TEXT, 
  latest_modified_insert TEXT, 
  remark TEXT
)
"""
# 删除语句
delete_entity_info_sql = """
DELETE FROM entity_info 
WHERE rule_name = ? AND original_file_name = ?
"""

# 插入语句
insert_entity_info_sql = """
INSERT INTO entity_info (
  rule_name, 
  original_file_name, 
  new_file_name, 
  entity_name, 
  result, 
  latest_modified_insert, 
  remark
) 
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

# 查询语句
select_entity_info_sql = """
SELECT 
  rule_name, 
  original_file_name, 
  new_file_name, 
  entity_name, 
  result, 
  latest_modified_insert, 
  remark
FROM entity_info
WHERE rule_name = ? AND original_file_name = ?
"""

select_rule_file_name_sql = """
SELECT distinct
  original_file_name
FROM entity_info
WHERE rule_name = ? 
"""
