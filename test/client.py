# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import requests

file_path = 'z_using_files/pics/00006737.jpg'
json_input = {
    "content": file_path,
    "prompt": "以markdown表格形式描述图片"
}
response = requests.post("http://127.0.0.1:8927/predict", json=json_input)
print(f"Status: {response.status_code}\nResponse:\n {response.text}")
