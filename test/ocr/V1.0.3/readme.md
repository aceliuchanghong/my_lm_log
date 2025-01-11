- 下载
```
from modelscope import snapshot_download
model_dir = snapshot_download(
    "RapidAI/RapidTable",
    local_dir="/mnt/data/llch/my_lm_log/no_git_oic/rapid_table_model",
)
```

- 安装

```
pip install rapidocr_onnxruntime
pip install rapid_table

# 基于torch来推理unitable模型
pip install rapid_table[torch] # for unitable inference

# onnxruntime-gpu推理
pip uninstall onnxruntime
pip install onnxruntime-gpu # for onnx gpu inference
```

- 官方示例

```
# 输入
@dataclass
class RapidTableInput:
    model_type: Optional[str] = ModelType.SLANETPLUS.value
    model_path: Union[str, Path, None, Dict[str, str]] = None
    use_cuda: bool = False
    device: str = "cpu"

# 输出
@dataclass
class RapidTableOutput:
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None

# 使用示例
input_args = RapidTableInput(model_type="unitable")
table_engine = RapidTable(input_args)

img_path = 'test_images/table.jpg'
table_results = table_engine(img_path)

print(table_results.pred_html)
```

```
from pathlib import Path

from rapid_table import RapidTable, VisTable
from rapidocr_onnxruntime import RapidOCR
from rapid_table.table_structure.utils import trans_char_ocr_res

# 默认是slanet_plus模型
table_engine = RapidTable()

# 开启onnx-gpu推理
# input_args = RapidTableInput(use_cuda=True)
# table_engine = RapidTable(input_args)

# 使用torch推理版本的unitable模型
# input_args = RapidTableInput(model_type="unitable", use_cuda=True, device="cuda:0")
# table_engine = RapidTable(input_args)

ocr_engine = RapidOCR()
viser = VisTable()

img_path = 'test_images/table.jpg'
ocr_result, _ = ocr_engine(img_path)

# 单字匹配
# ocr_result, _ = ocr_engine(img_path, return_word_box=True)
# ocr_result = trans_char_ocr_res(ocr_result)

table_results = table_engine(img_path, ocr_result)

save_dir = Path("./inference_results/")
save_dir.mkdir(parents=True, exist_ok=True)

save_html_path = save_dir / f"{Path(img_path).stem}.html"
save_drawed_path = save_dir / f"vis_{Path(img_path).name}"

viser(
    img_path,
    table_results.pred_html,
    save_html_path,
    table_results.cell_bboxes,
    save_drawed_path,
    table_results.logic_points,
    save_logic_path,
)
print(table_html_str)
```