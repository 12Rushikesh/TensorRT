from optimum.intel import OVModelForVisualCausalLM
from optimum.intel.openvino.configuration import OVWeightQuantizationConfig
from transformers import AutoProcessor

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"   # or local HF path
OUTPUT_DIR = "qwen3-vl-2b-int8-weightonly-ov"

print("🔹 Exporting Qwen3-VL-2B to OpenVINO INT8 (weight-only)...")

# 🔑 Weight-only quantization config (SAFE)
weight_only_config = OVWeightQuantizationConfig(
    bits=8,            # INT8 weights
    sym=True,          # symmetric quantization
    group_size=-1,     # per-channel (most stable, no accuracy loss)
)

# ✅ Correct class for Vision-Language Models
model = OVModelForVisualCausalLM.from_pretrained(
    MODEL_ID,
    export=True,                       # convert to OpenVINO IR
    trust_remote_code=True,
    quantization_config=weight_only_config,
)

model.save_pretrained(OUTPUT_DIR)

print(f"✅ INT8 weight-only OpenVINO model saved to: {OUTPUT_DIR}")


'''from optimum.intel import OVModelForVisualCausalLM
from optimum.intel.openvino.configuration import OVWeightQuantizationConfig
from transformers import AutoProcessor
import os

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"   # or local path
OUTPUT_DIR = r"E:\ocr\flocr\qwen3-vl-8b-int8-weightonly-ov"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("🔹 Exporting Qwen3-VL-8B to OpenVINO INT8 (weight-only)...")

# 🔑 Weight-only quantization config
weight_only_config = OVWeightQuantizationConfig(
    bits=8,            # INT8 weights
    sym=True,          # symmetric quantization
    group_size=-1,     # per-channel (best stability)
)

# ✅ Load + Export model
model = OVModelForVisualCausalLM.from_pretrained(
    MODEL_ID,
    export=True,                      # convert to OpenVINO IR
    trust_remote_code=True,
    quantization_config=weight_only_config,
    compile=False,                   # IMPORTANT: avoid immediate compile
)

# ✅ Save model
model.save_pretrained(OUTPUT_DIR)

# ✅ Save processor (VERY IMPORTANT)
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
processor.save_pretrained(OUTPUT_DIR)

print(f"✅ INT8 OpenVINO model + processor saved to: {OUTPUT_DIR}")

'''