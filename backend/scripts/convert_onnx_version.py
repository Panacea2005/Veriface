"""
Convert ONNX models for compatibility with onnxruntime:
- IR version: max 11
- Opset version: max 23
"""
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

try:
    import onnx
except ImportError:
    print("Error: Please install onnx: pip install onnx")
    sys.exit(1)

def convert_model(input_path: Path, output_path: Path = None):
    """Convert ONNX model to IR version 11 and opset 23."""
    if output_path is None:
        output_path = input_path
    
    print(f"Loading model: {input_path}")
    model = onnx.load(str(input_path))
    
    current_ir = model.ir_version
    # Get opset version from model
    opset_imports = model.opset_import
    current_opset = None
    for opset in opset_imports:
        if opset.domain == '' or opset.domain == 'ai.onnx':
            current_opset = opset.version
            break
    
    print(f"Current IR version: {current_ir}")
    print(f"Current opset version: {current_opset}")
    
    needs_conversion = False
    
    # Convert IR version if needed (max 11 for onnxruntime compatibility)
    if current_ir > 11:
        model.ir_version = 11
        needs_conversion = True
        print(f"Downgrading IR version: {current_ir} -> 11")
    
    # Convert opset if needed (max 23 for onnxruntime compatibility)
    if current_opset and current_opset > 23:
        for opset in model.opset_import:
            if opset.domain == '' or opset.domain == 'ai.onnx':
                opset.version = 23
                needs_conversion = True
                print(f"Downgrading opset: {current_opset} -> 23")
                break
    
    if not needs_conversion:
        print(f"[OK] Model already compatible (IR: {current_ir}, opset: {current_opset})")
        return
    
    # Validate
    try:
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"[WARN] Model validation warning: {e}")
        print(f"         Continuing anyway...")
    
    # Save
    onnx.save(model, str(output_path))
    print(f"[OK] Converted: {output_path}")

def main():
    models_dir = Path(__file__).parent.parent / "app" / "models"
    
    models = [
        "embedding_A.onnx",
        "embedding_B.onnx",
        "emotion.onnx",
        "liveness.onnx"
    ]
    
    converted = 0
    for model_name in models:
        model_path = models_dir / model_name
        if model_path.exists():
            try:
                convert_model(model_path)
                converted += 1
            except Exception as e:
                print(f"[ERROR] Failed to convert {model_name}: {e}")
        else:
            print(f"[WARN] Model not found: {model_path}")
    
    print(f"\n[OK] Converted {converted} model(s)")

if __name__ == "__main__":
    main()

